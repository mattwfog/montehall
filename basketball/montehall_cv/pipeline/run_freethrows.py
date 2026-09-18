"""Free-throw classification of shot windows — geometry-first, no VLM.

run_shots' adjudicator answers "was a ball shot toward this basket?" — a
free throw passes that test, so without this stage every FT lands in the
box score as a made-or-missed FGA2 (two points for one, inflated field
goal counts). The fix is a geometric signature no live-play shot has:

    (a) the shooter's last control sits at the free-throw point
    (b) the shooter is stationary through the preceding seconds
    (c) other players are parked along the lane edges (the rebounding
        lineup) instead of moving through open play

Each attempt gets an is_ft verdict with its component features as the
audit trail. Confidence = fraction of criteria met, so downstream can
gate harder if precision needs it.

Output: <out>/<job_id>/ft_events/ (one row per shot_events attempt).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from montehall_cv.court import NCAA_LENGTH_FT
from montehall_cv.store.artifacts import ArtifactWriter, read_stage, stage_complete
from montehall_cv.zones import FT_LINE_FROM_BASELINE_FT, RIM_Y_FT

FT_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("event_id", pa.int32()),  # fk into shot_events
        pa.field("is_ft", pa.bool_()),
        pa.field("shooter_entity", pa.int32(), nullable=True),
        pa.field("dist_to_ft_point_ft", pa.float32(), nullable=True),
        pa.field("max_disp_ft", pa.float32(), nullable=True),
        pa.field("lane_lineup_count", pa.int32()),
        pa.field("confidence", pa.float32()),
    ]
)

FT_POINT_RADIUS_FT = 5.0
STATIONARY_WINDOW_MS = 2500
STATIONARY_MAX_DISP_FT = 4.0
LANE_LINEUP_MIN = 3
# lane-edge bands where the rebounding lineup stands: just outside the
# 12ft lane, between the baseline and the FT line
LANE_BAND_Y_FT = (5.5, 9.5)  # |y - 25| within this band
SHOOTER_LOOKBACK_MS = 2500  # run_boxscore.SHOOTER_LOOKBACK_MS


def ft_point(court_end: str) -> tuple[float, float]:
    x = (
        FT_LINE_FROM_BASELINE_FT
        if court_end == "left"
        else NCAA_LENGTH_FT - FT_LINE_FROM_BASELINE_FT
    )
    return (x, RIM_Y_FT)


def classify_window(
    shot: dict,
    shooter: dict | None,
    shooter_track: list[tuple[int, float, float]],
    lineup_positions: list[tuple[float, float]],
) -> dict:
    """FT verdict for one attempt from its geometric features.

    shooter_track: (ts_ms, x, y) samples of the shooter entity in the
    STATIONARY_WINDOW_MS before the shot. lineup_positions: court (x, y)
    of every OTHER entity around the window start.
    """
    court_end = shot["court_end"]
    dist = None
    max_disp = None
    if shooter is not None and court_end in ("left", "right"):
        fx, fy = ft_point(court_end)
        dist = float(np.hypot(shooter["court_x"] - fx, shooter["court_y"] - fy))
        if len(shooter_track) >= 2:
            xs = np.array([p[1] for p in shooter_track])
            ys = np.array([p[2] for p in shooter_track])
            max_disp = float(
                np.hypot(xs - xs[0], ys - ys[0]).max()
            )
    lineup = _lane_lineup_count(lineup_positions, court_end)

    criteria = [
        dist is not None and dist <= FT_POINT_RADIUS_FT,
        max_disp is not None and max_disp <= STATIONARY_MAX_DISP_FT,
        lineup >= LANE_LINEUP_MIN,
    ]
    confidence = sum(criteria) / len(criteria)
    return {
        "is_ft": all(criteria),
        "shooter_entity": shooter["entity_id"] if shooter else None,
        "dist_to_ft_point_ft": round(dist, 2) if dist is not None else None,
        "max_disp_ft": round(max_disp, 2) if max_disp is not None else None,
        "lane_lineup_count": lineup,
        "confidence": round(confidence, 3),
    }


def _lane_lineup_count(
    positions: list[tuple[float, float]], court_end: str | None
) -> int:
    if court_end not in ("left", "right"):
        return 0
    count = 0
    for x, y in positions:
        from_baseline = x if court_end == "left" else NCAA_LENGTH_FT - x
        if (
            0 <= from_baseline <= FT_LINE_FROM_BASELINE_FT
            and LANE_BAND_Y_FT[0] <= abs(y - RIM_Y_FT) <= LANE_BAND_Y_FT[1]
        ):
            count += 1
    return count


def run(out_root: Path, job_id: str) -> dict:
    job_dir = out_root / job_id
    if stage_complete(job_dir / "ft_events"):
        return {"job_id": job_id, "skipped": True, "reason": "stage already complete"}
    started = time.monotonic()

    shots = [
        e for e in read_stage(job_dir / "shot_events").to_pylist() if e["verdict_attempt"]
    ]
    controls = sorted(
        read_stage(job_dir / "ball_controls").to_pylist(), key=lambda r: r["ts_ms"]
    )
    positions = [
        p
        for p in read_stage(job_dir / "positions").to_pylist()
        if p["entity_id"] is not None
    ]

    writer = ArtifactWriter(job_dir / "ft_events", FT_EVENTS_SCHEMA)
    n_ft = 0
    for shot in shots:
        shooter = _last_control_before(controls, shot["ts_start_ms"], SHOOTER_LOOKBACK_MS)
        track = []
        if shooter is not None:
            t0 = shot["ts_start_ms"] - STATIONARY_WINDOW_MS
            track = sorted(
                (p["ts_ms"], p["court_x"], p["court_y"])
                for p in positions
                if p["entity_id"] == shooter["entity_id"]
                and t0 <= p["ts_ms"] <= shot["ts_start_ms"]
            )
        verdict = classify_window(shot, shooter, track, _dedupe_entities(
            positions, shot["ts_start_ms"],
            shooter["entity_id"] if shooter else None,
        ))
        n_ft += int(verdict["is_ft"])
        writer.add({"job_id": job_id, "event_id": shot["event_id"], **verdict})
    writer.close()

    return {
        "job_id": job_id,
        "attempts": len(shots),
        "free_throws": n_ft,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def _dedupe_entities(
    positions: list[dict], ts_ms: int, shooter_entity: int | None
) -> list[tuple[float, float]]:
    """One representative court position per non-shooter entity near ts."""
    best: dict[int, tuple[int, float, float]] = {}
    for p in positions:
        gap = abs(p["ts_ms"] - ts_ms)
        if gap > 500 or p["entity_id"] == shooter_entity:
            continue
        if p["entity_id"] not in best or gap < best[p["entity_id"]][0]:
            best[p["entity_id"]] = (gap, p["court_x"], p["court_y"])
    return [(x, y) for _, x, y in best.values()]


def _last_control_before(controls: list[dict], ts_ms: int, window_ms: int) -> dict | None:
    eligible = [c for c in controls if ts_ms - window_ms <= c["ts_ms"] <= ts_ms]
    return eligible[-1] if eligible else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.job_id), indent=2))


if __name__ == "__main__":
    main()
