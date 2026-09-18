"""Play-generation payloads: one consumable JSON object per possession.

The join layer over the atomic substrate — each play carries its
participants (entity grain + late-bound jersey meta), court-frame
trajectories (downsampled to <=4 Hz), zone entries relative to the
attacked basket, the atomic events in its span, and any shot events —
enough to drive play diagrams, animation, or an LLM description without
touching pixels.

Runs after the LLM block when shot evidence exists (attacked ends bind
via zones.derive_attacked_ends); degrades cleanly without it (trajectories
and events, zones null) so harvest-mode runs still produce plays.

Output: <out>/<job_id>/plays/ (PLAYS_SCHEMA: payload_json per play).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from montehall_cv.store.artifacts import ArtifactWriter, read_stage, stage_complete
from montehall_cv.store.records import DetClass, possession_offense_cluster
from montehall_cv.store.schemas import PLAYS_SCHEMA
from montehall_cv.zones import derive_attacked_ends, is_three, zone_of

TRAJECTORY_MIN_GAP_MS = 250  # <=4 Hz per participant
SHOT_ATTACH_SLACK_MS = 2000  # traces.py uses the same slack around a span
SHOOTER_LOOKBACK_MS = 2500  # run_boxscore.SHOOTER_LOOKBACK_MS
MIN_JERSEY_PROB = 0.6


def run(out_root: Path, job_id: str) -> dict:
    job_dir = out_root / job_id
    if stage_complete(job_dir / "plays"):
        return {"job_id": job_id, "skipped": True, "reason": "stage already complete"}
    started = time.monotonic()

    possessions = read_stage(job_dir / "possessions").to_pylist()
    positions = read_stage(job_dir / "positions").to_pylist()
    controls = sorted(
        read_stage(job_dir / "ball_controls").to_pylist(), key=lambda r: r["ts_ms"]
    )
    atomic = read_stage(job_dir / "atomic_events").to_pylist()
    shots = (
        [e for e in read_stage(job_dir / "shot_events").to_pylist() if e["verdict_attempt"]]
        if stage_complete(job_dir / "shot_events")
        else []
    )
    ft_of = (
        {r["event_id"]: bool(r["is_ft"])
         for r in read_stage(job_dir / "ft_events").to_pylist()}
        if stage_complete(job_dir / "ft_events")
        else {}
    )
    for shot in shots:
        shot["is_ft"] = ft_of.get(shot["event_id"], False)
    jerseys = _jersey_by_entity(job_dir)
    attacked = _attacked_ends(shots, controls)

    writer = ArtifactWriter(job_dir / "plays", PLAYS_SCHEMA)
    n_with_end = 0
    for possession in possessions:
        payload = _play_payload(
            possession, positions, controls, atomic, shots, jerseys, attacked
        )
        if payload["attacked_end"] is not None:
            n_with_end += 1
        writer.add(
            {
                "job_id": job_id,
                "play_id": possession["possession_id"],
                "payload_json": json.dumps(payload, separators=(",", ":")),
            }
        )
    writer.close()

    return {
        "job_id": job_id,
        "plays": len(possessions),
        "plays_with_attacked_end": n_with_end,
        "attacked_ends": {str(k): v for k, v in attacked.items()},
        "shot_events_joined": len(shots),
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def _attacked_ends(shots: list[dict], controls: list[dict]) -> dict[int, str | None]:
    """Shot court_end evidence grouped by the shooter's team -> end binding."""
    ends_by_team: dict[int, list[str]] = defaultdict(list)
    for shot in shots:
        shooter = _last_control_before(
            controls, shot["ts_start_ms"], SHOOTER_LOOKBACK_MS
        )
        if shooter is not None and shot["court_end"] is not None:
            ends_by_team[shooter["team_cluster"]].append(shot["court_end"])
    if not ends_by_team:
        return {}
    return derive_attacked_ends(dict(ends_by_team))


def _play_payload(
    possession: dict,
    positions: list[dict],
    controls: list[dict],
    atomic: list[dict],
    shots: list[dict],
    jerseys: dict[int, str],
    attacked: dict[int, str | None],
) -> dict:
    span = (possession["ts_start_ms"], possession["ts_end_ms"])
    team = possession_offense_cluster(possession)
    end = attacked.get(team) if team is not None else None

    span_positions = [p for p in positions if span[0] <= p["ts_ms"] <= span[1]]
    trajectories = _trajectories(span_positions)
    participants = sorted(
        {
            p["entity_id"]
            for p in span_positions
            if p["entity_id"] is not None
        }
    )
    zone_entries = (
        {str(eid): _zone_entries(points, end) for eid, points in trajectories.items() if eid != "ball"}
        if end is not None
        else None
    )

    events = []
    for e in atomic:
        if e["possession_id"] == possession["possession_id"] or (
            span[0] <= e["ts_ms"] <= span[1]
        ):
            events.append(
                {
                    "type": e["event_type"],
                    "ts_ms": e["ts_ms"],
                    "entity_id": e["entity_id"],
                    "team_cluster": e["team_cluster"],
                    "court_x": e["court_x"],
                    "court_y": e["court_y"],
                    "detail": json.loads(e["payload_json"]) if e["payload_json"] else None,
                }
            )
    for shot in shots:
        if span[0] - SHOT_ATTACH_SLACK_MS <= shot["ts_start_ms"] <= span[1] + SHOT_ATTACH_SLACK_MS:
            shooter = _last_control_before(
                controls, shot["ts_start_ms"], SHOOTER_LOOKBACK_MS
            )
            shot_end = shot["court_end"]
            events.append(
                {
                    "type": "shot",
                    "ts_ms": shot["ts_start_ms"],
                    "entity_id": shooter["entity_id"] if shooter else None,
                    "team_cluster": shooter["team_cluster"] if shooter else None,
                    "court_x": shooter["court_x"] if shooter else None,
                    "court_y": shooter["court_y"] if shooter else None,
                    "detail": {
                        "made": bool(shot["verdict_made"]),
                        "is_ft": shot.get("is_ft", False),
                        "court_end": shot_end,
                        "confidence": shot["confidence"],
                        "is_three": (
                            is_three(shooter["court_x"], shooter["court_y"], shot_end)
                            if shooter is not None and shot_end in ("left", "right")
                            else None
                        ),
                        "zone": (
                            zone_of(shooter["court_x"], shooter["court_y"], shot_end)
                            if shooter is not None and shot_end in ("left", "right")
                            else None
                        ),
                    },
                }
            )
    events.sort(key=lambda e: e["ts_ms"])

    return {
        "play_id": possession["possession_id"],
        "team_cluster": team,
        "attacked_end": end,
        "start_ms": span[0],
        "end_ms": span[1],
        "start_frame": possession["frame_start"],
        "end_frame": possession["frame_end"],
        "duration_s": round((span[1] - span[0]) / 1000, 1),
        "outcome": possession["outcome"],
        "participants": [
            {
                "entity_id": eid,
                "team_cluster": _participant_team(eid, span_positions),
                "jersey": jerseys.get(eid),
            }
            for eid in participants
        ],
        "trajectories": trajectories,
        "zone_entries": zone_entries,
        "events": events,
    }


def _trajectories(span_positions: list[dict]) -> dict[str, list[list[float]]]:
    """Per-participant [ts_ms, x, y] series, downsampled to >=250ms gaps.
    The ball's ungrouped samples ride under the "ball" key."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(span_positions, key=lambda r: r["ts_ms"]):
        key = "ball" if p["cls"] == int(DetClass.BALL) else str(p["entity_id"])
        grouped[key].append(p)

    out: dict[str, list[list[float]]] = {}
    for key, rows in grouped.items():
        series: list[list[float]] = []
        last_ts: int | None = None
        for row in rows:
            if last_ts is not None and row["ts_ms"] - last_ts < TRAJECTORY_MIN_GAP_MS:
                continue
            series.append(
                [row["ts_ms"], round(row["court_x"], 1), round(row["court_y"], 1)]
            )
            last_ts = row["ts_ms"]
        out[key] = series
    return out


def _zone_entries(points: list[list[float]], end: str) -> list[dict]:
    """Zone transitions along a downsampled trajectory (consecutive-distinct)."""
    entries: list[dict] = []
    prev_zone: str | None = None
    for ts_ms, x, y in points:
        zone = zone_of(x, y, end)
        if zone != prev_zone:
            entries.append({"ts_ms": int(ts_ms), "zone": zone})
            prev_zone = zone
    return entries


def _participant_team(entity_id: int, span_positions: list[dict]) -> int | None:
    for p in span_positions:
        if p["entity_id"] == entity_id:
            return p["team_cluster"]
    return None


def _jersey_by_entity(job_dir: Path) -> dict[int, str]:
    """Best strong posterior per entity — perception-grain meta (the
    roster-filtered attribution grain lives in run_boxscore)."""
    entities = {
        r["track_id"]: r for r in read_stage(job_dir / "entities").to_pylist()
    }
    best: dict[int, tuple[str, float]] = {}
    for row in read_stage(job_dir / "identity").to_pylist():
        entity = entities.get(row["track_id"])
        if entity is None or row["prob"] < MIN_JERSEY_PROB:
            continue
        eid = entity["entity_id"]
        if eid not in best or row["prob"] > best[eid][1]:
            best[eid] = (row["candidate"], row["prob"])
    return {eid: jersey for eid, (jersey, _) in best.items()}


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
