"""Possession segmentation v0: ball-control runs in court space.

FIBA's possession definition needs shot/rebound events to close properly;
v0 segments the observable core — runs of same-team ball control — from
existing artifacts only (no new GPU work). Ball coverage is sparse
(occlusion), so control is assigned at ball observations (nearest on-court
player within a radius) and bridged across gaps up to a time bound.

Output: <out>/<job_id>/possessions/ (Possession rows, outcome=None at v0).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from montehall_cv.store.artifacts import ArtifactWriter, read_stage, stage_complete
from montehall_cv.store.records import DetClass
from montehall_cv.store.schemas import POSSESSIONS_SCHEMA

CONTROL_RADIUS_FT = 6.0
MAX_BRIDGE_MS = 4000
MIN_POSSESSION_MS = 2000


def run(out_root: Path, job_id: str) -> dict:
    job_dir = out_root / job_id
    if stage_complete(job_dir / "possessions"):
        return {"job_id": job_id, "skipped": True, "reason": "stage already complete"}
    started = time.monotonic()

    team_of = {
        row["track_id"]: row["team_cluster"]
        for row in read_stage(job_dir / "entities").to_pylist()
        if row["team_cluster"] in (0, 1)
    }
    controls = _control_samples(job_dir, team_of)
    possessions = _segment(controls)
    _write(job_dir, job_id, possessions)

    durations = [p[3] - p[2] for p in possessions]
    return {
        "job_id": job_id,
        "ball_observations_used": len(controls),
        "possessions": len(possessions),
        "median_possession_s": round(float(np.median(durations)) / 1000, 1) if durations else None,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def _control_samples(job_dir: Path, team_of: dict[int, int]) -> list[tuple[int, int, int]]:
    """(frame_idx, ts_ms, team) at each ball observation with a nearby team player."""
    by_frame_players: dict[int, list[dict]] = defaultdict(list)
    balls: list[dict] = []
    for row in read_stage(job_dir / "localized").to_pylist():
        if row["court_x"] is None:
            continue
        if row["cls"] == int(DetClass.BALL):
            balls.append(row)
        elif row["track_id"] in team_of:
            by_frame_players[row["frame_idx"]].append(row)

    controls: list[tuple[int, int, int]] = []
    for ball in sorted(balls, key=lambda r: r["frame_idx"]):
        players = by_frame_players.get(ball["frame_idx"])
        if not players:
            continue
        dists = [
            (np.hypot(p["court_x"] - ball["court_x"], p["court_y"] - ball["court_y"]), p)
            for p in players
        ]
        dist, nearest = min(dists, key=lambda t: t[0])
        if dist <= CONTROL_RADIUS_FT:
            controls.append((ball["frame_idx"], ball["ts_ms"], team_of[nearest["track_id"]]))
    return controls


def _segment(controls: list[tuple[int, int, int]]) -> list[tuple[int, int, int, int, int]]:
    """Merge control samples into (team, frame_start, ts_start, ts_end, frame_end) runs."""
    runs: list[list] = []
    for frame_idx, ts_ms, team in controls:
        if (
            runs
            and runs[-1][0] == team
            and ts_ms - runs[-1][3] <= MAX_BRIDGE_MS
        ):
            runs[-1][3] = ts_ms
            runs[-1][4] = frame_idx
        else:
            runs.append([team, frame_idx, ts_ms, ts_ms, frame_idx])
    return [
        (team, f_start, ts_start, ts_end, f_end)
        for team, f_start, ts_start, ts_end, f_end in runs
        if ts_end - ts_start >= MIN_POSSESSION_MS
    ]


def _write(job_dir: Path, job_id: str, possessions: list) -> None:
    writer = ArtifactWriter(job_dir / "possessions", POSSESSIONS_SCHEMA)
    for i, (team, f_start, ts_start, ts_end, f_end) in enumerate(possessions):
        writer.add(
            {
                "job_id": job_id,
                "possession_id": i,
                "frame_start": f_start,
                "frame_end": f_end,
                "ts_start_ms": ts_start,
                "ts_end_ms": ts_end,
                "offense_team_cluster": team,
                "outcome": None,
            }
        )
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.job_id), indent=2))


if __name__ == "__main__":
    main()
