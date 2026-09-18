"""Entity-grain ball-control samples — the single implementation.

A control sample is a ball observation with a nearby on-court player
(nearest entity within CONTROL_RADIUS_FT, both in court feet). This logic
previously lived verbatim in run_boxscore._ball_controls and
traces._controls; run_events persists its output as the ball_controls
artifact so downstream stages share one computation.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from montehall_cv.store.artifacts import read_stage
from montehall_cv.store.records import DetClass

CONTROL_RADIUS_FT = 6.0
BALL_INTERP_MAX_GAP_MS = 1500
BALL_INTERP_STEP_MS = 150


def interpolate_ball(balls: list[dict]) -> list[dict]:
    """Linear court-space fill between nearby ball observations.

    Ball visibility (~20% of frames even with the v2 detector) is THE
    shooter-bind limiter (UConn 2026-07-09: 1 of 5 attempts bindable).
    Filled samples are honest: is_detected=False, and every control they
    produce carries interpolated=True. Gaps beyond the bound stay gaps —
    a pass and an occlusion look identical over long spans.
    """
    filled: list[dict] = []
    ordered = sorted(balls, key=lambda r: r["ts_ms"])
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        gap = cur["ts_ms"] - prev["ts_ms"]
        if not (BALL_INTERP_STEP_MS < gap <= BALL_INTERP_MAX_GAP_MS):
            continue
        for i in range(1, int(gap // BALL_INTERP_STEP_MS) + 1):
            f = i * BALL_INTERP_STEP_MS / gap
            if f >= 1.0:
                break
            filled.append(
                {
                    **prev,
                    "ts_ms": prev["ts_ms"] + int(gap * f),
                    "frame_idx": prev["frame_idx"]
                    + int(round((cur["frame_idx"] - prev["frame_idx"]) * f)),
                    "court_x": prev["court_x"] + (cur["court_x"] - prev["court_x"]) * f,
                    "court_y": prev["court_y"] + (cur["court_y"] - prev["court_y"]) * f,
                    "is_detected": False,
                }
            )
    return sorted(ordered + filled, key=lambda r: r["ts_ms"])


def entity_control_samples(job_dir: Path, entity_of: dict[int, dict]) -> list[dict]:
    """ts-ordered control samples from the localized stage.

    entity_of: track_id -> entities row (entity_id, team_cluster, ...).
    Returns dicts with ts_ms, frame_idx, entity_id, team_cluster,
    court_x/court_y (the controlling player's), ball_x/ball_y, dist_ft,
    interpolated (True when the ball sample was gap-filled).
    """
    by_frame: dict[int, list[dict]] = defaultdict(list)
    balls: list[dict] = []
    for row in read_stage(job_dir / "localized").to_pylist():
        if row["court_x"] is None:
            continue
        if row["cls"] == int(DetClass.BALL):
            balls.append(row)
        elif row["track_id"] in entity_of:
            by_frame[row["frame_idx"]].append(row)

    controls: list[dict] = []
    for ball in interpolate_ball(balls):
        players = by_frame.get(ball["frame_idx"])
        if not players:
            continue
        # key= keeps equidistant players from falling through to dict
        # comparison (TypeError); the first in track order wins the tie.
        dist, nearest = min(
            ((np.hypot(p["court_x"] - ball["court_x"], p["court_y"] - ball["court_y"]), p)
             for p in players),
            key=lambda pair: pair[0],
        )
        if dist <= CONTROL_RADIUS_FT:
            entity = entity_of[nearest["track_id"]]
            controls.append(
                {
                    "ts_ms": ball["ts_ms"],
                    "frame_idx": ball["frame_idx"],
                    "entity_id": entity["entity_id"],
                    "team_cluster": entity["team_cluster"],
                    "court_x": nearest["court_x"],
                    "court_y": nearest["court_y"],
                    "ball_x": ball["court_x"],
                    "ball_y": ball["court_y"],
                    "dist_ft": float(dist),
                    "interpolated": not ball.get("is_detected", True),
                }
            )
    return controls
