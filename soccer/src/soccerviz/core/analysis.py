"""Event-derived possession intervals and transparent spatial baselines.

These possession labels use provider events, not a video-only possession model.
The space-control proxy is a nearest-distance soft assignment, not calibrated EPV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from soccerviz.core.data import LENGTH, WIDTH, Match, load_match, write_json

FEATURES = [
    "ball_progress_m",
    "ball_distance_touchline_m",
    "ball_forward_speed_mps",
    "team_width_m",
    "team_depth_m",
    "opponent_width_m",
    "opponent_depth_m",
    "nearest_opponent_m",
    "teammates_ahead",
    "opponents_ahead",
    "team_center_to_ball_m",
    "advanced_space_share",
    "observed_teammates",
    "observed_opponents",
]
FEATURE_LABELS = {
    "ball_progress_m": "Ball position toward goal",
    "ball_distance_touchline_m": "Distance from touchline",
    "ball_forward_speed_mps": "Recent forward ball speed",
    "team_width_m": "Attacking team width",
    "team_depth_m": "Attacking team depth",
    "opponent_width_m": "Defending team width",
    "opponent_depth_m": "Defending team depth",
    "nearest_opponent_m": "Nearest opponent to ball",
    "teammates_ahead": "Teammates ahead of ball",
    "opponents_ahead": "Opponents ahead of ball",
    "team_center_to_ball_m": "Team center distance to ball",
    "advanced_space_share": "Advanced-zone space share (proxy)",
    "observed_teammates": "Observed teammates",
    "observed_opponents": "Observed opponents",
}
TARGET = "advance_10m_within_5s"
CONTROL_EVENTS = {"PASS", "RECOVERY", "SET PIECE"}
TERMINAL_EVENTS = {"BALL LOST", "BALL OUT", "SHOT", "FAULT RECEIVED"}


def possessions(match: Match) -> pd.DataFrame:
    rows, active = [], None

    def close(end: float, reason: str) -> None:
        nonlocal active
        if active is not None and end > active["start_s"]:
            active.update(end_s=float(end), end_reason=reason)
            active["possession_id"] = f"g{match.game}-p{len(rows):04d}"
            rows.append(active)
        active = None

    for event in match.events.to_dict("records"):
        team = {"Home": 0, "Away": 1}.get(event["Team"])
        if team is None:
            continue
        time, period, kind = float(event["Start Time [s]"]), int(event["Period"]), event["Type"]
        if active is not None and period != active["period"]:
            close(float(match.times[match.periods == active["period"]][-1]), "period_end")
        if kind in CONTROL_EVENTS:
            if active is not None and (team != active["team"] or kind == "SET PIECE"):
                close(time, "team_change" if team != active["team"] else "restart")
            if active is None:
                active = {
                    "game": match.game,
                    "period": period,
                    "team": team,
                    "start_s": time,
                    "start_reason": kind,
                    "events": 0,
                }
            active["events"] += 1
        elif kind in TERMINAL_EVENTS and active is not None:
            # A foul received by the opponent still stops play. Do not bridge the stoppage.
            if team == active["team"] or kind in {"BALL OUT", "FAULT RECEIVED"}:
                close(time, kind)
    if active is not None:
        close(float(match.times[-1]), "recording_end")
    result = pd.DataFrame(rows)
    if not result.empty:
        result["duration_s"] = result.end_s - result.start_s
    return result


def orient(points: np.ndarray, direction: int) -> np.ndarray:
    points = np.asarray(points).copy()
    if direction == -1:
        points[..., 0] = LENGTH - points[..., 0]
    return points


def control_grid(
    ours: np.ndarray, theirs: np.ndarray, nx: int = 24, ny: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth nearest-distance proxy. No probabilities of real possession are claimed."""
    ours = ours[np.isfinite(ours).all(axis=1)]
    theirs = theirs[np.isfinite(theirs).all(axis=1)]
    xs, ys = np.linspace(0, LENGTH, nx), np.linspace(0, WIDTH, ny)
    if not len(ours) or not len(theirs):
        return xs, ys, np.full((ny, nx), np.nan)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1)
    ours_d = np.linalg.norm(grid[:, :, None, :] - ours, axis=-1).min(axis=-1)
    theirs_d = np.linalg.norm(grid[:, :, None, :] - theirs, axis=-1).min(axis=-1)
    return xs, ys, expit((theirs_d - ours_d) / 3.0)


def snapshot_features(match: Match, index: int, team: int) -> dict | None:
    direction = int(match.direction[index, team])
    ball = orient(match.ball[index], direction)
    ours = orient(match.xy[index, match.teams == team], direction)
    theirs = orient(match.xy[index, match.teams != team], direction)
    ours = ours[np.isfinite(ours).all(axis=1)]
    theirs = theirs[np.isfinite(theirs).all(axis=1)]
    if not np.isfinite(ball).all() or len(ours) < 7 or len(theirs) < 7:
        return None
    if not (0 <= ball[0] <= LENGTH and 0 <= ball[1] <= WIDTH):
        return None
    prior = np.searchsorted(match.times, match.times[index] - 0.6)
    if (
        prior >= index
        or match.periods[prior] != match.periods[index]
        or not np.isfinite(match.ball[prior : index + 1]).all()
    ):
        return None
    speed = (match.ball[index, 0] - match.ball[prior, 0]) * direction
    speed /= match.times[index] - match.times[prior]
    xs, _, control = control_grid(ours, theirs, 18, 12)

    def spread(points: np.ndarray, axis: int) -> float:
        return float(np.diff(np.quantile(points[:, axis], [0.1, 0.9]))[0])

    return dict(
        zip(
            FEATURES,
            [
                float(ball[0]),
                float(min(ball[1], WIDTH - ball[1])),
                float(np.clip(speed, -40, 40)),
                spread(ours, 1),
                spread(ours, 0),
                spread(theirs, 1),
                spread(theirs, 0),
                float(np.linalg.norm(theirs - ball, axis=1).min()),
                int((ours[:, 0] > ball[0]).sum()),
                int((theirs[:, 0] > ball[0]).sum()),
                float(np.linalg.norm(ours.mean(axis=0) - ball)),
                float(control[:, xs >= 70].mean()),
                len(ours),
                len(theirs),
            ],
            strict=True,
        )
    )


def forward_target(match: Match, index: int, end_s: float, team: int) -> int | None:
    if not np.isfinite(match.ball[index]).all():
        return None
    end = min(
        np.searchsorted(match.times, match.times[index] + 5.0, side="right"),
        np.searchsorted(match.times, end_s, side="left"),
    )
    future = match.ball[index + 1 : end]
    if not len(future):
        return None
    # Do not turn unobserved futures into negative outcomes.
    if np.isfinite(future).all(axis=1).mean() < 0.9:
        return None
    signed_gain = (future[:, 0] - match.ball[index, 0]) * match.direction[index, team]
    return int(np.nanmax(signed_gain) >= 10.0)


def analyze(root: Path, game: int) -> dict:
    match = load_match(root, game)
    segments = possessions(match)
    samples = []
    skipped = 0
    for segment in segments.to_dict("records"):
        start = np.searchsorted(match.times, segment["start_s"], side="left")
        stop = np.searchsorted(match.times, segment["end_s"], side="left")
        period_start = match.times[match.periods == segment["period"]][0]
        # Non-overlapping five-second prediction horizons limit pseudo-replication.
        for requested in np.arange(
            max(segment["start_s"] + 1, period_start + 5), segment["end_s"] - 1, 5.0
        ):
            i = int(np.searchsorted(match.times, requested, side="left"))
            if i >= stop or i < start:
                continue
            features = snapshot_features(match, i, segment["team"])
            label = forward_target(match, i, segment["end_s"], segment["team"])
            # Period/recording endings censor the future, rather than imply failure.
            censored = (
                segment["end_reason"] in {"period_end", "recording_end"}
                and segment["end_s"] < match.times[i] + 5
            )
            if features is None or label is None or censored:
                skipped += 1
                continue
            samples.append(
                {
                    "sample_id": f"g{game}-f{match.frames[i]}",
                    "game": game,
                    "possession_id": segment["possession_id"],
                    "index": i,
                    "frame": int(match.frames[i]),
                    "time_s": float(match.times[i]),
                    "team": segment["team"],
                    "period": segment["period"],
                    TARGET: label,
                    **features,
                }
            )
    folder = root / "processed" / f"game_{game}"
    table = pd.DataFrame(samples)
    if table.empty:
        raise ValueError("No usable feature/target rows")
    segments.to_parquet(folder / "possessions.parquet", index=False)
    table.to_parquet(folder / "samples.parquet", index=False)
    report = {
        "game": game,
        "possession_segments": len(segments),
        "samples": len(table),
        "skipped_samples": skipped,
        "positive_labels": int(table[TARGET].sum()),
        "positive_rate": float(table[TARGET].mean()),
        "target": TARGET,
        "target_definition": "Ball advances >=10m before 5s or the end of its event-derived possession",
        "possession_source": "Metrica provider events; retrospective segmentation",
        "space_control": "sigmoid of nearest-opponent minus nearest-teammate distance / 3m",
        "feature_policy": "Positions at sample time; ball velocity from past 0.6s only",
    }
    write_json(folder / "analysis.json", report)
    return report
