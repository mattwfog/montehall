"""Causal off-screen estimates and their measured downstream error under simulated crops."""

from pathlib import Path

import numpy as np
import pandas as pd

from soccerviz.core.analysis import control_grid
from soccerviz.core.data import load_match, write_json


class LastSeenState:
    def __init__(self, players, ttl_s=3.0):
        self.position = np.full((players, 2), np.nan)
        self.velocity = np.zeros((players, 2))
        self.last_seen = np.full(players, -np.inf)
        self.ttl_s = ttl_s

    def update(self, observations, timestamp):
        observed = np.isfinite(observations).all(axis=1)
        elapsed = timestamp - self.last_seen
        continuous = (
            observed & np.isfinite(self.position).all(axis=1) & (elapsed <= 0.5) & (elapsed > 0)
        )
        self.velocity[observed & ~continuous] = 0
        self.velocity[continuous] = (
            observations[continuous] - self.position[continuous]
        ) / elapsed[continuous, None]
        speed = np.linalg.norm(self.velocity, axis=1)
        self.velocity *= np.minimum(1, 10 / np.maximum(speed, 1e-6))[:, None]
        self.position[observed] = observations[observed]
        self.last_seen[observed] = timestamp
        age = timestamp - self.last_seen
        eligible = ~observed & (age <= self.ttl_s)
        held, moving = observations.copy(), observations.copy()
        held[eligible] = self.position[eligible]
        moving[eligible] = np.clip(
            self.position[eligible] + self.velocity[eligible] * age[eligible, None],
            [0, 0],
            [105, 68],
        )
        return held, moving, age, eligible


def crop_benchmark(match):
    memory = LastSeenState(len(match.players))
    camera = np.array([52.5, 34.0])
    errors, control_errors, state_rows = [], [], []
    total_hidden, estimated = 0, 0
    for i, timestamp in enumerate(match.times):
        if i == 0 or match.periods[i] != match.periods[i - 1]:
            memory = LastSeenState(len(match.players))
            camera = np.array([52.5, 34.0])
        if np.isfinite(match.ball[i]).all():
            target = np.clip(match.ball[i], [27.5, 22.5], [77.5, 45.5])
            camera = 0.85 * camera + 0.15 * target
        truth = match.xy[i]
        visible = np.isfinite(truth).all(axis=1) & (np.abs(truth - camera) <= [27.5, 22.5]).all(
            axis=1
        )
        observations = np.where(visible[:, None], truth, np.nan)
        held, moving, age, eligible = memory.update(observations, timestamp)
        hidden = ~visible & np.isfinite(truth).all(axis=1)
        total_hidden += int(hidden.sum())
        estimated += int((hidden & eligible).sum())
        for j in np.flatnonzero(hidden & eligible):
            errors.append(
                {
                    "game": match.game,
                    "index": i,
                    "player": int(j),
                    "age_s": float(age[j]),
                    "last_seen_error_m": float(np.linalg.norm(held[j] - truth[j])),
                    "velocity_error_m": float(np.linalg.norm(moving[j] - truth[j])),
                }
            )
        if i % 25 == 0 and visible[match.teams == 0].sum() and visible[match.teams == 1].sum():
            full = control_grid(truth[match.teams == 0], truth[match.teams == 1], 18, 12)[2]
            for method, positions in [
                ("visible_only", observations),
                ("last_seen", held),
                ("velocity", moving),
            ]:
                approximate = control_grid(
                    positions[match.teams == 0], positions[match.teams == 1], 18, 12
                )[2]
                control_errors.append(
                    {
                        "game": match.game,
                        "index": i,
                        "method": method,
                        "control_grid_mae": float(np.nanmean(np.abs(full - approximate))),
                        "team_share_absolute_error": float(
                            abs(np.nanmean(full) - np.nanmean(approximate))
                        ),
                    }
                )
            for j in range(len(match.players)):
                state_rows.append(
                    {
                        "game": match.game,
                        "index": i,
                        "timestamp_s": float(timestamp),
                        "player_id": match.players[j],
                        "team": int(match.teams[j]),
                        "x_m": float(moving[j, 0]),
                        "y_m": float(moving[j, 1]),
                        "status": "observed"
                        if visible[j]
                        else "estimated"
                        if eligible[j]
                        else "unavailable",
                        "age_s": float(age[j]) if np.isfinite(age[j]) else None,
                    }
                )
    return (
        pd.DataFrame(errors),
        pd.DataFrame(control_errors),
        pd.DataFrame(state_rows),
        {
            "hidden_observations": total_hidden,
            "estimated_hidden_observations": estimated,
            "hidden_coverage_within_3s_ttl": estimated / total_hidden,
        },
    )


def run_uncertainty(data: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    train, _train_control, _, train_coverage = crop_benchmark(load_match(data, 1))
    test, test_control, states, test_coverage = crop_benchmark(load_match(data, 2))
    radii = {}
    for bucket in (1, 2, 3):
        values = train.loc[
            (train.age_s > bucket - 1) & (train.age_s <= bucket + 1e-6), "velocity_error_m"
        ]
        radii[bucket] = float(np.quantile(values, 0.9, method="higher"))
    test["empirical_90_radius_m"] = [
        radii[min(3, max(1, int(np.ceil(age - 1e-6))))] for age in test.age_s
    ]
    states["empirical_90_radius_m"] = [
        radii[min(3, max(1, int(np.ceil(age - 1e-6))))] if status == "estimated" else None
        for status, age in zip(states.status, states.age_s, strict=True)
    ]
    states.to_parquet(out / "reconstructed-state.parquet", index=False)
    test.to_parquet(out / "occlusion-errors.parquet", index=False)
    test_control.to_parquet(out / "control-errors.parquet", index=False)
    report = {
        "benchmark": "simulated 55x45m broadcast viewport on provider tracking; causal camera follows observed ball",
        "calibration_game": 1,
        "evaluation_game": 2,
        "max_estimation_age_s": 3,
        "train_coverage": train_coverage,
        "evaluation_coverage": test_coverage,
        "last_seen_mae_m": float(test.last_seen_error_m.mean()),
        "velocity_mae_m": float(test.velocity_error_m.mean()),
        "velocity_p90_error_m": float(test.velocity_error_m.quantile(0.9)),
        "calibration_radius_by_age_second_m": radii,
        "evaluation_empirical_90_coverage": float(
            (test.velocity_error_m <= test.empirical_90_radius_m).mean()
        ),
        "control_grid_mae_by_method": test_control.groupby("method")
        .control_grid_mae.mean()
        .to_dict(),
        "limitations": [
            "Simulated visibility with perfect observed tracking and known identities",
            "Estimates expire after 3s; unseen players remain unknown",
            "Radii are empirical error quantiles from game 1, not guaranteed confidence bounds",
            "Timestamps and players are correlated; no independent-sample claim",
            "Nearest-distance control proxy is not calibrated pitch control",
        ],
    }
    write_json(out / "report.json", report)
    return report
