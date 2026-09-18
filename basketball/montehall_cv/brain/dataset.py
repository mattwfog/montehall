"""Token streams -> per-step training arrays for the brain (v0).

Every game becomes a [T, F] float32 feature matrix on a fixed 500ms grid
plus aligned target vectors. pbp_anchor tokens are TARGETS ONLY — the
featurizer ignores that channel entirely, so anchor leakage into inference
inputs is impossible by construction (contract §2 train-only rule).

Real align-only games contribute event targets over sparse channels;
sim games contribute full channels + dense ball-mode targets
(sim_states). numpy only — torch stays in the trainer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from montehall_cv.brain.tokens import (
    CH_BALL,
    CH_BALL_CONTROL,
    CH_CLOCK,
    CH_CUT,
    CH_JERSEY_READ,
    CH_NAME_CALL,
    CH_PBP_ANCHOR,
    CH_PLAYER,
    CH_RIM,
    CH_SCORE_DELTA,
)
from montehall_cv.store.artifacts import read_stage, stage_complete

BIN_MS = 500
N_FEATURES = 36
MODES = ("held", "dribbled", "ballistic", "dead")
MODE_IGNORE = -100
TARGET_SMEAR_BINS = 2  # a shot anchor marks +-2 bins (+-1s)
NEAR_RIM_FT = 6.0

# Inverse-ablation channel sets (2026-07-30): "align" is exactly what the
# accidental align-only ablation had (clock+cut); "perception" is its
# complement. pbp_anchor stays target-only in every variant.
ALIGN_CHANNELS = frozenset({CH_CLOCK, CH_CUT})
PERCEPTION_CHANNELS = frozenset({
    CH_BALL, CH_BALL_CONTROL, CH_JERSEY_READ, CH_NAME_CALL,
    CH_PLAYER, CH_RIM, CH_SCORE_DELTA,
})
CHANNEL_SETS: dict[str, frozenset[str] | None] = {
    "all": None,
    "align": ALIGN_CHANNELS,
    "perception": PERCEPTION_CHANNELS,
}


def _bin_of(t_ms: int) -> int:
    return int(t_ms) // BIN_MS


def featurize(tokens: list[dict],
              channels: frozenset[str] | None = None) -> np.ndarray:
    """Observation tokens -> [T, N_FEATURES]; pbp_anchor never enters.

    channels: optional observation-channel whitelist (CHANNEL_SETS values).
    T always spans ALL observation channels, so masked variants of one game
    stay length-aligned with each other and with their event targets.
    """
    obs = [t for t in tokens if t["channel"] != CH_PBP_ANCHOR]
    if not obs:
        return np.zeros((1, N_FEATURES), dtype=np.float32)
    n_bins = _bin_of(max(t["t_ms"] for t in obs)) + 1
    if channels is not None:
        obs = [t for t in obs if t["channel"] in channels]
    x = np.zeros((n_bins, N_FEATURES), dtype=np.float32)
    players: dict[int, list[dict]] = {}
    for t in obs:
        b = _bin_of(t["t_ms"])
        ch = t["channel"]
        if ch == CH_CLOCK:
            x[b, 0] = 1.0
            x[b, 1] = (t["value"] or 0.0) / 2400.0
            x[b, 2] = (t["period"] or 0) / 2.0
            x[b, 3] = t["conf"] or 0.0
        elif ch == CH_CUT:
            x[b, 4] = 1.0
        elif ch == CH_SCORE_DELTA:
            x[b, 5] = 1.0
            x[b, 6] = (t["value"] or 0.0) / 3.0
            x[b, 7] = 1.0 if t["text"] == "home" else -1.0
        elif ch == CH_BALL:
            x[b, 8] = min(x[b, 8] + 0.25, 1.0)
            x[b, 9] = (t["court_x"] or 0.0) / 94.0
            x[b, 10] = (t["court_y"] or 0.0) / 50.0
            x[b, 11] = max(x[b, 11], t["conf"] or 0.0)
        elif ch == CH_RIM:
            x[b, 12] = 1.0
            x[b, 13] = (t["court_x"] or 0.0) / 94.0
            x[b, 14] = (t["court_y"] or 0.0) / 50.0
        elif ch == CH_BALL_CONTROL:
            x[b, 15] = 1.0
            d = min((t["value"] or 10.0) / 10.0, 1.0)
            x[b, 16] = d if x[b, 16] == 0 else min(x[b, 16], d)
            x[b, 17] = min(x[b, 17] + 0.25, 1.0)
        elif ch == CH_PLAYER:
            players.setdefault(b, []).append(t)
        elif ch == CH_JERSEY_READ:
            x[b, 28] = min(x[b, 28] + 1 / 3, 1.0)
        elif ch == CH_NAME_CALL:
            x[b, 29] = min(x[b, 29] + 1 / 3, 1.0)
        x[b, 34] = min(x[b, 34] + 1 / 50, 1.0)

    for b, rows in players.items():
        xs = np.array([r["court_x"] or 0.0 for r in rows])
        ys = np.array([r["court_y"] or 0.0 for r in rows])
        clusters = np.array([r["team_cluster"] if r["team_cluster"] is not None
                             else -1 for r in rows])
        x[b, 18] = min(len(rows) / 12.0, 1.0)
        for cluster, (ci, cxi, cyi) in ((0, (19, 21, 22)), (1, (20, 23, 24))):
            sel = clusters == cluster
            if sel.any():
                x[b, ci] = min(sel.sum() / 6.0, 1.0)
                x[b, cxi] = xs[sel].mean() / 94.0
                x[b, cyi] = ys[sel].mean() / 50.0
        x[b, 25] = xs.std() / 47.0 if len(rows) > 1 else 0.0
        x[b, 26] = ys.std() / 25.0 if len(rows) > 1 else 0.0
        confs = [r["conf"] for r in rows if r["conf"] is not None]
        x[b, 27] = float(np.mean(confs)) if confs else 0.0

    # second pass: ball velocity proxy + ball-near-rim geometry
    for b in range(1, n_bins):
        if x[b, 8] > 0 and x[b - 1, 8] > 0:
            x[b, 32] = np.clip((x[b, 9] - x[b - 1, 9]) * 94.0 / 10.0, -1, 1)
            x[b, 33] = np.clip((x[b, 10] - x[b - 1, 10]) * 50.0 / 10.0, -1, 1)
    both = (x[:, 8] > 0) & (x[:, 12] > 0)
    if both.any():
        d = np.hypot((x[:, 9] - x[:, 13]) * 94.0, (x[:, 10] - x[:, 14]) * 50.0)
        near = both & (d < NEAR_RIM_FT)
        x[near, 30] = 1.0
        x[both, 31] = np.clip(d[both] / 10.0, 0, 1)
    x[:, 35] = 1.0
    return x


def event_targets(tokens: list[dict], n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """(shot[T], made[T]) from pbp_anchor tokens, smeared +-TARGET_SMEAR_BINS."""
    shot = np.zeros(n_bins, dtype=np.float32)
    made = np.zeros(n_bins, dtype=np.float32)
    for t in tokens:
        if t["channel"] != CH_PBP_ANCHOR:
            continue
        payload = json.loads(t["payload_json"] or "{}")
        if not payload.get("shooting_play"):
            continue
        b = _bin_of(t["t_ms"])
        lo, hi = max(0, b - TARGET_SMEAR_BINS), min(n_bins, b + TARGET_SMEAR_BINS + 1)
        shot[lo:hi] = 1.0
        if payload.get("scoring_play"):
            made[lo:hi] = 1.0
    return shot, made


def mode_targets(game_dir: Path, n_bins: int) -> np.ndarray:
    """Sim-only dense ball-mode labels; MODE_IGNORE where unavailable."""
    out = np.full(n_bins, MODE_IGNORE, dtype=np.int64)
    stage = Path(game_dir) / "sim_states"
    if not stage_complete(stage):
        return out
    for r in read_stage(stage).to_pylist():
        b = _bin_of(r["t_ms"])
        if b < n_bins:
            out[b] = MODES.index(r["ball_mode"])
    return out


def shot_times_ms(tokens: list[dict]) -> list[tuple[int, bool]]:
    """Truth (t_ms, scoring) per shooting anchor — the eval join input."""
    out = []
    for t in tokens:
        if t["channel"] != CH_PBP_ANCHOR:
            continue
        payload = json.loads(t["payload_json"] or "{}")
        if payload.get("shooting_play"):
            out.append((int(t["t_ms"]), bool(payload.get("scoring_play"))))
    return sorted(out)


def load_game(game_dir: Path,
              channels: frozenset[str] | None = None) -> dict | None:
    """One game dir -> arrays; None when it has no token stage."""
    game_dir = Path(game_dir)
    if not stage_complete(game_dir / "tokens"):
        return None
    tokens = read_stage(game_dir / "tokens").to_pylist()
    x = featurize(tokens, channels)
    shot, made = event_targets(tokens, len(x))
    return {
        "game_key": tokens[0]["game_key"] if tokens else game_dir.name,
        "x": x,
        "shot": shot,
        "made": made,
        "mode": mode_targets(game_dir, len(x)),
        "n_shots": int(len(shot_times_ms(tokens))),
    }


def discover_games(roots: list[Path]) -> list[Path]:
    """Every dir under the roots holding a complete tokens stage."""
    out: list[Path] = []
    for root in roots:
        root = Path(root)
        if stage_complete(root / "tokens"):
            out.append(root)
            continue
        for child in sorted(root.iterdir()) if root.is_dir() else []:
            if stage_complete(child / "tokens"):
                out.append(child)
    return out
