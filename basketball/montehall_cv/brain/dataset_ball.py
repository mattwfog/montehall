"""Token streams -> ball-state training arrays (contract §1 ball slice).

The ball estimator's data layer: inputs are the v0 global stream plus an
explicit per-bin ball-observation block (current obs + carry-forward last
obs + age — the v1.1 sticky-evidence-as-input lesson) plus the per-track
streams from dataset_slots (held-mode position lives on a player track).
Targets come from sim_states: true ball x/y + mode per bin, and the
holder as a TRACK index — sim holder_entity mapped through the identity
permutation, so the label lives in the same observed-id space the model
sees. numpy only — torch stays in the trainer.
"""

from __future__ import annotations

import random

import numpy as np

from pathlib import Path

from montehall_cv.brain.dataset import (
    MODE_IGNORE,
    MODES,
    _bin_of,
    featurize,
)
from montehall_cv.brain.dataset_slots import (
    featurize_tracks,
    identity_labels,
    track_ids,
)
from montehall_cv.brain.tokens import CH_BALL, CH_PBP_ANCHOR
from montehall_cv.store.artifacts import read_stage, stage_complete

N_BALL_FEATURES = 7
HOLDER_IGNORE = -100

# per-bin ball-observation feature slots
B_PRESENT = 0
B_X, B_Y = 1, 2
B_CONF = 3
# carry-forward: last observed position + age, so the estimator's job is
# gating stale observations through flights, not long-horizon copying
B_LAST_X, B_LAST_Y = 4, 5
B_AGE = 6
OBS_AGE_CAP_BINS = 40  # 20s of 500ms bins


def featurize_ball(tokens: list[dict]) -> np.ndarray:
    """[T, N_BALL_FEATURES] on the v0 bin grid (T matches featurize)."""
    obs = [t for t in tokens if t["channel"] != CH_PBP_ANCHOR]
    n_bins = _bin_of(max(t["t_ms"] for t in obs)) + 1 if obs else 1
    x = np.zeros((n_bins, N_BALL_FEATURES), dtype=np.float32)
    for t in obs:
        if t["channel"] != CH_BALL or t["court_x"] is None:
            continue
        b = _bin_of(t["t_ms"])
        if t["conf"] is not None and t["conf"] < x[b, B_CONF]:
            continue  # keep the highest-confidence obs in the bin
        x[b, B_PRESENT] = 1.0
        x[b, B_X] = t["court_x"] / 94.0
        x[b, B_Y] = t["court_y"] / 50.0
        x[b, B_CONF] = t["conf"] or 0.0

    last_xy, since = None, 0
    for b in range(n_bins):
        if x[b, B_PRESENT] > 0:
            last_xy, since = (x[b, B_X], x[b, B_Y]), 0
        if last_xy is not None:
            x[b, B_LAST_X], x[b, B_LAST_Y] = last_xy
            x[b, B_AGE] = min(since / OBS_AGE_CAP_BINS, 1.0)
            since += 1
        else:
            x[b, B_AGE] = 1.0  # "never observed" is distinguishable
    return x


def ball_truth(game_dir: Path, n_bins: int) -> dict | None:
    """Per-bin sim truth: normalized xy, mode index, holder TRUE entity.

    The last sim tick landing in a bin wins (matches mode_targets).
    Returns None when the game has no sim_states stage (real games).
    """
    stage = Path(game_dir) / "sim_states"
    if not stage_complete(stage):
        return None
    xy = np.full((n_bins, 2), np.nan, dtype=np.float32)
    mode = np.full(n_bins, MODE_IGNORE, dtype=np.int64)
    holder = np.full(n_bins, HOLDER_IGNORE, dtype=np.int64)
    for r in read_stage(stage).to_pylist():
        b = _bin_of(r["t_ms"])
        if b >= n_bins:
            continue
        xy[b] = (r["ball_x"] / 94.0, r["ball_y"] / 50.0)
        mode[b] = MODES.index(r["ball_mode"])
        holder[b] = -1 if r["holder_entity"] is None else r["holder_entity"]
    return {"xy": xy, "mode": mode, "holder_entity": holder}


def holder_track_labels(holder_entity: np.ndarray, labels: np.ndarray,
                        k_none: int) -> np.ndarray:
    """[T] holder as a track index; k_none = "nobody holds it".

    labels is dataset_slots.identity_labels [K, T] (track -> true entity
    per bin); the holder label is the track whose true entity is the
    holder. HOLDER_IGNORE when truth is absent or the holder's observed
    track is not among the tracks.
    """
    n_bins = holder_entity.shape[0]
    out = np.full(n_bins, HOLDER_IGNORE, dtype=np.int64)
    for b in range(min(n_bins, labels.shape[1])):
        h = holder_entity[b]
        if h == HOLDER_IGNORE:
            continue
        if h == -1:
            out[b] = k_none
            continue
        hits = np.flatnonzero(labels[:, b] == h)
        if hits.size:
            out[b] = int(hits[0])
    return out


def drop_ball_tokens(tokens: list[dict], p: float,
                     seed: int) -> list[dict]:
    """Train-time density knob: drop each ball observation with prob p.

    Token-level so every downstream feature (global ball block,
    carry-forward, per-track ball distances) recomputes consistently —
    an array-level drop would leak the ball through x_global.
    """
    if p <= 0:
        return tokens
    rng = random.Random(seed)
    return [t for t in tokens
            if t["channel"] != CH_BALL or rng.random() >= p]


def load_ball_features(game_dir: Path, ball_dropout: float = 0.0,
                       seed: int = 0) -> dict | None:
    """Tokens-only load (works on real extract dirs): inputs, no targets."""
    game_dir = Path(game_dir)
    if not stage_complete(game_dir / "tokens"):
        return None
    tokens = drop_ball_tokens(
        read_stage(game_dir / "tokens").to_pylist(), ball_dropout, seed)
    tracks = track_ids(tokens)
    x_global = featurize(tokens)
    x_ball = featurize_ball(tokens)
    x_tracks = (featurize_tracks(tokens, tracks) if tracks
                else np.zeros((0, x_global.shape[0], 14), dtype=np.float32))
    t = min(x_global.shape[0], x_ball.shape[0])
    return {
        "tokens": tokens,
        "tracks": tracks,
        "x_global": x_global[:t],
        "x_ball": x_ball[:t],
        "x_tracks": x_tracks[:, :t],
    }


def load_ball_game(game_dir: Path, ball_dropout: float = 0.0,
                   seed: int = 0) -> dict | None:
    """One sim game dir -> ball training arrays; None without sim truth."""
    game_dir = Path(game_dir)
    feats = load_ball_features(game_dir, ball_dropout, seed)
    if feats is None or not feats["tracks"]:
        return None
    truth = ball_truth(game_dir, feats["x_global"].shape[0])
    if truth is None or not stage_complete(game_dir / "sim_identity"):
        return None
    labels = identity_labels(
        read_stage(game_dir / "sim_identity").to_pylist(),
        feats["tracks"], feats["x_global"].shape[0])
    t = feats["x_global"].shape[0]
    holder = holder_track_labels(
        truth["holder_entity"][:t], labels, k_none=len(feats["tracks"]))
    return {
        "game_key": game_dir.name,
        "x_global": feats["x_global"],
        "x_ball": feats["x_ball"],
        "x_tracks": feats["x_tracks"],
        "tracks": feats["tracks"],
        "ball_xy": truth["xy"][:t],
        "pos_mask": ~np.isnan(truth["xy"][:t, 0]),
        "mode": truth["mode"][:t],
        "holder": holder,
        "obs_mask": feats["x_ball"][:, B_PRESENT] > 0,
    }
