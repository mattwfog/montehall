"""Per-track token streams -> slot-estimator training arrays (thesis §10).

Where dataset.py (v0) collapses all players into team centroids, this
layer keeps one feature stream PER OBSERVED TRACK on the same 500ms
grid, so identity evidence (jersey-read values, ball control, proximity)
stays bound to the track it was observed on. Targets come from the
sim_identity stage: the observed->true permutation replayed per bin.

Named v1 simplifications: name_call is not a per-track feature (no track
binding exists in real payloads); real rosters come from the caller
(shot-attribution eval derives them from pbp anchors). numpy only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from montehall_cv.brain.dataset import BIN_MS, _bin_of, featurize
from montehall_cv.brain.tokens import (
    CH_BALL,
    CH_BALL_CONTROL,
    CH_JERSEY_READ,
    CH_PBP_ANCHOR,
    CH_PLAYER,
)
from montehall_cv.store.artifacts import read_stage, stage_complete

N_TRACK_FEATURES = 14
K_MAX = 16
LABEL_IGNORE = -100
MAX_JERSEY = 99
READ_AGE_CAP_BINS = 240  # 2 min of 500ms bins

# per-track feature slots
F_PRESENT = 0
F_X, F_Y = 1, 2
F_DX, F_DY = 3, 4
F_CONF = 5
F_BALL_DIST = 6
F_CONTROL_DIST = 7
F_NEIGHBOR_DIST = 8
F_READ = 9
F_READ_VALUE = 10
F_READ_CONF = 11
# carry-forward: the sticky baseline as an INPUT, so the model's job is
# gating stale evidence through crossovers, not long-horizon copying
F_LAST_READ_VALUE = 12
F_READ_AGE = 13


def parse_jersey(text: str | None) -> int | None:
    if text is None:
        return None
    digits = "".join(c for c in str(text) if c.isdigit())
    if not digits:
        return None
    n = int(digits)
    return n if 0 <= n <= MAX_JERSEY else None


def track_ids(tokens: list[dict], k_max: int = K_MAX) -> list[int]:
    """Observed track ids ordered by token count, capped at k_max."""
    counts: dict[int, int] = {}
    for t in tokens:
        e = t.get("entity_id")
        if e is None:
            continue
        if t["channel"] in (CH_PLAYER, CH_BALL_CONTROL):
            counts[e] = counts.get(e, 0) + 1
    ranked = sorted(counts, key=lambda e: (-counts[e], e))
    return ranked[:k_max]


def featurize_tracks(tokens: list[dict],
                     tracks: list[int]) -> np.ndarray:
    """[K, T, N_TRACK_FEATURES] on the v0 bin grid (T matches featurize)."""
    obs = [t for t in tokens if t["channel"] != CH_PBP_ANCHOR]
    n_bins = _bin_of(max(t["t_ms"] for t in obs)) + 1 if obs else 1
    idx = {e: k for k, e in enumerate(tracks)}
    x = np.zeros((len(tracks), n_bins, N_TRACK_FEATURES), dtype=np.float32)
    ball = np.full((n_bins, 2), np.nan, dtype=np.float32)

    for t in obs:
        b = _bin_of(t["t_ms"])
        ch = t["channel"]
        if ch == CH_BALL and t["court_x"] is not None:
            ball[b] = (t["court_x"], t["court_y"])
            continue
        if ch == CH_JERSEY_READ:  # track binding lives in the payload
            payload = json.loads(t["payload_json"] or "{}")
            k2 = idx.get(payload.get("track_id"))
            value = parse_jersey(t["text"])
            if k2 is not None and value is not None:
                x[k2, b, F_READ] = 1.0
                x[k2, b, F_READ_VALUE] = value / MAX_JERSEY
                x[k2, b, F_READ_CONF] = t["conf"] or 0.0
            continue
        e = t.get("entity_id")
        k = idx.get(e) if e is not None else None
        if k is None:
            continue
        if ch == CH_PLAYER:
            x[k, b, F_PRESENT] = 1.0
            x[k, b, F_X] = (t["court_x"] or 0.0) / 94.0
            x[k, b, F_Y] = (t["court_y"] or 0.0) / 50.0
            x[k, b, F_CONF] = max(x[k, b, F_CONF], t["conf"] or 0.0)
        elif ch == CH_BALL_CONTROL:
            x[k, b, F_CONTROL_DIST] = min((t["value"] or 10.0) / 10.0, 1.0)

    for k in range(len(tracks)):
        present = x[k, :, F_PRESENT] > 0
        for b in np.flatnonzero(present):
            if b > 0 and present[b - 1]:
                x[k, b, F_DX] = np.clip(
                    (x[k, b, F_X] - x[k, b - 1, F_X]) * 94.0 / 10.0, -1, 1)
                x[k, b, F_DY] = np.clip(
                    (x[k, b, F_Y] - x[k, b - 1, F_Y]) * 50.0 / 10.0, -1, 1)
            if not np.isnan(ball[b, 0]):
                d = float(np.hypot(x[k, b, F_X] * 94.0 - ball[b, 0],
                                   x[k, b, F_Y] * 50.0 - ball[b, 1]))
                x[k, b, F_BALL_DIST] = min(d / 10.0, 1.0)

    # carry-forward last read per track (value persists, age grows;
    # age stays 1.0 before any read so "never read" is distinguishable)
    for k in range(len(tracks)):
        last_value, since = None, 0
        for b in range(n_bins):
            if x[k, b, F_READ] > 0:
                last_value, since = x[k, b, F_READ_VALUE], 0
            if last_value is not None:
                x[k, b, F_LAST_READ_VALUE] = last_value
                x[k, b, F_READ_AGE] = min(since / READ_AGE_CAP_BINS, 1.0)
                since += 1
            else:
                x[k, b, F_READ_AGE] = 1.0

    # nearest-other-track distance (the occlusion/crossover signal)
    for b in range(n_bins):
        live = [k for k in range(len(tracks)) if x[k, b, F_PRESENT] > 0]
        for k in live:
            others = [k2 for k2 in live if k2 != k]
            if not others:
                continue
            d = min(
                float(np.hypot((x[k, b, F_X] - x[k2, b, F_X]) * 94.0,
                               (x[k, b, F_Y] - x[k2, b, F_Y]) * 50.0))
                for k2 in others)
            x[k, b, F_NEIGHBOR_DIST] = min(d / 10.0, 1.0)
    return x


def identity_labels(rows: list[dict], tracks: list[int],
                    n_bins: int) -> np.ndarray:
    """[K, T] roster-slot labels from sim_identity (observed->true at the
    end of each bin); LABEL_IGNORE for tracks the stage never mentions."""
    idx = {e: k for k, e in enumerate(tracks)}
    labels = np.full((len(tracks), n_bins), LABEL_IGNORE, dtype=np.int64)
    events = sorted(rows, key=lambda r: r["t_ms"])
    mapping: dict[int, int] = {}  # observed -> true
    e_i = 0
    for b in range(n_bins):
        end_ms = (b + 1) * BIN_MS - 1
        while e_i < len(events) and events[e_i]["t_ms"] <= end_ms:
            r = events[e_i]
            mapping = {obs: true for obs, true in mapping.items()
                       if true != r["true_entity"]}
            mapping[r["observed_id"]] = r["true_entity"]
            e_i += 1
        for obs, true in mapping.items():
            k = idx.get(obs)
            if k is not None:
                labels[k, b] = true
    return labels


def load_track_features(game_dir: Path) -> dict | None:
    """Tokens-only load (works on real extract dirs): features, no labels."""
    game_dir = Path(game_dir)
    if not stage_complete(game_dir / "tokens"):
        return None
    tokens = read_stage(game_dir / "tokens").to_pylist()
    tracks = track_ids(tokens)
    if not tracks:
        return None
    return {
        "tokens": tokens,
        "tracks": tracks,
        "x_global": featurize(tokens),
        "x_tracks": featurize_tracks(tokens, tracks),
    }


def load_slot_game(game_dir: Path) -> dict | None:
    """One sim game dir -> slot training arrays; None without the stages."""
    game_dir = Path(game_dir)
    for stage in ("tokens", "sim_identity"):
        if not stage_complete(game_dir / stage):
            return None
    tokens = read_stage(game_dir / "tokens").to_pylist()
    meta = json.loads((game_dir / "sim_meta.json").read_text())
    jerseys = meta.get("jerseys")
    if not jerseys:
        return None
    tracks = track_ids(tokens)
    x_global = featurize(tokens)
    x_tracks = featurize_tracks(tokens, tracks)
    labels = identity_labels(
        read_stage(game_dir / "sim_identity").to_pylist(),
        tracks, x_tracks.shape[1])
    t = min(x_global.shape[0], x_tracks.shape[1])
    return {
        "game_key": meta.get("game_key", game_dir.name),
        "x_global": x_global[:t],
        "x_tracks": x_tracks[:, :t],
        "labels": labels[:, :t],
        "tracks": tracks,
        "roster": np.array([parse_jersey(j) or 0 for j in jerseys],
                           dtype=np.int64),
    }
