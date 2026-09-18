"""Token vocabulary constants + ordering helpers (contract §2/§4)."""

from __future__ import annotations

import json
from pathlib import Path

TOKENS_VERSION = 1

# Channel names — the observation vocabulary. Appending a channel is a
# contract addition; renaming or reusing one is a version bump.
CH_CLOCK = "clock"
CH_CUT = "cut"
CH_SCORE_DELTA = "score_delta"
CH_RIM = "rim"
CH_PLAYER = "player"
CH_BALL = "ball"
CH_BALL_CONTROL = "ball_control"
CH_JERSEY_READ = "jersey_read"
CH_NAME_CALL = "name_call"
CH_PBP_ANCHOR = "pbp_anchor"

# Supervision-only channels: the dataloader may target them, inference
# never consumes them.
TRAIN_ONLY_CHANNELS = frozenset({CH_PBP_ANCHOR})

# Stable sort priority within one t_ms (anchor spine first, densest last).
CHANNEL_PRIORITY: dict[str, int] = {
    CH_CLOCK: 0,
    CH_CUT: 1,
    CH_SCORE_DELTA: 2,
    CH_RIM: 3,
    CH_PLAYER: 4,
    CH_BALL: 5,
    CH_BALL_CONTROL: 6,
    CH_JERSEY_READ: 7,
    CH_NAME_CALL: 8,
    CH_PBP_ANCHOR: 9,
}

# Downsample grids (contract §4).
PLAYER_GRID_MS = 200  # 5 Hz — the B1 overlay keyframe precedent
RIM_GRID_MS = 1000
CUT_GAP_S = 5.0  # confident-clock-read gap that marks a replay/cut span
CLOCK_CONF_FLOOR = 0.85  # mirrors eval.pbp_align.CONF_FLOOR

META_NAME = "tokens_meta.json"


def make_token(game_key: str, t_ms: int, channel: str, **kw) -> dict:
    """One TOKENS_SCHEMA row with every optional field defaulted."""
    row = {
        "game_key": game_key,
        "token_idx": 0,
        "t_ms": int(t_ms),
        "channel": channel,
        "entity_id": None,
        "team_cluster": None,
        "court_x": None,
        "court_y": None,
        "value": None,
        "text": None,
        "conf": None,
        "period": None,
        "payload_json": None,
    }
    row.update(kw)
    return row


def sort_key(row: dict) -> tuple:
    return (
        row["t_ms"],
        CHANNEL_PRIORITY.get(row["channel"], 99),
        row.get("entity_id") if row.get("entity_id") is not None else -1,
    )


def write_meta(out_dir: Path, game_key: str, counts: dict[str, int],
               sources: list[str]) -> dict:
    meta = {
        "version": TOKENS_VERSION,
        "game_key": game_key,
        "channels": dict(sorted(counts.items())),
        "total": sum(counts.values()),
        "sources": sorted(sources),
    }
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=2))
    return meta
