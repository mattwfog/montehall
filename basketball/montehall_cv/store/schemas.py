"""pyarrow schemas for the primitive store — one per record contract."""

from __future__ import annotations

import pyarrow as pa

DETECTIONS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("frame_idx", pa.int32()),
        pa.field("ts_ms", pa.int64()),
        pa.field("det_idx", pa.int32()),
        pa.field("cls", pa.int8()),
        pa.field("x1", pa.float32()),
        pa.field("y1", pa.float32()),
        pa.field("x2", pa.float32()),
        pa.field("y2", pa.float32()),
        pa.field("conf", pa.float32()),
        pa.field("track_id", pa.int32(), nullable=True),
        pa.field("is_detected", pa.bool_()),
        pa.field("court_x", pa.float32(), nullable=True),
        pa.field("court_y", pa.float32(), nullable=True),
        pa.field("court_conf", pa.float32(), nullable=True),
    ]
)

TRACKLETS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("track_id", pa.int32()),
        pa.field("cls", pa.int8()),
        pa.field("frame_start", pa.int32()),
        pa.field("frame_end", pa.int32()),
        pa.field("ts_start_ms", pa.int64()),
        pa.field("ts_end_ms", pa.int64()),
        pa.field("n_detections", pa.int32()),
        pa.field("mean_conf", pa.float32()),
    ]
)

COURT_FRAMES_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("frame_idx", pa.int32()),
        pa.field("ts_ms", pa.int64()),
        pa.field("h", pa.list_(pa.float64()), nullable=True),  # 9 row-major image->court
        pa.field("residual_ft", pa.float32(), nullable=True),
        pa.field("n_points", pa.int32()),
    ]
)

IDENTITY_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("track_id", pa.int32()),
        pa.field("candidate", pa.string()),
        pa.field("prob", pa.float32()),
        pa.field("bound_at_stage", pa.string()),
        pa.field("bound_at_frame", pa.int32(), nullable=True),
    ]
)

EVIDENCE_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("track_id", pa.int32()),
        pa.field("frame_idx", pa.int32(), nullable=True),
        pa.field("source_kind", pa.string()),
        pa.field("value_json", pa.string()),
        pa.field("weight", pa.float32()),
    ]
)

# Geometry-translated track samples: every localized on-court detection
# at entity grain (players) or ungrouped (ball). The court-frame substrate
# for zones, plays, and rendering.
POSITIONS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("frame_idx", pa.int32()),
        pa.field("ts_ms", pa.int64()),
        pa.field("cls", pa.int8()),
        pa.field("entity_id", pa.int32(), nullable=True),  # None for the ball
        pa.field("team_cluster", pa.int8(), nullable=True),
        pa.field("court_x", pa.float32()),
        pa.field("court_y", pa.float32()),
        pa.field("court_conf", pa.float32(), nullable=True),
    ]
)

BALL_CONTROLS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("frame_idx", pa.int32()),
        pa.field("ts_ms", pa.int64()),
        pa.field("entity_id", pa.int32()),
        pa.field("team_cluster", pa.int8()),
        pa.field("court_x", pa.float32()),
        pa.field("court_y", pa.float32()),
        pa.field("ball_x", pa.float32()),
        pa.field("ball_y", pa.float32()),
        pa.field("dist_ft", pa.float32()),
        # True when the ball sample was gap-filled (controls.interpolate_ball)
        pa.field("interpolated", pa.bool_(), nullable=True),
    ]
)

# Entity-grain event stream (perception grain — no jersey attribution;
# identity binding stays late, box_events is the attribution grain).
ATOMIC_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("event_id", pa.int32()),
        pa.field("event_type", pa.string()),
        pa.field("frame_idx", pa.int32()),
        pa.field("ts_ms", pa.int64()),
        pa.field("entity_id", pa.int32(), nullable=True),
        pa.field("team_cluster", pa.int8(), nullable=True),
        pa.field("court_x", pa.float32(), nullable=True),
        pa.field("court_y", pa.float32(), nullable=True),
        pa.field("possession_id", pa.int32(), nullable=True),
        pa.field("payload_json", pa.string(), nullable=True),
    ]
)

# One row per possession: the play-generation payload (trajectories in
# court feet, zones, events, outcome) as JSON — the app-consumable shape.
PLAYS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("play_id", pa.int32()),
        pa.field("payload_json", pa.string()),
    ]
)

POSSESSIONS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("possession_id", pa.int32()),
        pa.field("frame_start", pa.int32()),
        pa.field("frame_end", pa.int32()),
        pa.field("ts_start_ms", pa.int64()),
        pa.field("ts_end_ms", pa.int64()),
        pa.field("offense_team_cluster", pa.int8(), nullable=True),
        pa.field("outcome", pa.string(), nullable=True),
    ]
)

# Brain observation stream — one row per token, ordered by
# (t_ms, channel priority, entity_id). Contract:
# docs/cv-brain-token-contract.md. Primitives and external
# truth only; model verdicts never enter this stream.
TOKENS_SCHEMA = pa.schema(
    [
        pa.field("game_key", pa.string()),
        pa.field("token_idx", pa.int64()),
        pa.field("t_ms", pa.int64()),
        pa.field("channel", pa.string()),
        pa.field("entity_id", pa.int32(), nullable=True),
        pa.field("team_cluster", pa.int8(), nullable=True),
        pa.field("court_x", pa.float32(), nullable=True),
        pa.field("court_y", pa.float32(), nullable=True),
        pa.field("value", pa.float32(), nullable=True),
        pa.field("text", pa.string(), nullable=True),
        pa.field("conf", pa.float32(), nullable=True),
        pa.field("period", pa.int8(), nullable=True),
        pa.field("payload_json", pa.string(), nullable=True),
    ]
)

# Sim channel dense truth (cv-brain-token-contract §2 / plan §4): the
# forward-run state model's per-tick latent state. Paired with a
# POSITIONS_SCHEMA truth stage and a degraded TOKENS_SCHEMA observation
# stream under the same sim game_key. Diagnostics/training only — never
# headline eval.
SIM_STATES_SCHEMA = pa.schema(
    [
        pa.field("game_key", pa.string()),
        pa.field("tick", pa.int32()),
        pa.field("t_ms", pa.int64()),
        pa.field("period", pa.int8()),
        pa.field("clock_s", pa.float32()),
        pa.field("ball_mode", pa.string()),  # held|dribbled|ballistic|dead
        pa.field("ball_x", pa.float32()),
        pa.field("ball_y", pa.float32()),
        pa.field("holder_entity", pa.int32(), nullable=True),
        pa.field("possession_cluster", pa.int8()),
        pa.field("score_home", pa.int16()),
        pa.field("score_guest", pa.int16()),
        pa.field("dead_ball", pa.bool_()),
    ]
)

# Sim truth permutation, event-sourced: the true_entity -> observed_id
# mapping (sim_traces obs_map) at tick 0 for all 10 entities, then one
# row per entity whose mapping changed at a crossover swap. Replaying
# rows in tick order reconstructs the exact permutation at any tick —
# the supervision target for the slot/identity estimator (contract §1).
SIM_IDENTITY_SCHEMA = pa.schema(
    [
        pa.field("game_key", pa.string()),
        pa.field("tick", pa.int32()),
        pa.field("t_ms", pa.int64()),
        pa.field("true_entity", pa.int32()),
        pa.field("observed_id", pa.int32()),
    ]
)

# Announcer name-call extraction (whisper + exact surname match vs the
# ESPN boxscore roster). Shared surnames emit ambiguous=True with every
# candidate — never a guess.
NAME_CALLS_SCHEMA = pa.schema(
    [
        pa.field("game_key", pa.string()),
        pa.field("t_ms", pa.int64()),
        pa.field("surname", pa.string()),
        pa.field("athlete_ids", pa.list_(pa.string())),
        pa.field("jerseys", pa.list_(pa.string())),
        pa.field("ambiguous", pa.bool_()),
        pa.field("conf", pa.float32()),
        pa.field("segment_text", pa.string()),
    ]
)
