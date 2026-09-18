"""Primitive-store record contracts (design rulings D2/D3/D5).

The tracklet is the first-class object; identity is a late-bound attribute
bundle. Committed identity is a projection over IdentityHypothesis rows,
never a stored primitive. Every automated verdict lands as an Evidence row
so confidence-gated exposure has an audit trail.

Timing contract: ts_ms comes from the container pts (VFR-safe). frame_idx
is a decode ordinal — never use it to derive wall time (corpus has
variable-frame-rate captures averaging 35.7fps).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DetClass(IntEnum):
    PERSON = 0  # any human before role classification (player/ref/bench)
    BALL = 1
    REF = 2
    RIM = 3  # emitted by fine-tuned detectors only (COCO has no rim class)


EvidenceSource = Literal[
    "ocr_read", "reid_match", "spatial", "substitution", "roster", "llm_verdict", "manual"
]
BindingStage = Literal["ocr_vote", "reid", "ilp", "llm", "human", "roster"]


class Detection(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    frame_idx: int = Field(ge=0)
    ts_ms: int = Field(ge=0)
    det_idx: int = Field(ge=0)
    cls: DetClass
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = Field(ge=0.0, le=1.0)
    track_id: int | None = None
    is_detected: bool = True  # False = interpolated/extrapolated, not observed
    court_x: float | None = None
    court_y: float | None = None
    court_conf: float | None = None


class TrackletSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    track_id: int = Field(ge=0)
    cls: DetClass
    frame_start: int = Field(ge=0)
    frame_end: int = Field(ge=0)
    ts_start_ms: int = Field(ge=0)
    ts_end_ms: int = Field(ge=0)
    n_detections: int = Field(gt=0)
    mean_conf: float = Field(ge=0.0, le=1.0)


class IdentityHypothesis(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    track_id: int = Field(ge=0)
    candidate: str  # jersey number or player_id key, per binding stage
    prob: float = Field(ge=0.0, le=1.0)
    bound_at_stage: BindingStage
    bound_at_frame: int | None = None


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    track_id: int = Field(ge=0)
    frame_idx: int | None = None
    source_kind: EvidenceSource
    value_json: str
    weight: float = 1.0


class Possession(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    possession_id: int = Field(ge=0)
    frame_start: int = Field(ge=0)
    frame_end: int = Field(ge=0)
    ts_start_ms: int = Field(ge=0)
    ts_end_ms: int = Field(ge=0)
    # The k-means appearance cluster controlling the ball — an OBSERVED
    # int, not a court end. Which basket that team attacks is a downstream
    # join against shot evidence (zones.derive_attacked_ends), never
    # stored here (the stage runs before any shot event exists).
    offense_team_cluster: int | None = None
    outcome: str | None = None


def possession_offense_cluster(row: dict) -> int | None:
    """Offense team cluster from a possession row, old artifacts included.

    Rows written before 2026-07-09 carry the cluster id costumed as an
    "offense_team" court end ("left"=0, "right"=1 — the finding-2 defect);
    reading them back through this helper keeps them usable without
    re-running the stage.
    """
    if row.get("offense_team_cluster") is not None:
        return int(row["offense_team_cluster"])
    legacy = row.get("offense_team")
    if legacy in ("left", "right"):
        return 0 if legacy == "left" else 1
    return None
