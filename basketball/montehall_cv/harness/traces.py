"""Possession trace assembly: the symbolic record the LLM harness reasons over.

A trace is everything the pipeline knows about one possession, as compact
JSON — no pixels. The harness adjudicates traces against the rulebook;
pixels are only fetched by the (separate) VLM tier when a trace is flagged.
"""

from __future__ import annotations

import json
from pathlib import Path

from montehall_cv.pipeline.controls import entity_control_samples
from montehall_cv.store.artifacts import read_stage, stage_complete
from montehall_cv.store.records import possession_offense_cluster


def build_traces(job_dir: Path) -> list[dict]:
    possessions = read_stage(job_dir / "possessions").to_pylist()
    shot_events = read_stage(job_dir / "shot_events").to_pylist()
    entities = {r["track_id"]: r for r in read_stage(job_dir / "entities").to_pylist()}
    jerseys = _jersey_by_entity(job_dir, entities)
    controls = _controls(job_dir, entities)

    traces = []
    for possession in possessions:
        span = (possession["ts_start_ms"], possession["ts_end_ms"])
        p_controls = [c for c in controls if span[0] <= c["ts_ms"] <= span[1]]
        p_shots = [
            {
                "ts_s": round(e["ts_start_ms"] / 1000, 1),
                "attempt": e["verdict_attempt"],
                "made": e["verdict_made"],
                "confidence": e["confidence"],
                "court_end": e["court_end"],
                "rationale": e["rationale"],
            }
            for e in shot_events
            if e["verdict_attempt"]
            and span[0] - 2000 <= e["ts_start_ms"] <= span[1] + 2000
        ]
        involved = sorted({c["entity_id"] for c in p_controls})
        passes = _passes(p_controls)
        traces.append(
            {
                "possession_id": possession["possession_id"],
                "start_s": round(span[0] / 1000, 1),
                "end_s": round(span[1] / 1000, 1),
                "duration_s": round((span[1] - span[0]) / 1000, 1),
                "offense_team_cluster": possession_offense_cluster(possession),
                "ball_controls": [
                    {
                        "ts_s": round(c["ts_ms"] / 1000, 1),
                        "entity": c["entity_id"],
                        "team": c["team_cluster"],
                        "court_x": round(c["court_x"], 1),
                        "court_y": round(c["court_y"], 1),
                    }
                    for c in p_controls
                ],
                "shot_events": p_shots,
                "passes": passes,
                "entities_involved": [
                    {"entity": eid, "jersey": jerseys.get(eid)} for eid in involved
                ],
            }
        )
    return traces


MAX_PASS_GAP_MS = 2000


def _passes(controls: list[dict]) -> list[dict]:
    """Same-team control transitions within a short gap = pass candidates.

    Candidates only — occlusion gaps can masquerade as passes; the harness
    weighs them against the rest of the trace rather than trusting them.
    """
    passes = []
    for prev, cur in zip(controls, controls[1:], strict=False):
        if (
            prev["entity_id"] != cur["entity_id"]
            and prev["team_cluster"] == cur["team_cluster"]
            and 0 < cur["ts_ms"] - prev["ts_ms"] <= MAX_PASS_GAP_MS
        ):
            passes.append(
                {
                    "ts_s": round(cur["ts_ms"] / 1000, 1),
                    "from_entity": prev["entity_id"],
                    "to_entity": cur["entity_id"],
                    "to_court_x": round(cur["court_x"], 1),
                    "to_court_y": round(cur["court_y"], 1),
                }
            )
    return passes


def _jersey_by_entity(job_dir: Path, entities: dict) -> dict[int, str]:
    best: dict[int, tuple[str, float]] = {}
    for row in read_stage(job_dir / "identity").to_pylist():
        entity = entities.get(row["track_id"])
        if entity is None:
            continue
        eid = entity["entity_id"]
        if eid not in best or row["prob"] > best[eid][1]:
            best[eid] = (row["candidate"], row["prob"])
    return {eid: jersey for eid, (jersey, prob) in best.items() if prob >= 0.6}


def _controls(job_dir: Path, entities: dict) -> list[dict]:
    """Shared computation (controls.py); the persisted ball_controls
    artifact wins when the events stage already ran."""
    if stage_complete(job_dir / "ball_controls"):
        return sorted(
            read_stage(job_dir / "ball_controls").to_pylist(),
            key=lambda r: r["ts_ms"],
        )
    return entity_control_samples(job_dir, entities)


def trace_json(trace: dict) -> str:
    return json.dumps(trace, separators=(",", ":"))
