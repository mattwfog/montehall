"""Atomic events stage: geometry-translated positions + entity-grain events.

Post-possession, CPU-only, no LLM dependency — runs in harvest mode too.
Three artifacts:

    <out>/<job_id>/positions/      every localized on-court sample, entity
                                   grain for players, ungrouped for the ball
    <out>/<job_id>/ball_controls/  the shared control-sample computation
                                   (controls.py) persisted once
    <out>/<job_id>/atomic_events/  possession_start/end, control_gain,
                                   pass_candidate, steal_candidate

Zone semantics intentionally live downstream (run_plays): attacking
direction needs shot evidence, which does not exist yet at this point in
the chain on a fresh run. Positions here ARE the geometry translation —
court feet in the canonical frame (court.py).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from montehall_cv.court import CourtSpec
from montehall_cv.pipeline.controls import entity_control_samples
from montehall_cv.store.artifacts import ArtifactWriter, read_stage, stage_complete
from montehall_cv.store.records import DetClass, possession_offense_cluster
from montehall_cv.store.schemas import (
    ATOMIC_EVENTS_SCHEMA,
    BALL_CONTROLS_SCHEMA,
    POSITIONS_SCHEMA,
)

MAX_HANDOFF_GAP_MS = 2000  # traces.MAX_PASS_GAP_MS — same candidate semantics
ON_COURT_MARGIN_FT = 3.0

STAGES = ("positions", "ball_controls", "atomic_events")


def run(out_root: Path, job_id: str) -> dict:
    job_dir = out_root / job_id
    if all(stage_complete(job_dir / s) for s in STAGES):
        return {"job_id": job_id, "skipped": True, "reason": "stages already complete"}
    started = time.monotonic()

    entity_of = {
        row["track_id"]: row for row in read_stage(job_dir / "entities").to_pylist()
    }
    n_positions, n_ball = _write_positions(job_dir, job_id, entity_of)
    controls = entity_control_samples(job_dir, entity_of)
    _write_controls(job_dir, job_id, controls)
    events = _atomic_events(job_dir, controls)
    _write_events(job_dir, job_id, events)

    by_type: dict[str, int] = {}
    for e in events:
        by_type[e["event_type"]] = by_type.get(e["event_type"], 0) + 1
    return {
        "job_id": job_id,
        "positions": n_positions,
        "ball_positions": n_ball,
        "ball_controls": len(controls),
        "atomic_events": len(events),
        "events_by_type": by_type,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def _write_positions(
    job_dir: Path, job_id: str, entity_of: dict[int, dict]
) -> tuple[int, int]:
    """Localized detections -> court-frame samples. Players resolve to
    entity grain (merged identity groups); the ball rides along with
    entity_id None. Off-court and unlocalized rows drop here."""
    court = CourtSpec()
    writer = ArtifactWriter(job_dir / "positions", POSITIONS_SCHEMA)
    n_players = 0
    n_ball = 0
    for row in read_stage(job_dir / "localized").to_pylist():
        if row["court_x"] is None:
            continue
        if not court.contains(row["court_x"], row["court_y"], margin_ft=ON_COURT_MARGIN_FT):
            continue
        if row["cls"] == int(DetClass.BALL):
            entity_id, team = None, None
            n_ball += 1
        else:
            entity = entity_of.get(row["track_id"])
            if entity is None:
                continue
            entity_id, team = entity["entity_id"], entity["team_cluster"]
            n_players += 1
        writer.add(
            {
                "job_id": job_id,
                "frame_idx": row["frame_idx"],
                "ts_ms": row["ts_ms"],
                "cls": row["cls"],
                "entity_id": entity_id,
                "team_cluster": team,
                "court_x": row["court_x"],
                "court_y": row["court_y"],
                "court_conf": row["court_conf"],
            }
        )
    writer.close()
    return n_players, n_ball


def _write_controls(job_dir: Path, job_id: str, controls: list[dict]) -> None:
    writer = ArtifactWriter(job_dir / "ball_controls", BALL_CONTROLS_SCHEMA)
    writer.add_many({"job_id": job_id, **c} for c in controls)
    writer.close()


def _atomic_events(job_dir: Path, controls: list[dict]) -> list[dict]:
    """Possession boundaries + control-transition events, ts-ordered."""
    events: list[dict] = []

    for p in read_stage(job_dir / "possessions").to_pylist():
        cluster = possession_offense_cluster(p)
        for edge, frame, ts in (
            ("possession_start", p["frame_start"], p["ts_start_ms"]),
            ("possession_end", p["frame_end"], p["ts_end_ms"]),
        ):
            events.append(
                {
                    "event_type": edge,
                    "frame_idx": frame,
                    "ts_ms": ts,
                    "entity_id": None,
                    "team_cluster": cluster,
                    "court_x": None,
                    "court_y": None,
                    "possession_id": p["possession_id"],
                    "payload_json": json.dumps({"outcome": p["outcome"]})
                    if p["outcome"]
                    else None,
                }
            )

    possession_spans = [
        (p["possession_id"], p["ts_start_ms"], p["ts_end_ms"])
        for p in read_stage(job_dir / "possessions").to_pylist()
    ]

    def possession_at(ts_ms: int) -> int | None:
        for pid, start, end in possession_spans:
            if start <= ts_ms <= end:
                return pid
        return None

    prev = None
    for control in controls:
        if prev is None or prev["entity_id"] != control["entity_id"]:
            if prev is None or control["ts_ms"] - prev["ts_ms"] > MAX_HANDOFF_GAP_MS:
                event_type = "control_gain"
                payload = None
            else:
                gap_ms = control["ts_ms"] - prev["ts_ms"]
                if prev["team_cluster"] == control["team_cluster"]:
                    event_type = "pass_candidate"
                    payload = {"from_entity": prev["entity_id"], "gap_ms": gap_ms}
                else:
                    event_type = "steal_candidate"
                    payload = {
                        "from_entity": prev["entity_id"],
                        "from_team": prev["team_cluster"],
                        "gap_ms": gap_ms,
                    }
            events.append(
                {
                    "event_type": event_type,
                    "frame_idx": control["frame_idx"],
                    "ts_ms": control["ts_ms"],
                    "entity_id": control["entity_id"],
                    "team_cluster": control["team_cluster"],
                    "court_x": control["court_x"],
                    "court_y": control["court_y"],
                    "possession_id": possession_at(control["ts_ms"]),
                    "payload_json": json.dumps(payload) if payload else None,
                }
            )
        prev = control

    events.sort(key=lambda e: (e["ts_ms"], e["event_type"]))
    return events


def _write_events(job_dir: Path, job_id: str, events: list[dict]) -> None:
    writer = ArtifactWriter(job_dir / "atomic_events", ATOMIC_EVENTS_SCHEMA)
    writer.add_many(
        {"job_id": job_id, "event_id": i, **e} for i, e in enumerate(events)
    )
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.job_id), indent=2))


if __name__ == "__main__":
    main()
