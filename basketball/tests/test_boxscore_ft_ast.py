"""Box-score integration: FT reclassification + assist credit, end to end
through run_boxscore and the adapter."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from montehall_cv.pipeline import run_boxscore
from montehall_cv.pipeline.run_freethrows import FT_EVENTS_SCHEMA
from montehall_cv.runner.adapter import build_results
from montehall_cv.store.artifacts import ArtifactWriter, read_stage
from montehall_cv.store.schemas import BALL_CONTROLS_SCHEMA, IDENTITY_SCHEMA

SHOT_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("event_id", pa.int32()),
        pa.field("frame_start", pa.int32()),
        pa.field("frame_end", pa.int32()),
        pa.field("ts_start_ms", pa.int64()),
        pa.field("ts_end_ms", pa.int64()),
        pa.field("court_end", pa.string(), nullable=True),
        pa.field("n_ball_obs", pa.int32()),
        pa.field("verdict_attempt", pa.bool_(), nullable=True),
        pa.field("verdict_made", pa.bool_(), nullable=True),
        pa.field("confidence", pa.float32(), nullable=True),
        pa.field("rationale", pa.string(), nullable=True),
    ]
)


def _control(ts: int, entity: int, team: int) -> dict:
    return {
        "job_id": "j", "frame_idx": ts // 33, "ts_ms": ts, "entity_id": entity,
        "team_cluster": team, "court_x": 12.0, "court_y": 25.0,
        "ball_x": 12.5, "ball_y": 25.0, "dist_ft": 0.5,
    }


def _shot(event_id: int, ts: int, made: bool = True) -> dict:
    return {
        "job_id": "j", "event_id": event_id, "frame_start": ts // 33,
        "frame_end": ts // 33 + 10, "ts_start_ms": ts, "ts_end_ms": ts + 400,
        "court_end": "left", "n_ball_obs": 4, "verdict_attempt": True,
        "verdict_made": made, "confidence": 0.9, "rationale": "test",
    }


@pytest.fixture
def job_dir(tmp_path: Path) -> Path:
    job = tmp_path / "j"

    entities = ArtifactWriter(
        job / "entities",
        pa.schema(
            [
                pa.field("job_id", pa.string()),
                pa.field("track_id", pa.int32()),
                pa.field("entity_id", pa.int32()),
                pa.field("team_cluster", pa.int8()),
            ]
        ),
    )
    for track, eid, team in ((1, 100, 0), (2, 200, 0), (3, 300, 1)):
        entities.add({"job_id": "j", "track_id": track, "entity_id": eid,
                      "team_cluster": team})
    entities.close()

    identity = ArtifactWriter(job / "identity", IDENTITY_SCHEMA)
    for track, jersey in ((1, "10"), (2, "24")):
        identity.add({"job_id": "j", "track_id": track, "candidate": jersey,
                      "prob": 0.9, "bound_at_stage": "ocr_vote",
                      "bound_at_frame": None})
    identity.close()

    controls = ArtifactWriter(job / "ball_controls", BALL_CONTROLS_SCHEMA)
    # play 1: e100 passes to e200 who makes a 2 (shot at 2000)
    controls.add(_control(500, 100, 0))
    controls.add(_control(1200, 200, 0))
    # play 2: e200 controls at the line before the FT window at 30000
    controls.add(_control(29_000, 200, 0))
    controls.close()

    shots = ArtifactWriter(job / "shot_events", SHOT_EVENTS_SCHEMA)
    shots.add(_shot(0, 2000, made=True))   # assisted FGA2
    shots.add(_shot(1, 30_000, made=True))  # free throw
    shots.close()

    fts = ArtifactWriter(job / "ft_events", FT_EVENTS_SCHEMA)
    fts.add({"job_id": "j", "event_id": 0, "is_ft": False, "shooter_entity": 200,
             "dist_to_ft_point_ft": 15.0, "max_disp_ft": 20.0,
             "lane_lineup_count": 0, "confidence": 0.0})
    fts.add({"job_id": "j", "event_id": 1, "is_ft": True, "shooter_entity": 200,
             "dist_to_ft_point_ft": 0.4, "max_disp_ft": 1.0,
             "lane_lineup_count": 4, "confidence": 1.0})
    fts.close()
    return job


def test_ft_reclassified_and_assist_credited(job_dir: Path) -> None:
    summary = run_boxscore.run(job_dir.parent, "j")
    assert summary["free_throws"] == 1
    assert summary["assists"] == 1

    rows = {
        (r["team_cluster"], r["player_key"]): r
        for r in read_stage(job_dir / "box_score").to_pylist()
    }
    shooter = rows[(0, "24")]
    assert shooter["fga"] == 1 and shooter["fgm"] == 1  # the FT is NOT an FGA
    assert shooter["fta"] == 1 and shooter["ftm"] == 1
    assert shooter["points"] == 3  # 2 + 1, not 4
    passer = rows[(0, "10")]
    assert passer["ast"] == 1

    types = {e["event_type"] for e in read_stage(job_dir / "box_events").to_pylist()}
    assert {"FGA", "FGM", "AST", "FTA", "FTM"} <= types


def test_adapter_projects_ft_and_assist_totals(job_dir: Path) -> None:
    run_boxscore.run(job_dir.parent, "j")
    results = build_results(job_dir, 30.0)
    totals = results["stats"]["team_totals"]["team_a"]
    assert totals["fta"] == 1 and totals["ftm"] == 1 and totals["assist"] == 1
    shooter = results["stats"]["player_totals"]["team_a"]["24"]
    assert shooter["fta"] == 1 and shooter["ftm"] == 1
    passer_events = results["stats"]["player_events"]["team_a"]["10"]
    assert "AST" in passer_events  # frame-list shape the app transformer reads


def test_without_ft_stage_everything_stays_field_goal(job_dir: Path) -> None:
    import shutil

    shutil.rmtree(job_dir / "ft_events")
    summary = run_boxscore.run(job_dir.parent, "j")
    assert summary["free_throws"] == 0
    rows = {
        (r["team_cluster"], r["player_key"]): r
        for r in read_stage(job_dir / "box_score").to_pylist()
    }
    assert rows[(0, "24")]["fga"] == 2  # legacy behavior preserved
