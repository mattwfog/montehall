"""Atomic events stage on a synthetic job: geometry-translated positions,
shared ball controls, control-transition events, possession boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from montehall_cv.pipeline import run_events
from montehall_cv.pipeline.controls import entity_control_samples
from montehall_cv.store.artifacts import ArtifactWriter, read_stage
from montehall_cv.store.records import DetClass
from montehall_cv.store.schemas import (
    DETECTIONS_SCHEMA,
    POSSESSIONS_SCHEMA,
)

ENTITIES_SCHEMA_FIELDS = ["job_id", "track_id", "entity_id", "team_cluster"]


def _detection(frame: int, ts: int, det: int, cls: int, track: int | None,
               cx: float | None, cy: float | None) -> dict:
    return {
        "job_id": "j", "frame_idx": frame, "ts_ms": ts, "det_idx": det,
        "cls": cls, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0,
        "conf": 0.9, "track_id": track, "is_detected": True,
        "court_x": cx, "court_y": cy,
        "court_conf": 0.9 if cx is not None else None,
    }


@pytest.fixture
def job_dir(tmp_path: Path) -> Path:
    """Synthetic 3-frame job: two players (opposite teams) + a ball that
    moves from player 1's hands to player 2's across a team boundary."""
    job = tmp_path / "j"

    det = ArtifactWriter(job / "localized", DETECTIONS_SCHEMA)
    # frame 0 @ ts 0: ball next to track 1 (team 0)
    det.add(_detection(0, 0, 0, int(DetClass.PERSON), 1, 20.0, 25.0))
    det.add(_detection(0, 0, 1, int(DetClass.PERSON), 2, 60.0, 25.0))
    det.add(_detection(0, 0, 2, int(DetClass.BALL), None, 21.0, 25.0))
    # frame 1 @ ts 1000: ball next to track 2 (team 1) -> steal candidate
    det.add(_detection(1, 1000, 0, int(DetClass.PERSON), 1, 22.0, 25.0))
    det.add(_detection(1, 1000, 1, int(DetClass.PERSON), 2, 60.0, 25.0))
    det.add(_detection(1, 1000, 2, int(DetClass.BALL), None, 59.0, 25.0))
    # frame 2 @ ts 2000: unlocalized ball (dropped), off-court player (dropped)
    det.add(_detection(2, 2000, 0, int(DetClass.PERSON), 1, -20.0, 25.0))
    det.add(_detection(2, 2000, 1, int(DetClass.BALL), None, None, None))
    det.close()

    import pyarrow as pa

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
    entities.add({"job_id": "j", "track_id": 1, "entity_id": 100, "team_cluster": 0})
    entities.add({"job_id": "j", "track_id": 2, "entity_id": 200, "team_cluster": 1})
    entities.close()

    poss = ArtifactWriter(job / "possessions", POSSESSIONS_SCHEMA)
    poss.add(
        {
            "job_id": "j", "possession_id": 0, "frame_start": 0, "frame_end": 1,
            "ts_start_ms": 0, "ts_end_ms": 1500, "offense_team_cluster": 0,
            "outcome": None,
        }
    )
    poss.close()
    return job


def test_positions_translate_and_filter(job_dir: Path) -> None:
    summary = run_events.run(job_dir.parent, "j")
    rows = read_stage(job_dir / "positions").to_pylist()
    # 4 on-court player samples + 2 localized ball samples; the off-court
    # player (-20 ft) and the unlocalized ball never land
    players = [r for r in rows if r["entity_id"] is not None]
    balls = [r for r in rows if r["entity_id"] is None]
    assert len(players) == 4
    assert len(balls) == 2
    assert {r["entity_id"] for r in players} == {100, 200}
    assert summary["positions"] == 4 and summary["ball_positions"] == 2


def test_ball_controls_artifact(job_dir: Path) -> None:
    run_events.run(job_dir.parent, "j")
    controls = read_stage(job_dir / "ball_controls").to_pylist()
    observed = [c for c in controls if not c["interpolated"]]
    assert [c["entity_id"] for c in observed] == [100, 200]
    assert observed[0]["dist_ft"] == pytest.approx(1.0, abs=0.01)
    # the 1s gap between the two ball obs is interpolation-eligible; any
    # gap-filled control must carry the flag
    for c in controls:
        assert c["interpolated"] in (True, False)


def test_atomic_events_stream(job_dir: Path) -> None:
    run_events.run(job_dir.parent, "j")
    events = read_stage(job_dir / "atomic_events").to_pylist()
    by_type = {}
    for e in events:
        by_type.setdefault(e["event_type"], []).append(e)

    assert len(by_type["possession_start"]) == 1
    assert len(by_type["possession_end"]) == 1
    assert by_type["possession_start"][0]["team_cluster"] == 0

    # first control = gain; cross-team handoff within 2s = steal candidate
    assert len(by_type["control_gain"]) == 1
    assert by_type["control_gain"][0]["entity_id"] == 100
    steal = by_type["steal_candidate"][0]
    assert steal["entity_id"] == 200
    assert json.loads(steal["payload_json"])["from_entity"] == 100
    assert steal["possession_id"] == 0


def test_stage_is_resumable(job_dir: Path) -> None:
    run_events.run(job_dir.parent, "j")
    again = run_events.run(job_dir.parent, "j")
    assert again["skipped"] is True


def test_control_tie_equidistant_players(tmp_path: Path) -> None:
    """Two players exactly equidistant from the ball: min() must never fall
    through to comparing the player dicts (TypeError); first track wins."""
    job = tmp_path / "j"
    det = ArtifactWriter(job / "localized", DETECTIONS_SCHEMA)
    det.add(_detection(0, 0, 0, int(DetClass.PERSON), 1, 38.0, 25.0))
    det.add(_detection(0, 0, 1, int(DetClass.PERSON), 2, 42.0, 25.0))
    det.add(_detection(0, 0, 2, int(DetClass.BALL), None, 40.0, 25.0))
    det.close()
    entity_of = {
        1: {"entity_id": 100, "team_cluster": 0},
        2: {"entity_id": 200, "team_cluster": 1},
    }
    controls = entity_control_samples(job, entity_of)
    assert len(controls) == 1
    assert controls[0]["entity_id"] == 100
    assert controls[0]["dist_ft"] == pytest.approx(2.0)
