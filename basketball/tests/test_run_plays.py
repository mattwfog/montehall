"""Play payloads: trajectories, attacked-end binding, zone entries, and
the shot join — on a synthetic job exercising the real artifact readers."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from montehall_cv.pipeline import run_events, run_plays
from montehall_cv.store.artifacts import ArtifactWriter, read_stage
from montehall_cv.store.records import DetClass
from montehall_cv.store.schemas import (
    DETECTIONS_SCHEMA,
    IDENTITY_SCHEMA,
    POSSESSIONS_SCHEMA,
)

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


def _detection(frame: int, ts: int, det: int, cls: int, track: int | None,
               cx: float, cy: float) -> dict:
    return {
        "job_id": "j", "frame_idx": frame, "ts_ms": ts, "det_idx": det,
        "cls": cls, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0,
        "conf": 0.9, "track_id": track, "is_detected": True,
        "court_x": cx, "court_y": cy, "court_conf": 0.9,
    }


@pytest.fixture
def job_dir(tmp_path: Path) -> Path:
    """One possession by team 0 attacking the left basket: entity 100
    dribbles from the backcourt into the paint and shoots; the ball rides
    nearby so controls bind the shot to entity 100."""
    job = tmp_path / "j"

    det = ArtifactWriter(job / "localized", DETECTIONS_SCHEMA)
    # 60 -> 15 -> 10 ft: backcourt -> midrange-ish -> paint (left end)
    path_x = [60.0, 15.0, 10.0]
    for i, x in enumerate(path_x):
        ts = i * 1000
        det.add(_detection(i, ts, 0, int(DetClass.PERSON), 1, x, 25.0))
        det.add(_detection(i, ts, 1, int(DetClass.BALL), None, x + 1.0, 25.0))
        # a defender (team 1) hovering near its own basket
        det.add(_detection(i, ts, 2, int(DetClass.PERSON), 2, 12.0, 20.0))
    det.close()

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

    identity = ArtifactWriter(job / "identity", IDENTITY_SCHEMA)
    identity.add(
        {
            "job_id": "j", "track_id": 1, "candidate": "24", "prob": 0.9,
            "bound_at_stage": "ocr_vote", "bound_at_frame": None,
        }
    )
    identity.close()

    poss = ArtifactWriter(job / "possessions", POSSESSIONS_SCHEMA)
    poss.add(
        {
            "job_id": "j", "possession_id": 0, "frame_start": 0, "frame_end": 2,
            "ts_start_ms": 0, "ts_end_ms": 2500, "offense_team_cluster": 0,
            "outcome": None,
        }
    )
    poss.close()

    shots = ArtifactWriter(job / "shot_events", SHOT_EVENTS_SCHEMA)
    shots.add(
        {
            "job_id": "j", "event_id": 0, "frame_start": 2, "frame_end": 2,
            "ts_start_ms": 2200, "ts_end_ms": 2400, "court_end": "left",
            "n_ball_obs": 3, "verdict_attempt": True, "verdict_made": True,
            "confidence": 0.9, "rationale": "clean make",
        }
    )
    shots.close()

    run_events.run(tmp_path, "j")
    return job


def _payloads(job: Path) -> list[dict]:
    return [
        json.loads(r["payload_json"])
        for r in read_stage(job / "plays").to_pylist()
    ]


def test_play_binds_attacked_end_from_shot_evidence(job_dir: Path) -> None:
    summary = run_plays.run(job_dir.parent, "j")
    play = _payloads(job_dir)[0]
    assert play["attacked_end"] == "left"
    assert play["team_cluster"] == 0
    assert summary["plays_with_attacked_end"] == 1


def test_trajectories_cover_participants_and_ball(job_dir: Path) -> None:
    run_plays.run(job_dir.parent, "j")
    play = _payloads(job_dir)[0]
    assert set(play["trajectories"]) == {"100", "200", "ball"}
    assert len(play["trajectories"]["100"]) == 3  # 1s apart > 250ms gap
    jerseys = {p["entity_id"]: p["jersey"] for p in play["participants"]}
    assert jerseys == {100: "24", 200: None}


def test_zone_entries_track_the_drive(job_dir: Path) -> None:
    run_plays.run(job_dir.parent, "j")
    play = _payloads(job_dir)[0]
    zones_seq = [z["zone"] for z in play["zone_entries"]["100"]]
    assert zones_seq[0] == "backcourt"
    assert zones_seq[-1] == "paint"


def test_shot_event_joined_with_zone(job_dir: Path) -> None:
    run_plays.run(job_dir.parent, "j")
    play = _payloads(job_dir)[0]
    shot = [e for e in play["events"] if e["type"] == "shot"][0]
    assert shot["entity_id"] == 100
    assert shot["detail"]["made"] is True
    assert shot["detail"]["zone"] == "paint"
    assert shot["detail"]["is_three"] is False


def test_degrades_without_shot_evidence(job_dir: Path) -> None:
    import shutil

    shutil.rmtree(job_dir / "shot_events")
    shutil.rmtree(job_dir / "plays", ignore_errors=True)
    summary = run_plays.run(job_dir.parent, "j")
    play = _payloads(job_dir)[0]
    assert play["attacked_end"] is None
    assert play["zone_entries"] is None
    assert play["trajectories"]["100"]  # geometry still present
    assert summary["plays_with_attacked_end"] == 0
