"""Overlay keyframe tracks (B1): stat-line entities only, ~4Hz downsample."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa

from montehall_cv.pipeline import run_tracks
from montehall_cv.pipeline.run_boxscore import BOX_SCORE_SCHEMA
from montehall_cv.store.artifacts import ArtifactWriter
from montehall_cv.store.schemas import DETECTIONS_SCHEMA

ENTITIES_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("track_id", pa.int32()),
        pa.field("entity_id", pa.int32()),
        pa.field("team_cluster", pa.int8()),
    ]
)


def _write(stage_dir: Path, schema: pa.Schema, rows: list[dict]) -> None:
    writer = ArtifactWriter(stage_dir, schema)
    writer.add_many(rows)
    writer.close()


def _det(track: int, ts: int) -> dict:
    return {
        "job_id": "j", "frame_idx": ts // 33, "ts_ms": ts, "det_idx": 0,
        "cls": 0, "x1": 10.0, "y1": 20.0, "x2": 60.0, "y2": 140.0,
        "conf": 0.9, "track_id": track, "is_detected": True,
    }


def test_tracks_cover_line_entities_at_keyframe_rate(tmp_path, monkeypatch):
    job = tmp_path / "j"
    _write(job / "box_score", BOX_SCORE_SCHEMA, [
        {"job_id": "j", "team_cluster": 0, "player_key": "e50", "entity_id": 50,
         "fga": 1, "fgm": 0, "fga3": 0, "fgm3": 0, "oreb": 0, "dreb": 0,
         "points": 0, "confidence": 0.5},
        {"job_id": "j", "team_cluster": -1, "player_key": "team", "entity_id": None,
         "fga": 1, "fgm": 0, "fga3": 0, "fgm3": 0, "oreb": 0, "dreb": 0,
         "points": 0, "confidence": 0.5},
    ])
    _write(job / "entities", ENTITIES_SCHEMA, [
        {"job_id": "j", "track_id": 1, "entity_id": 50, "team_cluster": 0},
        {"job_id": "j", "track_id": 2, "entity_id": 99, "team_cluster": 1},
    ])
    # track 1 (entity 50): samples every 100ms -> 250ms downsample keeps ~1/3;
    # track 2 belongs to a NON-line entity and must not appear
    _write(job / "localized", DETECTIONS_SCHEMA,
           [_det(1, ts) for ts in range(0, 1001, 100)]
           + [_det(2, ts) for ts in range(0, 1001, 100)])

    monkeypatch.setattr(
        "montehall_cv.pipeline.video.probe",
        lambda _: SimpleNamespace(width=1920, height=1080, average_fps=30.0),
    )
    summary = run_tracks.run(Path("fake.mp4"), tmp_path, "j")

    assert summary["entities_tracked"] == 1
    payload = json.loads((job / "entity_tracks" / "tracks.json").read_text())
    assert payload["video"] == {"width": 1920, "height": 1080}
    keyframes = payload["entities"]["50"]
    assert "99" not in payload["entities"]
    # 100ms samples through the 250ms min-gap gate -> 0,300,600,900
    assert [k[0] for k in keyframes] == [0, 300, 600, 900]
    assert keyframes[0][1:] == [10.0, 20.0, 60.0, 140.0]


def test_missing_upstream_stage_skips_cleanly(tmp_path):
    summary = run_tracks.run(Path("fake.mp4"), tmp_path, "j")
    assert summary["skipped"] and "box_score" in summary["reason"]
