"""Renderer smoke: real decode -> overlay draw -> h264 encode on a tiny
synthetic clip, with every overlay source present (boxes, jersey label,
ball, rim, possession banner, box_events ticker, minimap)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pytest

from montehall_cv.pipeline import render
from montehall_cv.pipeline.run_boxscore import BOX_EVENTS_SCHEMA
from montehall_cv.store.artifacts import ArtifactWriter
from montehall_cv.store.records import DetClass
from montehall_cv.store.schemas import (
    DETECTIONS_SCHEMA,
    IDENTITY_SCHEMA,
    POSSESSIONS_SCHEMA,
)

W, H, N_FRAMES = 128, 96, 4


def _write_video(path: Path) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 1000)
        for i in range(N_FRAMES):
            img = np.full((H, W, 3), 60 + 10 * i, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i * 100
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


@pytest.fixture
def job(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "video.mp4"
    _write_video(video)
    job_dir = tmp_path / "j"

    det = ArtifactWriter(job_dir / "localized", DETECTIONS_SCHEMA)
    for i in range(N_FRAMES):
        ts = i * 100
        base = {
            "job_id": "j", "frame_idx": i, "ts_ms": ts, "conf": 0.9,
            "is_detected": True, "court_conf": 0.9,
        }
        det.add({**base, "det_idx": 0, "cls": int(DetClass.PERSON), "track_id": 1,
                 "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 60.0,
                 "court_x": 20.0, "court_y": 25.0})
        det.add({**base, "det_idx": 1, "cls": int(DetClass.BALL), "track_id": None,
                 "x1": 40.0 + i, "y1": 30.0, "x2": 48.0 + i, "y2": 38.0,
                 "court_x": 22.0, "court_y": 25.0})
        det.add({**base, "det_idx": 2, "cls": int(DetClass.RIM), "track_id": None,
                 "x1": 60.0, "y1": 10.0, "x2": 80.0, "y2": 20.0,
                 "court_x": None, "court_y": None, "court_conf": None})
    det.close()

    entities = ArtifactWriter(
        job_dir / "entities",
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
    entities.close()

    identity = ArtifactWriter(job_dir / "identity", IDENTITY_SCHEMA)
    identity.add({"job_id": "j", "track_id": 1, "candidate": "24", "prob": 0.9,
                  "bound_at_stage": "ocr_vote", "bound_at_frame": None})
    identity.close()

    poss = ArtifactWriter(job_dir / "possessions", POSSESSIONS_SCHEMA)
    poss.add({"job_id": "j", "possession_id": 0, "frame_start": 0, "frame_end": 3,
              "ts_start_ms": 0, "ts_end_ms": 300, "offense_team_cluster": 0,
              "outcome": None})
    poss.close()

    box = ArtifactWriter(job_dir / "box_events", BOX_EVENTS_SCHEMA)
    box.add({"job_id": "j", "frame_idx": 1, "ts_ms": 100, "team_cluster": 0,
             "player_key": "24", "event_type": "FGM", "subtype": "FGM2"})
    box.close()

    return video, job_dir


def test_render_produces_playable_video(job: tuple[Path, Path]) -> None:
    video, job_dir = job
    summary = render.run(video, job_dir.parent, "j")
    assert summary["frames"] == N_FRAMES
    out = job_dir / "annotated" / "annotated.mp4"
    assert out.exists() and summary["bytes"] > 0

    with av.open(str(out)) as container:
        decoded = list(container.decode(container.streams.video[0]))
    assert len(decoded) == N_FRAMES
    assert decoded[0].width == W and decoded[0].height == H


def test_render_scale_halves_dimensions(job: tuple[Path, Path]) -> None:
    video, job_dir = job
    render.run(video, job_dir.parent, "j", scale=0.5)
    out = job_dir / "annotated" / "annotated.mp4"
    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        assert stream.width == W // 2 and stream.height == H // 2


def test_render_is_resumable(job: tuple[Path, Path]) -> None:
    video, job_dir = job
    render.run(video, job_dir.parent, "j")
    again = render.run(video, job_dir.parent, "j")
    assert again["skipped"] is True


def test_render_survives_missing_llm_artifacts(job: tuple[Path, Path]) -> None:
    """Harvest-mode jobs have no box_events/possessions — renderer still runs."""
    import shutil

    video, job_dir = job
    shutil.rmtree(job_dir / "box_events")
    shutil.rmtree(job_dir / "possessions")
    summary = render.run(video, job_dir.parent, "j")
    assert summary["frames"] == N_FRAMES
