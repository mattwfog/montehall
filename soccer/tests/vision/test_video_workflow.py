from typing import ClassVar

import av
import numpy as np
import pandas as pd
import pytest

from soccerviz.vision.specialists import import_video
from soccerviz.vision.video_workflow import run_workflow


class Detector:
    metadata: ClassVar = {"checkpoints": {}, "name": "synthetic test only"}

    def predict(self, image):
        people = [
            {"label": "person", "role": "player", "score": 0.9, "bbox_xyxy": [x, 40, x + 10, 70]}
            for x in (40, 70, 100, 130)
        ]
        return people + [{"label": "ball", "score": 0.8, "bbox_xyxy": [80, 60, 85, 65]}], []


class Camera:
    metadata: ClassVar = {"checkpoints": {}, "name": "synthetic test only"}

    def predict(self, image):
        return {"accepted": True, "homography": [[0.2, 0, -25], [0, 0.2, -20], [0, 0, 1]]}


def test_generated_clip_runs_to_harness_compatible_tables_and_preview(tmp_path):
    source = tmp_path / "synthetic.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=5)
        stream.width, stream.height, stream.pix_fmt = 192, 108, "yuv420p"
        for _ in range(5):
            frame = av.VideoFrame.from_ndarray(np.full((108, 192, 3), 90, np.uint8), format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    out = tmp_path / "run"
    report = run_workflow(
        source, out, seconds=1, hz=5, jersey=False, detector=Detector(), camera=Camera()
    )
    assert report["frames"] == 5
    assert (out / "preview.mp4").stat().st_size > 100
    frames = pd.read_parquet(out / "frames.parquet")
    assert frames.ball_status.tolist() == [
        "tentative",
        "detected",
        "detected",
        "detected",
        "detected",
    ]
    imported = import_video(out, 0, 1)
    assert imported["source_sha256"] == report["source_sha256"]
    assert len(imported["tables"]["state"]) == 25
    from soccerviz.harness.colony_runtime import execute_run
    from soccerviz.harness.engine import Harness
    from soccerviz.ui.video_workflow_ui import save_shape_review, view

    harness = Harness(tmp_path / "harness")
    run = harness.create(imported)
    execute_run(harness, run)
    status = harness.status(run)
    state_id = next(r["output_id"] for r in status["results"] if r["stage"] == "state")
    shared = harness.get(state_id)["payload"]
    assert shared["frames"][0]["ball_hypotheses"][0]["xy_m"] is None
    assert shared["frames"][1]["ball_hypotheses"][0]["xy_m"] is not None
    displayed = view(str(out), 3)
    assert displayed[4]["maximum"] == 4
    before = (out / "files.json").read_bytes()
    save_shape_review(
        tmp_path,
        str(out),
        0,
        0.8,
        "unknown",
        "unknown",
        "unknown",
        "low",
        "Synthetic test only",
        "Test reviewer",
    )
    assert (out / "files.json").read_bytes() == before
    assert len(list((tmp_path / "shape-reviews").glob("*.json"))) == 1
    with pytest.raises(ValueError, match="author"):
        save_shape_review(
            tmp_path, str(out), 0, 0.8, "unknown", "unknown", "unknown", "low", "", ""
        )
    with pytest.raises(FileExistsError):
        run_workflow(source, out, detector=Detector(), camera=Camera())
