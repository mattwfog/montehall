import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from soccerviz.candidates.detector_backends import (
    RFDETRDetector,
    UltralyticsDetector,
    nms,
    normalize_detections,
    tile_regions,
)


def normalize(boxes, scores, names, mode="coco"):
    return normalize_detections(
        boxes, scores, names, mode=mode, width=100, height=60, threshold=0.25
    )


def test_generic_classes_do_not_invent_soccer_roles():
    result = normalize([[1, 2, 10, 20]] * 3, [0.9, 0.8, 0.7], ["person", "sports ball", "car"])
    assert [x["label"] for x in result] == ["person", "ball"]
    assert all("role" not in x for x in result)
    with pytest.raises(ValueError, match="Soccer role"):
        normalize([[1, 2, 10, 20]], [0.9], ["player"])


def test_soccer_roles_clip_boxes_and_keep_threshold_boundary():
    result = normalize(
        [[-10, 2, 110, 70], [0, 0, 5, 5]], [0.25, 0.24], ["goalkeeper", "ball"], mode="soccer"
    )
    assert result == [
        {
            "bbox_xyxy": [0.0, 2.0, 100.0, 60.0],
            "score": 0.25,
            "label": "person",
            "source_class": "goalkeeper",
            "role": "goalkeeper",
        }
    ]
    with pytest.raises(ValueError, match="Unexpected soccer class"):
        normalize([[1, 2, 10, 20]], [0.9], ["car"], mode="soccer")


@pytest.mark.parametrize(
    "boxes,scores",
    [
        ([[np.nan, 0, 4, 5]], [0.4]),
        ([[0, 0, 4, 5]], [np.inf]),
        ([[0, 0, 4, 5]], [1.1]),
        ([[4, 0, 1, 5]], [0.5]),
    ],
)
def test_invalid_detector_output_is_not_silently_scored(boxes, scores):
    with pytest.raises(ValueError):
        normalize(boxes, scores, ["person"])


def test_nms_merges_tile_duplicates_but_keeps_different_classes():
    detections = [
        {"bbox_xyxy": [1, 1, 10, 10], "score": 0.9, "label": "ball"},
        {"bbox_xyxy": [2, 1, 10, 10], "score": 0.7, "label": "ball"},
        {"bbox_xyxy": [1, 1, 10, 10], "score": 0.8, "label": "person"},
    ]
    assert [d["score"] for d in nms(detections)] == [0.9, 0.8]
    assert tile_regions(100, 50) == [
        (0, 0, 60, 30),
        (40, 0, 100, 30),
        (0, 20, 60, 50),
        (40, 20, 100, 50),
    ]


def test_rf_predict_converts_bgr_to_rgb_and_preserves_source_boxes():
    received = {}

    def predict(image, **kwargs):
        received["image"] = image
        return SimpleNamespace(
            xyxy=np.array([[10, 5, 50, 20]]),
            confidence=np.array([0.8]),
            data={"class_name": np.array(["person"])},
        )

    backend = RFDETRDetector.__new__(RFDETRDetector)
    backend.model = SimpleNamespace(predict=predict)
    backend.metadata = {"no_object_filter": {"excluded_rows_including_warmup": 0}}
    backend.device, backend.threshold, backend.mode, backend.resolution = "cpu", 0.25, "coco", 512
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[:] = [3, 7, 11]
    detections, transforms = backend.predict(image)
    assert received["image"][0, 0].tolist() == [11, 7, 3]
    assert received["image"].flags.c_contiguous
    assert image[0, 0].tolist() == [3, 7, 11]
    assert detections[0]["bbox_xyxy"] == [10, 5, 50, 20]
    assert transforms[0]["scale_xy"] == [6.4, 12.8]


def test_rf_no_object_sentinel_is_counted_but_unknown_soccer_classes_still_fail():
    output = SimpleNamespace(
        xyxy=np.array([[1, 1, 5, 5], [4, 4, 8, 9]]),
        confidence=np.array([0.95, 0.7]),
        data={"class_name": np.array(["__background__", "player"])},
    )
    backend = RFDETRDetector.__new__(RFDETRDetector)
    backend.model = SimpleNamespace(predict=lambda image, **kwargs: output)
    backend.metadata = {"no_object_filter": {"excluded_rows_including_warmup": 0}}
    backend.device, backend.threshold, backend.mode, backend.resolution = "cpu", 0.25, "soccer", 576
    detections, _ = backend.predict(np.zeros((20, 30, 3), dtype=np.uint8))
    assert len(detections) == 1 and detections[0]["role"] == "player"
    assert backend.last_background_count == 1
    assert backend.metadata["no_object_filter"]["excluded_rows_including_warmup"] == 1
    output.data["class_name"][1] = "car"
    with pytest.raises(ValueError, match="Unexpected soccer class"):
        backend.predict(np.zeros((20, 30, 3), dtype=np.uint8))


def test_yolo_tiled_ball_offsets_and_discards_full_frame_ball():
    backend = UltralyticsDetector.__new__(UltralyticsDetector)
    backend.device, backend.resolution, backend.mode = "cpu", 1280, "soccer"
    backend.model, backend.ball_model, backend.tiled_ball = "people", "ball", True
    backend.ball_selection = "all"

    def predict(model, image, resolution, mode):
        return [
            {
                "bbox_xyxy": [1, 2, 5, 6],
                "score": 0.9,
                "label": "person" if model == "people" else "ball",
            }
        ]

    backend._predict = predict
    result, transforms = backend.predict(np.zeros((50, 100, 3), dtype=np.uint8))
    balls = [d["bbox_xyxy"] for d in result if d["label"] == "ball"]
    assert balls == [[1, 2, 5, 6], [41, 2, 45, 6], [1, 22, 5, 26], [41, 22, 45, 26]]
    assert len(transforms) == 5
    backend.ball_selection = "top1"
    result, _ = backend.predict(np.zeros((50, 100, 3), dtype=np.uint8))
    assert len([d for d in result if d["label"] == "ball"]) == 1


def test_runner_preserves_manifest_hash_with_path_remapping(tmp_path):
    import json

    import cv2

    from soccerviz.core.assets import sha256

    spec = importlib.util.spec_from_file_location(
        "detector_runner", Path(__file__).resolve().parents[2] / "scripts/benchmarks/run_detector_benchmark.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    completed = tmp_path / "completed.json"
    completed.write_text("existing predictions")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        runner.run(SimpleNamespace(out=completed))
    assert completed.read_text() == "existing predictions"
    image_path = tmp_path / "images" / "one.png"
    image_path.parent.mkdir()
    cv2.imwrite(str(image_path), np.zeros((20, 30, 3), dtype=np.uint8))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "vision-benchmark/v1",
                "frames": [
                    {
                        "sequence": "s",
                        "frame_id": 1,
                        "image_path": "/original/images/one.png",
                        "width": 30,
                        "height": 20,
                        "sha256": sha256(image_path),
                    }
                ],
            }
        )
    )
    before = sha256(manifest_path)
    frame = json.loads(manifest_path.read_text())["frames"][0]
    assert runner.load_image(frame, manifest_path, [(Path("/original"), tmp_path)]).shape == (
        20,
        30,
        3,
    )
    assert sha256(manifest_path) == before
    image_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA mismatch"):
        runner.load_image(frame, manifest_path, [(Path("/original"), tmp_path)])
