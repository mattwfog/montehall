import json
from typing import ClassVar

import cv2
import numpy as np
import pytest

from soccerviz.candidates.pose_model import (
    COCO_KEYPOINTS,
    body_extremes,
    box_xyxy_to_xywh,
    frame_person_boxes,
    infer,
    inside_box,
    summarize,
    validate_keypoints,
)
from soccerviz.core.assets import sha256


def upright_pose(x=10.0, top=10.0, height=80.0):
    """A plausible standing figure: head at top, hips mid, ankles at the bottom."""
    rows = {
        "nose": 0.05, "left_eye": 0.04, "right_eye": 0.04, "left_ear": 0.05, "right_ear": 0.05,
        "left_shoulder": 0.2, "right_shoulder": 0.2, "left_elbow": 0.35, "right_elbow": 0.35,
        "left_wrist": 0.5, "right_wrist": 0.5, "left_hip": 0.5, "right_hip": 0.5,
        "left_knee": 0.75, "right_knee": 0.75, "left_ankle": 0.97, "right_ankle": 0.99,
    }  # fmt: skip
    points = [[x + (i % 2), top + rows[name] * height] for i, name in enumerate(COCO_KEYPOINTS)]
    return points, [0.9] * len(COCO_KEYPOINTS)


def test_box_conversion_clips_to_image_and_rejects_degenerate():
    assert box_xyxy_to_xywh([-5, 2, 20, 40], 15, 40) == [0, 2, 15, 38]
    for box in ([1, 2, 1, 40], [50, 10, 60, 40], [0, 0, np.nan, 40]):
        with pytest.raises(ValueError):
            box_xyxy_to_xywh(box, 20, 40)


def test_keypoint_validation_requires_17_finite_bounded_scores():
    points, scores = upright_pose()
    validate_keypoints(points, scores)
    with pytest.raises(ValueError, match="17 finite COCO"):
        validate_keypoints(points[:-1], scores[:-1])
    with pytest.raises(ValueError, match="scores"):
        validate_keypoints(points, [1.5] + scores[1:])


def test_body_extremes_exclude_arms_and_need_a_confident_foot():
    points, scores = upright_pose()
    result = body_extremes(points, scores, 0.3)
    assert result["lowest_foot"]["name"] == "right_ankle"
    names = {e["name"] for e in result["eligible_points"]}
    assert not names & {"left_wrist", "right_wrist", "left_elbow", "right_elbow"}
    assert "nose" in names and "left_hip" in names
    scores = list(scores)
    scores[COCO_KEYPOINTS.index("left_ankle")] = 0.0
    scores[COCO_KEYPOINTS.index("right_ankle")] = 0.1
    assert body_extremes(points, scores, 0.3) is None


def test_inside_box_uses_margin():
    assert inside_box([0, 0], [0, 0, 10, 10])
    assert inside_box([-0.5, 5], [0, 0, 10, 10])
    assert not inside_box([-2, 5], [0, 0, 10, 10])


def benchmark(tmp_path, frames=3, people=2):
    manifest_frames, predictions = [], []
    for identifier in range(1, frames + 1):
        path = tmp_path / f"{identifier}.png"
        cv2.imwrite(str(path), np.full((40, 60, 3), identifier, dtype=np.uint8))
        manifest_frames.append(
            {
                "sequence": "s",
                "frame_id": identifier,
                "image_path": str(path),
                "sha256": sha256(path),
                "width": 60,
                "height": 40,
                "fps": 25.0,
                "timestamp_s": (identifier - 1) / 25,
            }
        )
        predictions.append(
            {
                "sequence": "s",
                "frame_id": identifier,
                "image_sha256": sha256(path),
                "detections": [
                    {"label": "ball", "score": 0.9, "bbox_xyxy": [1, 1, 3, 3]},
                    *[
                        {
                            "label": "person",
                            "role": "player",
                            "score": 0.8,
                            "bbox_xyxy": [10 * k, 5, 10 * k + 8, 35],
                        }
                        for k in range(1, people + 1)
                    ],
                ],
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "vision-benchmark/v1", "frames": manifest_frames}))
    detections = tmp_path / "detections.json"
    detections.write_text(
        json.dumps(
            {
                "schema": "vision-predictions/v1",
                "manifest_sha256": sha256(manifest),
                "backend": {"name": "fixture"},
                "frames": predictions,
            }
        )
    )
    return manifest, detections


def test_person_boxes_skip_non_person_and_reject_changed_image_hash(tmp_path):
    manifest, detections = benchmark(tmp_path)
    rows = frame_person_boxes(json.loads(manifest.read_text()), json.loads(detections.read_text()))
    assert [len(r["people"]) for r in rows] == [2, 2, 2]
    assert rows[0]["people"][0]["detection_index"] == 1
    data = json.loads(detections.read_text())
    data["frames"][0]["image_sha256"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        frame_person_boxes(json.loads(manifest.read_text()), data)


class FakeBackend:
    calls: ClassVar[list[int]] = []

    def __init__(self, name, cache_dir, *, device):
        self.metadata = {"name": name}
        self.last_timing_ms = {"total": 1.0}

    def predict(self, image, boxes):
        FakeBackend.calls.append(int(image[0, 0, 0]))
        poses = []
        for box in boxes:
            points, scores = upright_pose(x=box[0], top=box[1], height=box[3])
            poses.append({"keypoints_xy": points, "scores": scores})
        return poses


def test_infer_persists_each_frame_then_resumes_and_refuses_overwrite(tmp_path):
    manifest, detections = benchmark(tmp_path)
    out = tmp_path / "pose"
    FakeBackend.calls = []
    partial = infer(manifest, detections, out, limit=2, backend_factory=FakeBackend)
    assert partial["complete_manifest"] is False
    assert FakeBackend.calls == [1, 2]
    assert len((out / "frames.jsonl").read_text().splitlines()) == 2
    (out / "predictions.json").unlink()
    full = infer(manifest, detections, out, backend_factory=FakeBackend)
    assert FakeBackend.calls == [1, 2, 3]
    assert full["complete_manifest"] is True
    assert full["runtime"]["resumed_frames"] == 2
    assert [f["frame_id"] for f in full["frames"]] == [1, 2, 3]
    assert full["frames"][0]["people"][0]["detection_index"] == 1
    assert len(full["frames"][0]["people"][0]["keypoints_xy"]) == 17
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        infer(manifest, detections, out, backend_factory=FakeBackend)
    report = summarize(out / "predictions.json", tmp_path / "summary.json")
    assert report["kind"] == "sanity_rates_not_accuracy"
    assert report["people"] == 6
    assert report["keypoints_inside_box_rate"] == 1.0
    assert report["ankles_below_hips_rate_when_confident"] == 1.0
    assert report["people_with_confident_foot_rate"] == 1.0
    assert report["people_by_role"] == {"player": 6}


def test_infer_rejects_detections_not_bound_to_manifest(tmp_path):
    manifest, detections = benchmark(tmp_path)
    data = json.loads(detections.read_text())
    data["manifest_sha256"] = "other"
    detections.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not bound"):
        infer(manifest, detections, tmp_path / "pose", backend_factory=FakeBackend)
