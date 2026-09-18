"""Digit assembly rules, abstention, and the prediction contract with a fake detector."""

import hashlib
from typing import ClassVar
import json

import cv2
import numpy as np

from soccerviz.candidates import jersey_digits as jd


def digit(label, x0, score=0.9, width=20):
    return {"label": label, "score": score, "bbox_xyxy": [x0, 10.0, x0 + width, 40.0]}


def test_reads_two_digits_left_to_right_regardless_of_detection_order():
    number, score, reason = jd.read_number([digit("7", 60, 0.95), digit("1", 30, 0.85)])
    assert (number, score, reason) == (17, 0.85, "digits_read_left_to_right")


def test_abstains_on_no_digits_too_many_and_leading_zero():
    assert jd.read_number([])[2] == "no_digit_detected"
    assert jd.read_number([digit("1", 0), digit("2", 30), digit("3", 60)])[2] == "too_many_digits"
    assert jd.read_number([digit("0", 0), digit("7", 30)])[2] == "leading_zero"
    assert jd.read_number([digit("1", 0, score=0.3)])[2] == "no_digit_detected"


def test_overlapping_boxes_keep_the_stronger_digit():
    number, score, _ = jd.read_number([digit("8", 30, 0.7), digit("3", 32, 0.9)])
    assert (number, score) == (3, 0.9)


def test_decision_thresholds_the_weakest_digit():
    assert jd.decision(17, 0.79)["status"] == "abstain"
    assert jd.decision(17, 0.8) == {
        "status": "provisional",
        "jersey_number": 17,
        "reason": "uncalibrated_digit_detections_no_named_identity",
    }
    assert jd.decision(None, 0.0)["jersey_number"] is None


class FakeDetector:
    metadata: ClassVar = {"name": "fake", "checkpoints": {}}

    def predict(self, crop, threshold):
        return [digit("2", 5, 0.9), digit("3", 30, 0.95)] if crop.mean() > 100 else []


def test_infer_writes_the_jersey_predictions_contract(tmp_path):
    crops = []
    for crop_id, value in (("a", 200), ("b", 20)):
        path = tmp_path / "crops" / f"{crop_id}.png"
        path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(path), np.full((48, 64, 3), value, dtype=np.uint8))
        crops.append(
            {
                "crop_id": crop_id,
                "crop_path": f"crops/{crop_id}.png",
                "crop_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = tmp_path / "crop-manifest.json"
    manifest.write_text(json.dumps({"schema": "jersey-crops/v1", "crops": crops}))
    output = jd.infer(manifest, None, tmp_path / "inference", device="cpu", detector=FakeDetector())
    by_id = {r["crop_id"]: r for r in output["predictions"]}
    assert by_id["a"]["jersey_number"] == 23 and by_id["a"]["status"] == "provisional"
    assert by_id["b"]["jersey_number"] is None and by_id["b"]["status"] == "abstain"
    assert output["schema"] == "jersey-predictions/v1" and output["complete_crop_manifest"]
    assert (
        json.loads((tmp_path / "inference/predictions.json").read_text())["predictions"]
        == output["predictions"]
    )
