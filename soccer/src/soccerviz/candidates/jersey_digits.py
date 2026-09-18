"""Permissive jersey-number reader: RF-DETR digit boxes on torso crops → number hypothesis.

Same prediction contract as `jersey_model.infer` (Uncertainty-JNR, CC BY-NC-SA with
SoccerNet weights) so the video workflow and the 200-crop evaluator consume it
unchanged: one record per crop with `pred_number`, `pred_score`, `status`
(`provisional` or `abstain`), `jersey_number` and `reason`.
"""

from __future__ import annotations

import importlib.metadata
import json
import time
from pathlib import Path

import cv2
import numpy as np

from soccerviz.core.assets import sha256

DIGITS = tuple(str(d) for d in range(10))
DEFAULT_THRESHOLDS = {
    "digit_score": 0.5,
    "min_number_score": 0.8,
    "max_digits": 2,
    "overlap_iou": 0.5,
}


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_number(digits, thresholds=DEFAULT_THRESHOLDS):
    """Digit boxes ({label, score, bbox_xyxy}) → (number, score, reason).

    Keeps digits above the score floor, drops the weaker of two overlapping boxes,
    orders left to right, and abstains on zero or more than `max_digits` digits or a
    leading zero on a two-digit number. The score is the weakest kept digit's.
    """
    kept = []
    for digit in sorted(digits, key=lambda d: -d["score"]):
        if digit["label"] not in DIGITS or digit["score"] < thresholds["digit_score"]:
            continue
        if any(_iou(digit["bbox_xyxy"], k["bbox_xyxy"]) > thresholds["overlap_iou"] for k in kept):
            continue
        kept.append(digit)
    if not kept:
        return None, 0.0, "no_digit_detected"
    if len(kept) > thresholds["max_digits"]:
        return None, 0.0, "too_many_digits"
    kept.sort(key=lambda d: (d["bbox_xyxy"][0] + d["bbox_xyxy"][2]) / 2)
    text = "".join(d["label"] for d in kept)
    if len(text) == 2 and text[0] == "0":
        return None, 0.0, "leading_zero"
    return int(text), float(min(d["score"] for d in kept)), "digits_read_left_to_right"


def decision(number, score, thresholds=DEFAULT_THRESHOLDS):
    if number is None or score < thresholds["min_number_score"]:
        return {
            "status": "abstain",
            "jersey_number": None,
            "reason": "insufficient_digit_evidence_not_a_legibility_label",
        }
    return {
        "status": "provisional",
        "jersey_number": int(number),
        "reason": "uncalibrated_digit_detections_no_named_identity",
    }


class DigitDetector:
    """Trained RF-DETR digit checkpoint; predict(crop_bgr) → digit boxes in crop pixels."""

    def __init__(self, checkpoint, device="cuda:0", resolution=576):
        import rfdetr

        self.model = rfdetr.RFDETRMedium.from_checkpoint(
            str(checkpoint), device=device, resolution=resolution
        )
        self.class_names = list(self.model.class_names)
        if not set(DIGITS) <= set(self.class_names):
            raise ValueError(f"Digit checkpoint lacks the ten digit classes: {self.class_names}")
        self.metadata = {
            "name": "rfdetr-medium-digits",
            "checkpoints": {
                "digits": {"path": str(Path(checkpoint).resolve()), "sha256": sha256(checkpoint)}
            },
            "class_names": self.class_names,
            "resolution": resolution,
            "device": device,
            "versions": {p: importlib.metadata.version(p) for p in ("rfdetr", "torch")},
        }

    def predict(self, crop_bgr, threshold):
        detections = self.model.predict(
            np.ascontiguousarray(crop_bgr[:, :, ::-1]), threshold=threshold
        )
        return [
            {
                "label": self.class_names[int(class_id)],
                "score": float(score),
                "bbox_xyxy": [float(v) for v in box],
            }
            for box, class_id, score in zip(
                detections.xyxy, detections.class_id, detections.confidence
            )
        ]


def infer(
    crop_manifest,
    checkpoint,
    out,
    device="cuda:0",
    thresholds=DEFAULT_THRESHOLDS,
    detector=None,
    resolution=576,
):
    """Read every crop of a jersey crop manifest; writes out/predictions.json.

    `resolution` must be the one the checkpoint was trained at (training-evidence.json).
    """
    manifest = json.loads(Path(crop_manifest).read_text())
    root = Path(crop_manifest).parent
    detector = detector or DigitDetector(checkpoint, device, resolution)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    for row in manifest["crops"]:
        path = root / row["crop_path"]
        if sha256(path) != row["crop_sha256"]:
            raise ValueError(f"Crop hash mismatch: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable crop: {path}")
        digits = detector.predict(image, thresholds["digit_score"])
        number, score, reason = read_number(digits, thresholds)
        results.append(
            {
                "crop_id": row["crop_id"],
                "digits": digits,
                "pred_number": number,
                "pred_score": score,
                "read_reason": reason,
                **decision(number, score, thresholds),
            }
        )
    output = {
        "schema": "jersey-predictions/v1",
        "backend": "rfdetr-medium-digits-pusan",
        "crop_manifest_sha256": sha256(crop_manifest),
        "complete_crop_manifest": len(results) == len(manifest["crops"]),
        "ground_truth_used_in_inference": False,
        "detector": detector.metadata,
        "thresholds": dict(thresholds),
        "device": str(device),
        "inference_and_output_s": time.perf_counter() - started,
        "abstention_policy": {
            "min_number_score": thresholds["min_number_score"],
            "rule": "weakest kept digit score; abstain on no digit, more than two digits, leading zero",
            "calibrated": False,
        },
        "predictions": results,
    }
    (out / "predictions.json").write_text(json.dumps(output, indent=2) + "\n")
    return output
