"""Top-down 2D body keypoints on frozen detector person boxes; swappable pinned backends.

Inference consumes only the frozen image manifest and a saved detector prediction
file. There are no keypoint labels in the benchmark, so `summarize` reports
sanity rates (keypoints inside their box, ankles below hips), never accuracy.
Backends are pinned by Hugging Face revision and weight file hash so a later
model (a mesh model, a larger ViTPose) is a new BACKENDS entry, not new plumbing.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from soccerviz.core.assets import sha256

BACKENDS = {
    "vitpose-plus-base": {
        "repo_id": "usyd-community/vitpose-plus-base",
        "revision": "92be54d7a29e42fad47b6e2ca01dd9e685a61e0d",
        "weights_file": "model.safetensors",
        "weights_sha256": "640225e4a9dd544239f1da0ae36af865a6e76e21e5f33a12656aa7b77fa5a5fa",
        "license": "apache-2.0",
        "family": "ViTPose++ ViT-B mixture-of-experts, COCO expert",
    },
    "vitpose-plus-huge": {
        "repo_id": "usyd-community/vitpose-plus-huge",
        "revision": "9f36d7aec1800d23e97f10c2e74393aee92aa53f",
        "weights_file": "model.safetensors",
        "weights_sha256": "0ecb49f1ab0b18cc2f18446b8100442cec88bd99dd53779ad5a7f8c71aa08506",
        "license": "apache-2.0",
        "family": "ViTPose++ ViT-H mixture-of-experts, COCO expert",
    },
}
COCO_EXPERT_INDEX = 0
COCO_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
# Offside law: any part of the head, body or feet counts; hands and arms do not.
ARM_KEYPOINTS = frozenset({"left_elbow", "right_elbow", "left_wrist", "right_wrist"})
FOOT_KEYPOINTS = ("left_ankle", "right_ankle")
HIP_KEYPOINTS = ("left_hip", "right_hip")
BOX_MARGIN_FRACTION = 0.1


def save_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def resolve_path(value, prefixes=()):
    path = Path(value)
    for old, new in prefixes:
        try:
            return Path(new) / path.relative_to(old)
        except ValueError:
            pass
    return path


def box_xyxy_to_xywh(box, width, height):
    """Clip a finite source-pixel box to the image and return COCO x, y, w, h."""
    points = np.asarray(box, dtype=float)
    if points.shape != (4,) or not np.isfinite(points).all():
        raise ValueError("Box requires four finite source-pixel coordinates")
    x0, y0 = max(0.0, points[0]), max(0.0, points[1])
    x1, y1 = min(float(width), points[2]), min(float(height), points[3])
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("Box is degenerate or wholly outside the image")
    return [x0, y0, x1 - x0, y1 - y0]


def frame_person_boxes(manifest, predictions):
    """Person boxes per frozen frame from detector output only; hashes must agree."""
    frames = {(r["sequence"], r["frame_id"]): r for r in manifest["frames"]}
    if len(frames) != len(manifest["frames"]):
        raise ValueError("Duplicate source frame")
    rows, seen = [], set()
    for row in predictions["frames"]:
        key = row["sequence"], row["frame_id"]
        if key in seen or key not in frames:
            raise ValueError("Duplicate prediction frame or frame outside frozen manifest")
        seen.add(key)
        source = frames[key]
        if row.get("image_sha256") != source["sha256"]:
            raise ValueError("Detector source image hash differs from frozen image")
        people = []
        for index, detection in enumerate(row["detections"]):
            if detection.get("label") != "person":
                continue
            if not math.isfinite(detection["score"]) or not 0 <= detection["score"] <= 1:
                raise ValueError("Invalid detector confidence")
            people.append(
                {
                    "detection_index": index,
                    "detector_score": detection["score"],
                    "predicted_role": detection.get("role"),
                    "bbox_xyxy": [float(v) for v in detection["bbox_xyxy"]],
                    "bbox_xywh": box_xyxy_to_xywh(
                        detection["bbox_xyxy"], source["width"], source["height"]
                    ),
                }
            )
        rows.append({"frame": source, "people": people})
    if seen != set(frames):
        raise ValueError("Detector output does not cover the frozen manifest")
    return rows


def validate_keypoints(keypoints, scores):
    points = np.asarray(keypoints, dtype=float)
    confidences = np.asarray(scores, dtype=float)
    if points.shape != (len(COCO_KEYPOINTS), 2) or not np.isfinite(points).all():
        raise ValueError("Expected 17 finite COCO keypoints")
    if confidences.shape != (len(COCO_KEYPOINTS),) or not np.isfinite(confidences).all():
        raise ValueError("Expected 17 finite keypoint scores")
    if (confidences < 0).any() or (confidences > 1).any():
        raise ValueError("Keypoint scores must lie in [0,1]")
    return points, confidences


def body_extremes(keypoints, scores, min_score):
    """Lowest foot and the set of offside-eligible points; arms are excluded by law.

    Returns None when no foot keypoint reaches min_score. Which point is 'furthest
    forward' depends on the attacking direction and the pitch homography, so that
    decision is left to the calibration layer; this only names the candidates.
    """
    points, confidences = validate_keypoints(keypoints, scores)
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must lie in [0,1]")
    eligible = [
        {"name": name, "xy": points[i].tolist(), "score": float(confidences[i])}
        for i, name in enumerate(COCO_KEYPOINTS)
        if name not in ARM_KEYPOINTS and confidences[i] >= min_score
    ]
    feet = [e for e in eligible if e["name"] in FOOT_KEYPOINTS]
    if not feet:
        return None
    return {
        "lowest_foot": max(feet, key=lambda e: e["xy"][1]),
        "eligible_points": eligible,
        "rule": "head, body and feet only; wrists and elbows excluded; score floor applied",
        "min_score": min_score,
    }


def inside_box(xy, box_xyxy, margin_fraction=BOX_MARGIN_FRACTION):
    x0, y0, x1, y1 = box_xyxy
    mx, my = (x1 - x0) * margin_fraction, (y1 - y0) * margin_fraction
    return x0 - mx <= xy[0] <= x1 + mx and y0 - my <= xy[1] <= y1 + my


class ViTPoseBackend:
    """Pinned Hugging Face ViTPose++ checkpoint loaded offline from a local snapshot."""

    def __init__(self, name, cache_dir, *, device="cuda:0"):
        if name not in BACKENDS:
            raise ValueError(f"Unknown pose backend {name!r}; known: {sorted(BACKENDS)}")
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        spec = BACKENDS[name]
        snapshot = Path(
            snapshot_download(
                spec["repo_id"],
                revision=spec["revision"],
                cache_dir=str(cache_dir),
                local_files_only=True,
            )
        )
        weights = snapshot / spec["weights_file"]
        observed = sha256(weights)
        if observed != spec["weights_sha256"]:
            raise ValueError(
                f"{name} weights differ from the pinned revision: expected "
                f"{spec['weights_sha256']}, observed {observed}, path {weights}"
            )
        self.torch, self.device, self.name = torch, device, name
        self.processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        self.model = VitPoseForPoseEstimation.from_pretrained(snapshot, local_files_only=True)
        self.model.to(device).eval()
        self.last_timing_ms = {}
        self.metadata = {
            "name": name,
            **spec,
            "snapshot_path": str(snapshot),
            "dataset_index": COCO_EXPERT_INDEX,
            "keypoints": list(COCO_KEYPOINTS),
            "input_hw": [self.processor.size["height"], self.processor.size["width"]],
            "precision": "float32",
            "device": str(device),
            "score_kind": "heatmap_peak_clipped_to_unit_interval_uncalibrated",
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("torch", "transformers", "huggingface_hub", "numpy")
            },
        }

    def _sync(self):
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def predict(self, image_bgr, boxes_xywh):
        """Keypoints in source pixels for every box, in box order."""
        if not boxes_xywh:
            return []
        started = time.perf_counter()
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        inputs = self.processor(rgb, boxes=[boxes_xywh], return_tensors="pt").to(self.device)
        dataset_index = self.torch.full(
            (inputs["pixel_values"].shape[0],), COCO_EXPERT_INDEX, device=self.device
        )
        self._sync()
        prepared = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(**inputs, dataset_index=dataset_index)
            self._sync()
        forwarded = time.perf_counter()
        results = self.processor.post_process_pose_estimation(outputs, boxes=[boxes_xywh])[0]
        if len(results) != len(boxes_xywh):
            raise ValueError("Pose output count differs from box count")
        people = []
        for result in results:
            # Heatmap peaks are unbounded regression values; the raw peak is kept and
            # the bounded score is an explicit clip, not a calibrated probability.
            peaks = result["scores"].float().cpu().numpy()
            points, confidences = validate_keypoints(
                result["keypoints"].float().cpu().numpy(), np.clip(peaks, 0.0, 1.0)
            )
            people.append(
                {
                    "keypoints_xy": points.tolist(),
                    "scores": confidences.tolist(),
                    "heatmap_peaks": peaks.astype(float).tolist(),
                }
            )
        finished = time.perf_counter()
        self.last_timing_ms = {
            "preprocess_and_transfer": (prepared - started) * 1000,
            "forward": (forwarded - prepared) * 1000,
            "postprocess": (finished - forwarded) * 1000,
            "total": (finished - started) * 1000,
        }
        return people


def _load_partial(path):
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(r["sequence"], r["frame_id"]): r for r in rows}


def _read_verified(frame, prefixes):
    import cv2

    path = resolve_path(frame["image_path"], prefixes)
    if sha256(path) != frame["sha256"]:
        raise ValueError(f"Image hash changed: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (frame["height"], frame["width"]):
        raise ValueError(f"Unreadable image or incorrect dimensions: {path}")
    return image


def infer(
    manifest_path,
    predictions_path,
    out,
    *,
    backend_name="vitpose-plus-base",
    cache_dir="artifacts/models/hf",
    device="cuda:0",
    limit=None,
    prefixes=(),
    backend_factory=ViTPoseBackend,
):
    """Run the backend frame by frame; each frame is appended to frames.jsonl as it lands.

    A rerun with the same `out` resumes from frames.jsonl and skips finished frames.
    predictions.json is written only once every requested frame is present.
    """
    out = Path(out)
    if (out / "predictions.json").exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {out / 'predictions.json'}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    manifest = json.loads(Path(manifest_path).read_text())
    predictions = json.loads(Path(predictions_path).read_text())
    if manifest.get("schema") != "vision-benchmark/v1":
        raise ValueError("Expected a vision-benchmark/v1 manifest")
    if predictions.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("Detector predictions are not bound to this frozen manifest")
    rows = frame_person_boxes(manifest, predictions)
    selected = rows[:limit] if limit else rows
    out.mkdir(parents=True, exist_ok=True)
    partial_path = out / "frames.jsonl"
    done = _load_partial(partial_path)
    backend = backend_factory(backend_name, cache_dir, device=device)
    started = time.perf_counter()
    timings = []
    with partial_path.open("a") as partial:
        for number, row in enumerate(selected, start=1):
            frame = row["frame"]
            key = frame["sequence"], frame["frame_id"]
            if key in done:
                continue
            image = _read_verified(frame, prefixes)
            people = backend.predict(image, [p["bbox_xywh"] for p in row["people"]])
            record = _frame_record(frame, row["people"], people)
            partial.write(json.dumps(record, allow_nan=False) + "\n")
            partial.flush()
            done[key] = record
            timings.append(backend.last_timing_ms.get("total", 0.0))
            if number == 1 or number % 25 == 0 or number == len(selected):
                print(f"pose: {number}/{len(selected)} frames", flush=True)
    frames = [done[(r["frame"]["sequence"], r["frame"]["frame_id"])] for r in selected]
    output = {
        "schema": "pose-predictions/v1",
        "backend": backend.metadata,
        "manifest_sha256": sha256(manifest_path),
        "detector_predictions_sha256": sha256(predictions_path),
        "detector_backend": predictions["backend"],
        "box_source": "predicted_detector_person_boxes",
        "ground_truth_used_in_inference": False,
        "complete_manifest": len(frames) == len(rows),
        "adapter_sha256": sha256(__file__),
        "output_coordinates": "original-image pixels",
        "runtime": {
            "session_s": time.perf_counter() - started,
            "frames_inferred_this_session": len(timings),
            "median_total_ms_per_frame": float(np.median(timings)) if timings else None,
            "resumed_frames": len(frames) - len(timings),
        },
        "frames": frames,
    }
    save_json(out / "predictions.json", output)
    return output


def _frame_record(frame, people, poses):
    if len(poses) != len(people):
        raise ValueError("Backend returned a pose count that differs from the box count")
    return {
        "sequence": frame["sequence"],
        "frame_id": frame["frame_id"],
        "width": frame["width"],
        "height": frame["height"],
        "timestamp_s": frame["timestamp_s"],
        "image_sha256": frame["sha256"],
        "people": [
            {
                "detection_index": person["detection_index"],
                "detector_score": person["detector_score"],
                "predicted_role": person["predicted_role"],
                "bbox_xyxy": person["bbox_xyxy"],
                **pose,
            }
            for person, pose in zip(people, poses, strict=True)
        ],
    }


def summarize(predictions_path, out, *, min_score=0.3):
    """Sanity rates on saved predictions. No keypoint labels exist: this is not accuracy."""
    data = json.loads(Path(predictions_path).read_text())
    if data.get("schema") != "pose-predictions/v1":
        raise ValueError("Expected pose-predictions/v1")
    names = list(COCO_KEYPOINTS)
    foot = [names.index(n) for n in FOOT_KEYPOINTS]
    hip = [names.index(n) for n in HIP_KEYPOINTS]
    counts = Counter()
    score_sums = np.zeros(len(names))
    confident = np.zeros(len(names))
    for frame in data["frames"]:
        for person in frame["people"]:
            points, scores = validate_keypoints(person["keypoints_xy"], person["scores"])
            counts["people"] += 1
            counts["by_role", person.get("predicted_role")] += 1
            score_sums += scores
            confident += scores >= min_score
            box = person["bbox_xyxy"]
            counts["keypoints_inside_box"] += sum(inside_box(p, box) for p in points)
            counts["keypoints"] += len(points)
            if all(scores[i] >= min_score for i in foot + hip) and max(
                points[i][1] for i in foot
            ) > max(points[i][1] for i in hip):
                counts["ankles_below_hips"] += 1
            if all(scores[i] >= min_score for i in foot + hip):
                counts["feet_and_hips_confident"] += 1
            if body_extremes(points, scores, min_score) is not None:
                counts["with_confident_foot"] += 1
    people = counts["people"]
    report = {
        "schema": "pose-summary/v1",
        "kind": "sanity_rates_not_accuracy",
        "predictions_sha256": sha256(predictions_path),
        "backend": data["backend"]["name"],
        "min_score": min_score,
        "frames": len(data["frames"]),
        "people": people,
        "people_by_role": {
            str(k[1]): v for k, v in counts.items() if isinstance(k, tuple) and k[0] == "by_role"
        },
        "keypoints_inside_box_rate": counts["keypoints_inside_box"] / counts["keypoints"]
        if counts["keypoints"]
        else None,
        "box_margin_fraction": BOX_MARGIN_FRACTION,
        "people_with_confident_foot_rate": counts["with_confident_foot"] / people
        if people
        else None,
        "ankles_below_hips_rate_when_confident": counts["ankles_below_hips"]
        / counts["feet_and_hips_confident"]
        if counts["feet_and_hips_confident"]
        else None,
        "mean_score_by_keypoint": {n: float(score_sums[i] / people) for i, n in enumerate(names)}
        if people
        else {},
        "confident_rate_by_keypoint": {n: float(confident[i] / people) for i, n in enumerate(names)}
        if people
        else {},
        "limitations": [
            "No keypoint ground truth in the frozen benchmark; rates check plausibility only",
            "Boxes are detector predictions, so pose errors include detector box errors",
            "Heatmap peak scores are uncalibrated; the score floor is fixed, not tuned",
            "No temporal smoothing, no identity, no pitch projection in this worker",
        ],
    }
    save_json(out, report)
    return report
