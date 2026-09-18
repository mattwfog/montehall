"""Pinned PnLCalib and existing YOLO pitch-model adapters and isolated scoring.

Inference never receives annotations. Evaluation uses oracle image footpoints to
measure calibration only; these are not detector-to-game-state accuracy scores.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from soccerviz.core import geometry
from soccerviz.core.geometry import calibrate, pitch_landmarks, project
from soccerviz.datasets.soccernet_adapter import sha256, write_json

PNL_COMMIT = "8c87391d6f4ea40c5e4d65e61529916c7a49ce62"
# SHA256 of the published PnLCalib weight files.
PNL_POINTS_SHA256 = "7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113"
PNL_LINES_SHA256 = "d72f4ed71734a2e3df9fa084f666e9b8adaef21bf69bac8952d6d3f970ff7455"
PNL_WEIGHTS = {"keypoints": PNL_POINTS_SHA256, "lines": PNL_LINES_SHA256}
PNL_SOURCE_HASHES = {
    "inference.py": "8c7211b91378fe820d7e1895e1290a138bb646efdd48d031aca6a55e789ce02a",
    "model/cls_hrnet.py": "c48c8fcebe3cea0490f133181984a28a0ebac70f65c292c0e19428afa4344af8",
    "model/cls_hrnet_l.py": "8cd2a2ba50d29037cbfcfddc531bfc430458ee18c6edb13568f92c9ba67f8a9d",
    "utils/utils_calib.py": "5443a84f89b095d6d11779b8c75862648be3bf80ba745106510c3cc800120222",
    "utils/utils_optimize.py": "11680c014355226937fab8fa46b72980b32891e4892b0d25493e999db834064b",
    "utils/utils_heatmap.py": "b2ae3b0790c47e721414b020428f96182f1056cf6e2a6460241d5fdfe00b265a",
    "config/hrnetv2_w48.yaml": "1197a64ffaadc93b0d4be22b68f19bbefbeda645bcc32ee2a3a19b905d81900a",
    "config/hrnetv2_w48_l.yaml": "5fa19ede937ef5ff8dbf844a6a76643da15c5cc16fe5b71fa81ee69b3b88dbd8",
}


def key(frame):
    return frame["sequence"], int(frame["frame_id"])


def checked_homography(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Homography must be a finite 3x3 matrix")
    if np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Singular homography")
    norm = matrix[2, 2] if abs(matrix[2, 2]) > 1e-10 else np.linalg.norm(matrix)
    return matrix / norm


def centered_homography(origin_homography, length=105.0, width=68.0):
    """Translate existing top-left-origin pitch meters into SoccerNet coordinates."""
    shift = np.array([[1, 0, -length / 2], [0, 1, -width / 2], [0, 0, 1]])
    return checked_homography(shift @ checked_homography(origin_homography))


def ground_homography_from_projection(projection):
    """Invert upstream centered-world camera P on z=0; no calibration refitting."""
    projection = np.asarray(projection, dtype=float)
    if projection.shape != (3, 4) or not np.isfinite(projection).all():
        raise ValueError("Camera projection must be a finite 3x4 matrix")
    plane_to_image = checked_homography(projection[:, [0, 1, 3]])
    return checked_homography(np.linalg.inv(plane_to_image))


def package_versions():
    versions = {}
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "opencv-python",
        "PyYAML",
        "shapely",
        "matplotlib",
        "ultralytics",
        "Pillow",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def make_protocol(manifest_path, output_path, every=10):
    """Freeze the calibration sample: every `every`-th manifest frame per sequence, in
    manifest order, before any model output exists. The 2026-09-07 development sample
    took every tenth of 150 consecutive frames; a corpus that is already sparse (the
    game-disjoint evaluation, one frame a second) passes `every=1`."""
    if Path(output_path).exists():
        raise FileExistsError("Calibration protocol already frozen")
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("purpose") != "development":
        raise ValueError("Calibration experiment requires a frozen development corpus")
    if every < 1:
        raise ValueError("every must be a positive stride")
    ordinal = Counter()
    sample = []
    for frame in manifest["frames"]:
        position = ordinal[frame["sequence"]]
        ordinal[frame["sequence"]] += 1
        if position % every == 0:
            sample.append({"sequence": frame["sequence"], "frame_id": frame["frame_id"]})
    protocol = {
        "schema": "calibration-benchmark-protocol/v1",
        "purpose": "development",
        "manifest_sha256": sha256(manifest_path),
        "frames": sample,
        "sample_policy": f"Every {every}th frozen manifest frame per sequence, from the first, in manifest order. Selected before calibration model outputs.",
        "pitch": {
            "length_m": 105.0,
            "width_m": 68.0,
            "origin": "center",
            "axes": "SoccerNet/PnL x=-52.5..52.5,y=-34..34; no reflection or source-driven sign correction.",
        },
        "pnl": {
            "upstream_commit": PNL_COMMIT,
            "keypoint_threshold": 0.3434,
            "line_threshold": 0.7867,
            "resize_hw": [540, 960],
            "refine_lines": True,
            "refine": False,
            "acceptance": "Official heuristic_voting returns a camera; its ground-plane homography is finite and invertible. No ground-truth error rejection or additional reprojection cutoff.",
        },
        "yolo": {
            "imgsz": 640,
            "detection_confidence": 0.25,
            "landmark_confidence": 0.5,
            "acceptance": "Unmodified geometry.calibrate: >=6 landmarks/inliers, inlier fraction>=0.55, median leave-one-out<3m; RANSAC threshold1.5m.",
        },
        "evaluation": {
            "targets": "Official object category1/2/3 image bbox bottom-center and matching bbox_pitch bottom-middle; balls excluded because their center may be airborne.",
            "successful_point_threshold_m": 2.0,
            "point_policy": "All finite independent target pairs. No clipping to pitch or discarding large errors. Nonfinite projections and absent/rejected frames count as unavailable.",
            "metrics": "Frame calibration coverage, footpoint projection coverage, median/p95 error over available points, and points<=2m divided by ALL valid targets. Full failure reasons and per-sequence results.",
            "oracle_input_warning": "Oracle image footpoints isolate calibration accuracy; not detector-to-state or end-to-end tracking accuracy.",
            "source_labels_are_fit_inputs": False,
        },
        "frozen_before_model_outputs": True,
    }
    write_json(output_path, protocol)
    return protocol


def load_sample(manifest_path, protocol_path):
    manifest = json.loads(Path(manifest_path).read_text())
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol.get("schema") != "calibration-benchmark-protocol/v1":
        raise ValueError("Unsupported calibration protocol")
    if sha256(manifest_path) != protocol["manifest_sha256"]:
        raise ValueError("Calibration protocol manifest SHA256 mismatch")
    frames = {key(frame): frame for frame in manifest["frames"]}
    selected = [key(frame) for frame in protocol["frames"]]
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(k not in frames for k in selected)
    ):
        raise ValueError("Invalid frozen calibration frame selection")
    return protocol, [frames[k] for k in selected]


class PnLCalibrationModel:
    def __init__(self, upstream, weights_keypoints, weights_lines, protocol, device="cuda:0"):
        import torch
        import yaml

        upstream = Path(upstream).resolve()
        for relative, expected in PNL_SOURCE_HASHES.items():
            if sha256(upstream / relative) != expected:
                raise ValueError(f"Pinned PnLCalib source mismatch: {relative}")
        checkpoints = {}
        for label, path in (("keypoints", weights_keypoints), ("lines", weights_lines)):
            digest = sha256(path)
            if digest != PNL_WEIGHTS[label]:
                raise ValueError(f"Pinned PnLCalib {label} weight SHA256 mismatch")
            checkpoints[label] = {"path": str(Path(path).resolve()), "sha256": digest}
        sys.path.insert(0, str(upstream))
        spec = importlib.util.spec_from_file_location(
            "_soccerviz_pnl_inference", upstream / "inference.py"
        )
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        official.device = device
        official.transform2 = official.T.Resize((540, 960))
        self.official, self.device, self.protocol = official, device, protocol["pnl"]
        cfg = yaml.safe_load((upstream / "config/hrnetv2_w48.yaml").read_text())
        cfg_lines = yaml.safe_load((upstream / "config/hrnetv2_w48_l.yaml").read_text())
        self.model = official.get_cls_net(cfg)
        self.lines = official.get_cls_net_l(cfg_lines)
        self.model.load_state_dict(
            torch.load(weights_keypoints, map_location="cpu", weights_only=True)
        )
        self.lines.load_state_dict(torch.load(weights_lines, map_location="cpu", weights_only=True))
        self.model.to(device).eval()
        self.lines.to(device).eval()
        self.metadata = {
            "name": "PnLCalib-HRNet-W48+lines",
            "upstream_commit": PNL_COMMIT,
            "upstream_source_sha256": PNL_SOURCE_HASHES,
            "checkpoints": checkpoints,
            "protocol": self.protocol,
            "device": device,
            "versions": package_versions(),
            "coordinates": "Original image pixels→centered105×68 pitch meters",
            "temporal_state": "Fresh official FramebyFrameCalib per frame; no smoothing or earlier-frame camera reuse.",
        }

    def predict(self, image):
        import torch

        height, width = image.shape[:2]
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        camera = self.official.FramebyFrameCalib(iwidth=width, iheight=height, denormalize=True)
        result = self.official.inference(
            camera,
            image,
            self.model,
            self.lines,
            self.protocol["keypoint_threshold"],
            self.protocol["line_threshold"],
            self.protocol["refine_lines"],
        )
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        diagnostic = {
            "input_keypoints": len(camera.keypoints_dict),
            "input_lines": len(camera.lines_dict),
            "inference_and_geometry_ms": elapsed,
        }
        if result is None:
            return {
                "accepted": False,
                "reason": "upstream_no_camera",
                "homography": None,
                "diagnostics": diagnostic,
            }
        # Both projection construction and calibration come from upstream. Only
        # select z=0 and invert it to meet the shared source-pixel contract.
        projection = self.official.projection_from_cam_params(result)
        homography = ground_homography_from_projection(projection)
        diagnostic.update(
            {key: result.get(key) for key in ("mode", "use_ransac", "rep_err", "calib_plane")}
        )
        return {
            "accepted": True,
            "reason": "accepted",
            "homography": homography.tolist(),
            "diagnostics": diagnostic,
            "camera_parameters": result["cam_params"],
        }


class YoloCalibrationModel:
    def __init__(self, checkpoint, protocol, device="cuda:0"):
        from ultralytics import YOLO

        self.model = YOLO(str(checkpoint))
        self.device, self.protocol = device, protocol["yolo"]
        self.metadata = {
            "name": "existing-soccer-YOLO-pitch+geometry.calibrate",
            "device": device,
            "checkpoints": {
                "pitch": {"path": str(Path(checkpoint).resolve()), "sha256": sha256(checkpoint)}
            },
            "protocol": self.protocol,
            "versions": package_versions(),
            "geometry_source_sha256": sha256(Path(geometry.__file__)),
            "coordinates": "Existing0..105/0..68 projection translated by[-52.5,-34]",
        }

    def predict(self, image):
        import torch

        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        keypoints = self.model.predict(
            image,
            imgsz=self.protocol["imgsz"],
            conf=self.protocol["detection_confidence"],
            device=self.device,
            verbose=False,
        )[0].keypoints
        if keypoints is not None and len(keypoints.xy):
            points = keypoints.xy[0].cpu().numpy()
            confidence = (
                keypoints.conf[0].cpu().numpy()
                if keypoints.conf is not None
                else np.zeros(len(points))
            )
        else:
            points, confidence = np.zeros((32, 2)), np.zeros(32)
        if len(points) != 32:
            raise ValueError("Unexpected soccer pitch keypoint count")
        valid = (confidence >= self.protocol["landmark_confidence"]) & (points > 0).all(axis=1)
        fit = calibrate(points[valid], pitch_landmarks()[valid])
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        return {
            "accepted": fit.accepted,
            "reason": fit.reason,
            "homography": centered_homography(fit.matrix).tolist() if fit.accepted else None,
            "diagnostics": {
                **fit.diagnostics(),
                "inference_and_geometry_ms": (time.perf_counter() - start) * 1000,
            },
        }


def _error_summary(errors):
    values = np.asarray(errors, dtype=float)
    return {
        "count": len(errors),
        "median_m": float(np.median(values)) if len(values) else None,
        "p95_m": float(np.quantile(values, 0.95)) if len(values) else None,
        "mean_m": float(np.mean(values)) if len(values) else None,
    }


def evaluate_calibrations(manifest_path, protocol_path, predictions_path, ground_truth_index=None):
    protocol, selected = load_sample(manifest_path, protocol_path)
    predictions = json.loads(Path(predictions_path).read_text())
    if predictions.get("schema") != "calibration-predictions/v1":
        raise ValueError("Unsupported calibration prediction schema")
    if predictions.get("manifest_sha256") != sha256(manifest_path) or predictions.get(
        "protocol_sha256"
    ) != sha256(protocol_path):
        raise ValueError("Calibration predictions provenance mismatch")
    truth_index = Path(ground_truth_index or Path(manifest_path).parent / "ground-truth.json")
    index = json.loads(truth_index.read_text())
    if index["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Ground-truth index manifest mismatch")
    # Frames whose source annotation is known wrong are still evaluated and reported,
    # but never enter `overall` or `by_sequence`; the reason travels with the record.
    excluded = {
        (entry["sequence"], int(entry["frame_id"])): entry["reason"]
        for entry in index.get("excluded_frames", [])
    }
    sources = {}
    for source in index["sources"]:
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("Source ground-truth annotation SHA256 mismatch")
        sources[source["sequence"]] = json.loads(Path(source["path"]).read_text())
    target_points = {}
    selected_keys = {key(frame) for frame in selected}
    for sequence, source in sources.items():
        image_frames = {
            image["image_id"]: int(Path(image["file_name"]).stem) for image in source["images"]
        }
        for annotation in source["annotations"]:
            frame_key = (sequence, image_frames[annotation["image_id"]])
            if frame_key not in selected_keys or annotation.get("category_id") not in (1, 2, 3):
                continue
            box, pitch = annotation.get("bbox_image"), annotation.get("bbox_pitch")
            if not box or not pitch:
                target_points.setdefault(frame_key, {"pairs": [], "invalid_targets": 0})[
                    "invalid_targets"
                ] += 1
                continue
            image_point = [box["x"] + box["w"] / 2, box["y"] + box["h"]]
            world_point = [pitch.get("x_bottom_middle"), pitch.get("y_bottom_middle")]
            target = target_points.setdefault(frame_key, {"pairs": [], "invalid_targets": 0})
            if not all(
                value is not None and math.isfinite(value) for value in image_point + world_point
            ):
                target["invalid_targets"] += 1
            else:
                target["pairs"].append((image_point, world_point))
    by_key = {}
    for frame in predictions["frames"]:
        frame_key = key(frame)
        if frame_key not in selected_keys or frame_key in by_key:
            raise ValueError("Unexpected or duplicate calibration prediction frame")
        by_key[frame_key] = frame
    records = []
    all_errors = {}
    threshold = protocol["evaluation"]["successful_point_threshold_m"]
    for frame in selected:
        frame_key = key(frame)
        if sha256(frame["image_path"]) != frame["sha256"]:
            raise ValueError("Evaluation image SHA256 mismatch")
        prediction = by_key.get(
            frame_key, {"accepted": False, "reason": "missing_prediction_frame"}
        )
        target = target_points.get(frame_key, {"pairs": [], "invalid_targets": 0})
        pairs, errors = target["pairs"], []
        accepted, reason = prediction.get("accepted", False), prediction.get("reason", "unknown")
        if accepted:
            matrix = checked_homography(prediction["homography"])
            if pairs:
                projected = project([pair[0] for pair in pairs], matrix)
                distances = np.linalg.norm(
                    projected - np.asarray([pair[1] for pair in pairs]), axis=1
                )
                errors = distances[np.isfinite(distances)].tolist()
        all_errors[frame_key] = errors
        records.append(
            {
                "sequence": frame_key[0],
                "frame_id": frame_key[1],
                "accepted": bool(accepted),
                "reason": reason,
                "target_points": len(pairs),
                "invalid_source_targets": target["invalid_targets"],
                "projected_points": len(errors),
                "unavailable_points": len(pairs) - len(errors),
                "points_within_threshold": sum(error <= threshold for error in errors),
                "error": _error_summary(errors),
            }
        )

    def summary(rows):
        count = len(rows)
        target_count = sum(row["target_points"] for row in rows)
        projected_count = sum(row["projected_points"] for row in rows)
        successes = sum(row["points_within_threshold"] for row in rows)
        available = sum(row["accepted"] for row in rows)
        return {
            "frames": count,
            "accepted_frames": available,
            "frame_coverage": available / count if count else None,
            "frame_rejection_fraction": 1 - available / count if count else None,
            "failure_reasons": dict(Counter(row["reason"] for row in rows if not row["accepted"])),
            "valid_target_points": target_count,
            "projected_points": projected_count,
            "projection_coverage": projected_count / target_count if target_count else None,
            "unavailable_points": target_count - projected_count,
            "invalid_source_targets": sum(row["invalid_source_targets"] for row in rows),
            "points_within_2m": successes,
            "all_target_fraction_within_2m": successes / target_count if target_count else None,
            "accepted_point_error": _error_summary(
                [error for row in rows for error in all_errors[key(row)]]
            ),
        }

    scored = [row for row in records if key(row) not in excluded]
    excluded_records = [
        {**row, "exclusion_reason": excluded[key(row)]} for row in records if key(row) in excluded
    ]
    return {
        "schema": "calibration-benchmark-results/v1",
        "purpose": "development",
        "manifest_sha256": sha256(manifest_path),
        "protocol_sha256": sha256(protocol_path),
        "predictions_sha256": sha256(predictions_path),
        "ground_truth_index_sha256": sha256(truth_index),
        "backend": predictions["backend"],
        "overall": summary(scored),
        "by_sequence": {
            sequence: summary([row for row in scored if row["sequence"] == sequence])
            for sequence in sorted({row["sequence"] for row in scored})
        },
        "frames": scored,
        "excluded_frames": excluded_records,
        "missing_prediction_frames": len(selected_keys - set(by_key)),
        "oracle_image_footpoints": True,
        "ground_truth_used_for_fitting": False,
        "limitations": [
            "Calibration-only development diagnostic on45 sampled frames across3games.",
            "Oracle image footpoints; not end-to-end detection or tracking performance.",
            "Error summaries conditional on accepted finite projections; coverage and all-target success rate reported alongside.",
            "PnL and YOLO retain their separate declared acceptance policies.",
            "Assumes centered105x68m pitch; no truth-assisted reflection correction.",
        ],
    }
