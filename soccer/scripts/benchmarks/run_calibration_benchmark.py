"""Freeze, infer, or evaluate the PnLCalib/YOLO calibration development experiment."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from soccerviz.candidates.calibration_model import (
    PnLCalibrationModel,
    YoloCalibrationModel,
    evaluate_calibrations,
    load_sample,
    make_protocol,
)
from soccerviz.candidates.pitch_keypoints import HeatmapCalibrationModel
from soccerviz.datasets.soccernet_adapter import sha256, write_json


def native_json(value):
    if isinstance(value, np.ndarray):
        return native_json(value.tolist())
    if isinstance(value, np.generic):
        return native_json(value.item())
    if isinstance(value, dict):
        return {str(key): native_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run", "evaluate"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--backend", choices=("pnlcalib", "yolo", "heatmap"))
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--weights-keypoints", type=Path)
    parser.add_argument("--weights-lines", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--path-prefix", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--ground-truth-index", type=Path)
    parser.add_argument(
        "--every", type=int, default=10, help="prepare: keep every Nth manifest frame"
    )
    args = parser.parse_args()
    if args.operation == "prepare":
        protocol = make_protocol(args.manifest, args.protocol, every=args.every)
        print(
            json.dumps(
                {
                    "protocol": str(args.protocol),
                    "frames": len(protocol["frames"]),
                    "sha256": sha256(args.protocol),
                }
            )
        )
        return
    if args.out is None or args.out.exists():
        raise ValueError("Provide a new --out path; outputs are never overwritten")
    if args.operation == "evaluate":
        if args.predictions is None:
            raise ValueError("Evaluation requires --predictions")
        result = evaluate_calibrations(
            args.manifest, args.protocol, args.predictions, args.ground_truth_index
        )
        write_json(args.out, result)
        print(json.dumps(result["overall"], indent=2))
        return
    protocol, frames = load_sample(args.manifest, args.protocol)
    if args.backend == "pnlcalib":
        if not all((args.upstream, args.weights_keypoints, args.weights_lines)):
            raise ValueError("PnLCalib requires upstream and both model weight paths")
        model = PnLCalibrationModel(
            args.upstream, args.weights_keypoints, args.weights_lines, protocol, args.device
        )
    elif args.backend == "yolo":
        if args.checkpoint is None:
            raise ValueError("YOLO requires --checkpoint")
        model = YoloCalibrationModel(args.checkpoint, protocol, args.device)
    elif args.backend == "heatmap":
        if args.checkpoint is None:
            raise ValueError("The heatmap backend requires --checkpoint")
        model = HeatmapCalibrationModel(args.checkpoint, protocol, args.device)
    else:
        raise ValueError("Inference requires --backend")
    prefixes = [tuple(value.split("=", 1)) for value in args.path_prefix]
    if any(len(pair) != 2 for pair in prefixes):
        raise ValueError("Path prefix must be OLD=NEW")
    predictions = {
        "schema": "calibration-predictions/v1",
        "manifest_sha256": sha256(args.manifest),
        "protocol_sha256": sha256(args.protocol),
        "backend": model.metadata,
        "source_labels_are_model_inputs": False,
        "frames": [],
    }
    start = time.perf_counter()
    for ordinal, frame in enumerate(frames):
        path = Path(frame["image_path"])
        for source, target in prefixes:
            if path.is_relative_to(source):
                path = Path(target) / path.relative_to(source)
                break
        # Integrity failures abort the run; model fitting failures stay as explicit rows.
        if sha256(path) != frame["sha256"]:
            raise ValueError(f"Source image SHA256 mismatch: {path}")
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (frame["height"], frame["width"]):
            raise ValueError(f"Image decode/dimensions mismatch: {path}")
        frame_start = time.perf_counter()
        try:
            result = model.predict(image)
        except Exception as exc:  # noqa: BLE001 - every failed model frame must remain explicit.
            result = {
                "accepted": False,
                "homography": None,
                "reason": f"model_error:{type(exc).__name__}",
                "error": str(exc)[:2000],
            }
        result.update(
            {
                "sequence": frame["sequence"],
                "frame_id": frame["frame_id"],
                "width": frame["width"],
                "height": frame["height"],
                "image_sha256": frame["sha256"],
                "first_frame_includes_cold_start": ordinal == 0,
                "wall_ms": (time.perf_counter() - frame_start) * 1000,
            }
        )
        predictions["frames"].append(native_json(result))
        print(
            json.dumps(
                {
                    "completed": ordinal + 1,
                    "total": len(frames),
                    "sequence": frame["sequence"],
                    "frame_id": frame["frame_id"],
                    "accepted": result["accepted"],
                    "reason": result["reason"],
                }
            ),
            flush=True,
        )
    predictions["total_wall_seconds"] = time.perf_counter() - start
    predictions["complete"] = True
    write_json(args.out, predictions)


if __name__ == "__main__":
    main()
