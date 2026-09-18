"""Run a fixed image manifest through one detector, preserving its exact SHA.

Example (inside isolated worker):
  python scripts/benchmarks/run_detector_benchmark.py --manifest artifacts/vision-benchmark/manifest.json \
    --backend rf-detr-medium --class-mode coco --out artifacts/rf-medium-predictions.json \
    --path-prefix "$PWD"=/work

This writes predictions only. Ground truth is deliberately not an input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from soccerviz.candidates.detector_backends import RFDETRDetector, UltralyticsDetector, gpu_memory
from soccerviz.core.assets import sha256


def resolve_image_path(value, manifest_path, prefixes):
    path = Path(value)
    for source, target in prefixes:
        try:
            return target / path.relative_to(source)
        except ValueError:
            continue
    return path if path.is_absolute() else manifest_path.parent / path


def load_image(frame, manifest_path, prefixes):
    path = resolve_image_path(frame["image_path"], manifest_path, prefixes)
    if sha256(path) != frame["sha256"]:
        raise ValueError(f"Image SHA mismatch for {frame['sequence']}/{frame['frame_id']}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (frame["height"], frame["width"]):
        raise ValueError(f"Image could not be read or dimensions changed: {path}")
    return image


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--backend", required=True, choices=["ultralytics", "rf-detr-small", "rf-detr-medium"]
    )
    parser.add_argument("--class-mode", required=True, choices=["coco", "soccer"])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Local detector weights; required for Ultralytics and soccer RF-DETR",
    )
    parser.add_argument("--ball-checkpoint", type=Path)
    parser.add_argument("--tiled-ball", action="store_true")
    parser.add_argument(
        "--ball-selection",
        choices=["all", "top1"],
        default="all",
        help="Dedicated ball control: all NMS candidates or existing top1 policy",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resolution", type=int, help="Default1280 for YOLO;512/576 for RF Small/Medium"
    )
    parser.add_argument("--threshold", default=0.25, type=float)
    parser.add_argument("--warmup", default=3, type=int)
    parser.add_argument(
        "--limit", type=int, help="Smoke-test prefix only; output explicitly marked incomplete"
    )
    parser.add_argument(
        "--path-prefix",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Remap image path prefixes without modifying manifest bytes",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("threshold must be finite and in [0,1]")
    if args.warmup < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("warmup must be nonnegative and limit positive")
    if args.resolution is not None and args.resolution <= 0:
        parser.error("resolution must be positive")
    if args.backend == "ultralytics" and args.checkpoint is None:
        parser.error("Ultralytics requires an existing local --checkpoint")
    if args.backend != "ultralytics" and (args.ball_checkpoint or args.tiled_ball):
        parser.error("Dedicated ball checkpoint flags are for the Ultralytics control")
    args.prefixes = []
    for value in args.path_prefix:
        source, separator, target = value.partition("=")
        if not separator or not source or not target:
            parser.error("path-prefix must be OLD=NEW")
        args.prefixes.append((Path(source), Path(target)))
    return args


def run(args):
    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite completed predictions: {args.out}")
    manifest_hash = sha256(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "vision-benchmark/v1":
        raise ValueError("Expected a vision-benchmark/v1 manifest")
    source_frames = manifest["frames"]
    if not source_frames:
        raise ValueError("Manifest has no frames")
    identities = [(f["sequence"], f["frame_id"]) for f in source_frames]
    if len(identities) != len(set(identities)):
        raise ValueError("Manifest contains duplicate sequence/frame identifiers")
    selected = source_frames[: args.limit] if args.limit else source_frames
    for frame in selected:
        for key in ("sequence", "frame_id", "image_path", "sha256", "width", "height"):
            if key not in frame:
                raise ValueError(f"Manifest frame lacks {key}")
    init_start = time.perf_counter()
    if args.backend == "ultralytics":
        backend = UltralyticsDetector(
            args.checkpoint,
            mode=args.class_mode,
            device=args.device,
            resolution=args.resolution or 1280,
            threshold=args.threshold,
            ball_checkpoint=args.ball_checkpoint,
            tiled_ball=args.tiled_ball,
            ball_selection=args.ball_selection,
        )
    else:
        backend = RFDETRDetector(
            size=args.backend.removeprefix("rf-detr-"),
            checkpoint=args.checkpoint,
            mode=args.class_mode,
            device=args.device,
            resolution=args.resolution,
            threshold=args.threshold,
        )
    initialization_s = time.perf_counter() - init_start
    warm_start = time.perf_counter()
    warm_image = load_image(selected[0], args.manifest, args.prefixes)
    for _ in range(args.warmup):
        backend.predict(warm_image)
    warmup_s = time.perf_counter() - warm_start
    frames = []
    measured_start = time.perf_counter()
    for index, source in enumerate(selected):
        frame_start = time.perf_counter()
        image = load_image(source, args.manifest, args.prefixes)
        loaded = time.perf_counter()
        detections, transforms = backend.predict(image)
        finished = time.perf_counter()
        frames.append(
            {
                "sequence": source["sequence"],
                "frame_id": source["frame_id"],
                "width": source["width"],
                "height": source["height"],
                "timestamp_s": source.get("timestamp_s"),
                "image_sha256": source["sha256"],
                "detections": detections,
                "native_background_rows_excluded": getattr(backend, "last_background_count", 0),
                "transforms": transforms,
                "timing_ms": {
                    "image_read_and_sha": (loaded - frame_start) * 1000,
                    "predict": (finished - loaded) * 1000,
                    "model_api": backend.last_model_ms,
                    "total": (finished - frame_start) * 1000,
                },
            }
        )
        if (index + 1) % 25 == 0 or index == 0 or index + 1 == len(selected):
            print(f"{args.backend}: {index + 1}/{len(selected)} frames", flush=True)
    measured_s = time.perf_counter() - measured_start
    predict_ms = [f["timing_ms"]["predict"] for f in frames]
    output = {
        "schema": "vision-predictions/v1",
        "manifest_sha256": manifest_hash,
        "backend": backend.metadata,
        "checkpoint_hashes": {
            key: value["sha256"] for key, value in backend.metadata["checkpoints"].items()
        },
        "complete_manifest": len(frames) == len(source_frames),
        "purpose": "development detector comparison; no tracker or labels consumed",
        "frames": frames,
        "runtime": {
            "initialization_s": initialization_s,
            "warmup_s": warmup_s,
            "warmup_iterations": args.warmup,
            "measured_s": measured_s,
            "measured_frames_per_s": len(frames) / measured_s,
            "predict_median_ms": float(np.median(predict_ms)),
            "predict_p95_ms": float(np.percentile(predict_ms, 95)),
            "elapsed_before_output_write_s": time.perf_counter() - started,
            "gpu_memory": gpu_memory(args.device),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "timing_note": "CUDA synchronized; model_api includes library preprocess, "
            "forward and postprocess; not pure forward latency. "
            "Measured time excludes initialization and warmup.",
        },
    }
    try:
        import psutil

        output["runtime"]["process_rss_bytes_at_end"] = psutil.Process().memory_info().rss
    except ImportError:
        output["runtime"]["process_rss_bytes_at_end"] = None
    if sha256(args.manifest) != manifest_hash:
        raise ValueError("Manifest changed during inference")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("x") as handle:
        handle.write(json.dumps(output, indent=2, allow_nan=False) + "\n")
    try:
        # A hard link atomically refuses an output created by another process during inference.
        os.link(temporary, args.out)
    finally:
        temporary.unlink()
    print(
        json.dumps(
            {
                "output": str(args.out),
                "frames": len(frames),
                "complete_manifest": output["complete_manifest"],
                "runtime": output["runtime"],
            },
            indent=2,
        ),
        flush=True,
    )
    return output


if __name__ == "__main__":
    run(parse_args())
