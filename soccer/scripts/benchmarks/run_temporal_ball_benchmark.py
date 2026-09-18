"""Evaluate native WASB temporal context on an immutable, label-free image manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from soccerviz.candidates.detector_backends import gpu_memory
from soccerviz.candidates.temporal_ball import (
    SOCCER_CHECKPOINT_SHA256,
    WASBDetector,
    contiguous,
    frame_histogram,
    frame_key,
    histogram_cut_distance,
    plan_windows,
)
from soccerviz.core.assets import sha256


def path_for(frame, manifest_path, prefixes):
    path = Path(frame["image_path"])
    for source, target in prefixes:
        try:
            return target / path.relative_to(source)
        except ValueError:
            pass
    return path if path.is_absolute() else manifest_path.parent / path


def read_verified(frame, manifest_path, prefixes):
    path = path_for(frame, manifest_path, prefixes)
    if sha256(path) != frame["sha256"]:
        raise ValueError(f"Image hash changed: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (frame["height"], frame["width"]):
        raise ValueError(f"Unreadable image or incorrect dimensions: {path}")
    return image


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path("artifacts/upstreams/wasb"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tracker", choices=["peak", "online"], default="peak")
    parser.add_argument(
        "--cut-boundaries", type=Path, help="Optional JSON {sequence:[first_frame_after_cut,...]}"
    )
    parser.add_argument(
        "--cut-threshold",
        type=float,
        default=0.65,
        help="Fixed HSV-histogram cut guard; metadata labels this heuristic",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Smoke prefix only; output marked incomplete")
    parser.add_argument("--path-prefix", action="append", default=[], metavar="OLD=NEW")
    args = parser.parse_args(argv)
    if args.warmup < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("warmup must be nonnegative; limit must be positive")
    if not math.isfinite(args.cut_threshold) or not 0 <= args.cut_threshold <= 1:
        parser.error("cut-threshold must be finite and in [0,1]")
    args.prefixes = []
    for value in args.path_prefix:
        source, sep, target = value.partition("=")
        if not sep or not source or not target:
            parser.error("path-prefix must be OLD=NEW")
        args.prefixes.append((Path(source), Path(target)))
    return args


def run(args):
    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {args.out}")
    manifest_hash = sha256(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "vision-benchmark/v1" or not manifest.get("frames"):
        raise ValueError("Expected a nonempty vision-benchmark/v1 manifest")
    all_frames = manifest["frames"]
    frames = all_frames[: args.limit] if args.limit else all_frames
    cut_keys = set()
    explicit_cuts_hash = None
    if args.cut_boundaries:
        explicit_cuts_hash = sha256(args.cut_boundaries)
        for sequence, identifiers in json.loads(args.cut_boundaries.read_text()).items():
            cut_keys.update((str(sequence), int(identifier)) for identifier in identifiers)
    unknown_cut_keys = cut_keys - {frame_key(f) for f in all_frames}
    if unknown_cut_keys:
        raise ValueError(f"Cut boundary does not belong to manifest: {sorted(unknown_cut_keys)}")
    unavailable = set()
    cut_events = []
    previous, previous_hist = None, None
    for frame in frames:
        try:
            image = read_verified(frame, args.manifest, args.prefixes)
        except FileNotFoundError:
            unavailable.add(frame_key(frame))
            previous, previous_hist = None, None
            continue
        histogram = frame_histogram(image)
        if previous is not None and contiguous(previous, frame):
            distance = histogram_cut_distance(previous_hist, histogram)
            if distance >= args.cut_threshold:
                cut_keys.add(frame_key(frame))
                cut_events.append(
                    {
                        "sequence": frame["sequence"],
                        "frame_id": frame["frame_id"],
                        "histogram_distance": distance,
                    }
                )
        previous, previous_hist = frame, histogram
    windows, statuses = plan_windows(frames, cut_before=cut_keys, unavailable=unavailable)
    if not windows:
        raise ValueError("No complete, contiguous three-frame context is available")
    preflight_s = time.perf_counter() - started
    init_start = time.perf_counter()
    backend = WASBDetector(args.upstream, args.checkpoint, device=args.device, tracker=args.tracker)
    initialization_s = time.perf_counter() - init_start
    warm_start = time.perf_counter()
    warm_images = [
        read_verified(frames[i], args.manifest, args.prefixes) for i in windows[0]["indices"]
    ]
    for _ in range(args.warmup):
        backend.predict(warm_images)
    warmup_s = time.perf_counter() - warm_start
    outputs = [
        {
            "sequence": f["sequence"],
            "frame_id": f["frame_id"],
            "width": f["width"],
            "height": f["height"],
            "timestamp_s": f["timestamp_s"],
            "fps": f["fps"],
            "image_sha256": f["sha256"],
            "detections": [],
            "context_status": statuses[i],
            "context_frame_ids": [],
            "lookahead_frames": None,
        }
        for i, f in enumerate(frames)
    ]
    window_timings = []
    measured_start = time.perf_counter()
    last_segment = None
    for window_id, window in enumerate(windows):
        window_start = time.perf_counter()
        indices = window["indices"]
        if window["segment"] != last_segment:
            backend.reset_tracker()
            last_segment = window["segment"]
        images = [read_verified(frames[i], args.manifest, args.prefixes) for i in indices]
        loaded = time.perf_counter()
        candidates, transform = backend.predict(images)
        for output_slot, (index, frame_candidates) in enumerate(
            zip(indices, candidates, strict=True)
        ):
            selected = backend.select(frame_candidates)
            last_frame = frames[indices[-1]]
            outputs[index].update(
                {
                    "detections": selected,
                    "context_status": "inferred",
                    "context_frame_ids": [frames[i]["frame_id"] for i in indices],
                    "context_image_sha256": [frames[i]["sha256"] for i in indices],
                    "context_output_slot": output_slot,
                    "window_id": window_id,
                    "segment_id": window["segment"],
                    "native_candidate_count": len(frame_candidates),
                    "lookahead_frames": 2 - output_slot,
                    "lookahead_s": last_frame["timestamp_s"] - frames[index]["timestamp_s"],
                    "transform": transform,
                }
            )
        window_timings.append(
            {
                "window_id": window_id,
                "image_read_and_sha_ms": (loaded - window_start) * 1000,
                **backend.last_timing_ms,
                "end_to_end_ms": (time.perf_counter() - window_start) * 1000,
            }
        )
        if window_id == 0 or (window_id + 1) % 25 == 0 or window_id + 1 == len(windows):
            print(f"WASB: {window_id + 1}/{len(windows)} triples", flush=True)
    measured_s = time.perf_counter() - measured_start
    coverage = {}
    for sequence in dict.fromkeys(f["sequence"] for f in frames):
        selected = [f for f in outputs if f["sequence"] == sequence]
        coverage[sequence] = {
            "frames": len(selected),
            "statuses": dict(Counter(f["context_status"] for f in selected)),
            "frames_with_ball_prediction": sum(bool(f["detections"]) for f in selected),
        }
    output = {
        "schema": "vision-center-predictions/v1",
        "manifest_sha256": manifest_hash,
        "checkpoint_hashes": {"wasb_soccer": SOCCER_CHECKPOINT_SHA256},
        "backend": backend.metadata,
        "complete_manifest": len(frames) == len(all_frames),
        "frames": outputs,
        "coverage": coverage,
        "context_policy": {
            "native_step": 3,
            "input_frames": 3,
            "output_frames": 3,
            "maximum_lookahead_frames": 2,
            "missing_frames": "Explicit empty predictions; no repetition/interpolation",
            "boundaries": "Reset at sequence, cadence gap, image absence, size/shot change, "
            "explicit cut or histogram cut guard",
            "explicit_cuts_sha256": explicit_cuts_hash,
            "histogram_cut_threshold": args.cut_threshold,
            "histogram_cut_events": cut_events,
            "cut_guard_limitation": "HSV histogram cuts are heuristic. Unknown cuts below "
            "threshold can be missed; no manually verified shot "
            "boundary labels were supplied.",
        },
        "runtime": {
            "preflight_s": preflight_s,
            "initialization_s": initialization_s,
            "warmup_s": warmup_s,
            "warmup_iterations": args.warmup,
            "measured_s": measured_s,
            "inferred_frames": len(windows) * 3,
            "inferred_frames_per_s": len(windows) * 3 / measured_s,
            "median_forward_ms_per_triple": float(
                np.median([w["forward"] for w in window_timings])
            ),
            "elapsed_before_output_write_s": time.perf_counter() - started,
            "gpu_memory": gpu_memory(args.device),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "window_timings": window_timings,
        },
    }
    if sha256(args.manifest) != manifest_hash:
        raise ValueError("Manifest changed during inference")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("x") as handle:
        handle.write(json.dumps(output, indent=2, allow_nan=False) + "\n")
    try:
        os.link(temporary, args.out)
    finally:
        temporary.unlink()
    print(
        json.dumps(
            {"output": str(args.out), "coverage": coverage, "measured_s": measured_s}, indent=2
        ),
        flush=True,
    )
    return output


if __name__ == "__main__":
    run(parse_args())
