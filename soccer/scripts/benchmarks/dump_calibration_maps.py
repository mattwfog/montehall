"""Dump one heatmap calibrator's raw output maps for chosen benchmark frames.

Runs the network only (no fit) so the fit can be replayed and diagnosed offline in an
environment without torch. One .npz per frame: maps (32 + primitives, H, W) logits,
width, height, sequence, frame_id, image_sha256. Existing files are skipped.

usage: python scripts/benchmarks/dump_calibration_maps.py <checkpoint> <manifest> <out_dir>
           --frames SNGS-021:1,SNGS-021:11 [--device mps|cpu|cuda:0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from soccerviz.candidates.pitch_keypoints import HeatmapCalibrationModel


def parse_frames(spec):
    return [(s, int(f)) for s, f in (item.split(":") for item in spec.split(","))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("manifest")
    parser.add_argument("out_dir")
    parser.add_argument("--frames", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    wanted = set(parse_frames(args.frames))
    manifest = json.loads(Path(args.manifest).read_text())
    frames = [f for f in manifest["frames"] if (f["sequence"], f["frame_id"]) in wanted]
    missing = wanted - {(f["sequence"], f["frame_id"]) for f in frames}
    if missing:
        raise SystemExit(f"frames not in manifest: {sorted(missing)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = HeatmapCalibrationModel(args.checkpoint, {}, device=args.device)
    for frame in frames:
        target = out_dir / f"{frame['sequence']}-{frame['frame_id']:06d}.npz"
        if target.exists():
            print("skip", target.name)
            continue
        image = cv2.imread(frame["image_path"])
        if image is None:
            raise SystemExit(f"unreadable image {frame['image_path']}")
        maps = model.heatmaps(image)
        np.savez_compressed(
            target,
            maps=maps.astype(np.float16),
            width=image.shape[1],
            height=image.shape[0],
            sequence=frame["sequence"],
            frame_id=frame["frame_id"],
            image_sha256=frame["sha256"],
        )
        print("wrote", target.name, maps.shape)


if __name__ == "__main__":
    main()
