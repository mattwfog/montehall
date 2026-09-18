"""Self-training dataset builder: pipeline artifacts -> COCO detection dataset.

Labels harvested per the ruled bootstrap->validate->distill loop:
- player: high-confidence person detections (the model already knows these;
  included so the fine-tune doesn't forget them)
- ball: confident ball detections PLUS trajectory interpolation between
  temporally-close pairs — the interpolated frames are exactly the occluded/
  blurred cases the pretrained detector misses, which is where the recall
  gain lives
- rim: VLM-located, geometry-validated boxes propagated across their camera
  segment (static within a segment)

Output: <dataset_dir>/{train,valid}/_annotations.coco.json + JPEG frames
(the rfdetr fine-tune layout). Val split by trailing frame range so eval
is temporally held out.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from montehall_cv.pipeline.video import decode_frames
from montehall_cv.store.artifacts import read_stage
from montehall_cv.store.records import DetClass

CATEGORIES = [
    {"id": 1, "name": "player"},
    {"id": 2, "name": "ball"},
    {"id": 3, "name": "rim"},
]
PLAYER_MIN_CONF = 0.6
BALL_MIN_CONF = 0.45
MAX_INTERP_GAP_FRAMES = 6
VAL_FRACTION = 0.15


def build(video: Path, job_dir: Path, dataset_dir: Path, max_frames: int = 1500) -> dict:
    started = time.monotonic()
    labels = _harvest(job_dir)
    frame_ids = sorted(labels)
    if len(frame_ids) > max_frames:
        picks = np.linspace(0, len(frame_ids) - 1, max_frames).astype(int)
        frame_ids = [frame_ids[i] for i in sorted(set(picks.tolist()))]
    split_at = frame_ids[int(len(frame_ids) * (1 - VAL_FRACTION))]

    from PIL import Image

    coco = {
        "train": {"images": [], "annotations": [], "categories": CATEGORIES},
        "valid": {"images": [], "annotations": [], "categories": CATEGORIES},
    }
    for split in coco:
        (dataset_dir / split).mkdir(parents=True, exist_ok=True)

    wanted = set(frame_ids)
    ann_id = 1
    n_ball = n_interp = n_rim = 0
    for frame in decode_frames(video):
        if frame.frame_idx not in wanted:
            continue
        split = "train" if frame.frame_idx < split_at else "valid"
        name = f"frame_{frame.frame_idx:06d}.jpg"
        h, w = frame.image.shape[:2]
        Image.fromarray(frame.image).save(dataset_dir / split / name, quality=90)
        image_id = frame.frame_idx
        coco[split]["images"].append(
            {"id": image_id, "file_name": name, "width": w, "height": h}
        )
        for label in labels[frame.frame_idx]:
            x1, y1, x2, y2, cat, interp = label
            coco[split]["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            if cat == 2:
                n_ball += 1
                n_interp += int(interp)
            elif cat == 3:
                n_rim += 1
        wanted.discard(frame.frame_idx)
        if not wanted:
            break

    for split, data in coco.items():
        with open(dataset_dir / split / "_annotations.coco.json", "w") as f:
            json.dump(data, f)

    return {
        "frames": len(frame_ids),
        "train_frames": len(coco["train"]["images"]),
        "valid_frames": len(coco["valid"]["images"]),
        "annotations": ann_id - 1,
        "ball_labels": n_ball,
        "ball_interpolated": n_interp,
        "rim_labels": n_rim,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def _harvest(job_dir: Path) -> dict[int, list[tuple]]:
    labels: dict[int, list[tuple]] = defaultdict(list)

    balls: list[dict] = []
    for row in read_stage(job_dir / "localized").to_pylist():
        if row["cls"] == int(DetClass.PERSON) and row["conf"] >= PLAYER_MIN_CONF:
            labels[row["frame_idx"]].append(
                (row["x1"], row["y1"], row["x2"], row["y2"], 1, False)
            )
        elif row["cls"] == int(DetClass.BALL) and row["conf"] >= BALL_MIN_CONF:
            balls.append(row)
            labels[row["frame_idx"]].append(
                (row["x1"], row["y1"], row["x2"], row["y2"], 2, False)
            )

    balls.sort(key=lambda r: r["frame_idx"])
    for prev, cur in zip(balls, balls[1:], strict=False):
        gap = cur["frame_idx"] - prev["frame_idx"]
        if not 1 < gap <= MAX_INTERP_GAP_FRAMES:
            continue
        for step in range(1, gap):
            t = step / gap
            box = tuple(
                prev[k] * (1 - t) + cur[k] * t for k in ("x1", "y1", "x2", "y2")
            )
            labels[prev["frame_idx"] + step].append((*box, 2, True))

    for rim in read_stage(job_dir / "rims").to_pylist():
        if not rim["validated"]:
            continue
        for frame_idx in list(labels):
            if rim["frame_start"] <= frame_idx <= rim["frame_end"]:
                labels[frame_idx].append(
                    (rim["x1"], rim["y1"], rim["x2"], rim["y2"], 3, False)
                )
    return dict(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=1500)
    args = parser.parse_args()
    print(json.dumps(build(args.video, args.job_dir, args.dataset_dir, args.max_frames), indent=2))


if __name__ == "__main__":
    main()
