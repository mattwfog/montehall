"""Merge COCO detection datasets from multiple sources into one rfdetr layout.

Each source contributes its train/valid splits (a bare `test` split folds
into train). Category names are mapped to OUR ids (1=player, 2=ball, 3=rim)
by lowercased name so externally-labeled sets (Roboflow Universe vocab:
"Hoop", "basketball", "referee", ...) land on the right class; unmapped
categories are dropped, and images left with zero annotations are dropped
with them. Images are hardlinked when the filesystem allows (copy fallback)
and prefixed per-source to avoid filename collisions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

CATEGORIES = [
    {"id": 1, "name": "player"},
    {"id": 2, "name": "ball"},
    {"id": 3, "name": "rim"},
]
# PERSON means "any human before role classification" (records.DetClass), so
# referee-class labels from external sets fold into player.
NAME_TO_ID = {
    "player": 1, "person": 1, "referee": 1, "ref": 1, "players": 1,
    "ball": 2, "basketball": 2, "sports ball": 2, "basketballs": 2,
    "rim": 3, "hoop": 3, "basketball hoop": 3, "basketball-hoop": 3, "rims": 3,
}
SPLIT_ALIASES = {"train": "train", "valid": "valid", "val": "valid", "test": "train"}


def merge(sources: list[Path], out_dir: Path) -> dict:
    started = time.monotonic()
    merged = {
        split: {"images": [], "annotations": [], "categories": CATEGORIES}
        for split in ("train", "valid")
    }
    for split in merged:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    next_image_id = 1
    next_ann_id = 1
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for src_idx, source in enumerate(sources):
        splits = _find_splits(source)
        if not splits:
            raise FileNotFoundError(f"no COCO splits under {source}")
        for split_dir, target_split in splits:
            data = json.loads((split_dir / "_annotations.coco.json").read_text())
            cat_map = {
                c["id"]: NAME_TO_ID[str(c["name"]).strip().lower()]
                for c in data["categories"]
                if str(c["name"]).strip().lower() in NAME_TO_ID
            }
            anns_by_image: dict[int, list[dict]] = defaultdict(list)
            for ann in data["annotations"]:
                target = cat_map.get(ann["category_id"])
                if target is not None:
                    anns_by_image[ann["image_id"]].append({**ann, "category_id": target})

            for image in data["images"]:
                anns = anns_by_image.get(image["id"])
                if not anns:
                    stats[source.name]["images_dropped_empty"] += 1
                    continue
                new_name = f"s{src_idx}_{image['file_name']}"
                _place(split_dir / image["file_name"], out_dir / target_split / new_name)
                merged[target_split]["images"].append(
                    {**image, "id": next_image_id, "file_name": new_name}
                )
                for ann in anns:
                    merged[target_split]["annotations"].append(
                        {**ann, "id": next_ann_id, "image_id": next_image_id}
                    )
                    stats[source.name][f"class_{ann['category_id']}"] += 1
                    next_ann_id += 1
                stats[source.name][f"images_{target_split}"] += 1
                next_image_id += 1

    for split, data in merged.items():
        with open(out_dir / split / "_annotations.coco.json", "w") as f:
            json.dump(data, f)

    return {
        "sources": {k: dict(v) for k, v in sorted(stats.items())},
        "train_images": len(merged["train"]["images"]),
        "valid_images": len(merged["valid"]["images"]),
        "annotations": next_ann_id - 1,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def _find_splits(source: Path) -> list[tuple[Path, str]]:
    """(split_dir, target_split) pairs; a bare COCO dir counts as train."""
    found = [
        (source / name, SPLIT_ALIASES[name])
        for name in SPLIT_ALIASES
        if (source / name / "_annotations.coco.json").exists()
    ]
    if not found and (source / "_annotations.coco.json").exists():
        found = [(source, "train")]
    return found


def _place(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True,
                        help="dataset dir (repeatable); expects train/valid/test subdirs or a bare COCO dir")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.source, args.out), indent=2))


if __name__ == "__main__":
    main()
