"""A small, frozen TRAIN-only soccer corpus shared by detector candidates."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from soccerviz.core.assets import sha256
from soccerviz.datasets.soccernet_adapter import validate_version

CLASSES = ("player", "goalkeeper", "referee", "ball")
SEED = 20260907
DATASET_REVISION = "3cc710eb6d53a23350a3d2863311c1c0a2e645d7"
MAX_DOWNLOAD_BYTES = 250_000_000
FRAME_IDS = tuple(range(1, 751, 10))
RF_TRAINING = {
    "epochs": 5,
    "batch_size": 2,
    "grad_accum_steps": 4,
    "seed": SEED,
    "resolution": 576,
    "lr": 1e-4,
    "lr_encoder": 1.5e-4,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "use_ema": True,
    "early_stopping": False,
    "multi_scale": False,
    "expanded_scales": False,
    "scale_jitter": False,
    "augmentation_backend": "torchvision",
    "checkpoint_interval": 1,
    "run_test": False,
    "tensorboard": False,
    "wandb": False,
    "mlflow": False,
    "progress_bar": "tqdm",
}
ROBOFLOW_SCHEMA = "detector-training-protocol/roboflow-v1"
ROBOFLOW_SOURCE_SPLIT = "roboflow"
ROBOFLOW_HOLDOUT_FRACTION = 0.15
ROBOFLOW_IMAGE_NAME = re.compile(r"^(?P<video>[0-9a-f]{6})_\d+_\d+_png\.rf\.[0-9a-f]{32}\.jpg$")


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_partition(protocol):
    if protocol.get("source_split") != "train" or protocol.get("revision") != DATASET_REVISION:
        raise ValueError("Only the pinned official SoccerNet TRAIN split may supply training data")
    sequences = protocol["sequences"]
    if len(sequences) != 3 or len({s["sequence"] for s in sequences}) != 3:
        raise ValueError("The pilot requires exactly three different sequences")
    games = [str(s["game_id"]) for s in sequences]
    if len(set(games)) != 3:
        raise ValueError("Training and internal development must use distinct game IDs")
    if set(games) & set(map(str, protocol["excluded_external_game_ids"])):
        raise ValueError("A training game overlaps the external development benchmark")
    if Counter(s["partition"] for s in sequences) != {"train": 2, "valid": 1}:
        raise ValueError("Expected two training games and one internal-development game")
    if protocol.get("training") != RF_TRAINING:
        raise ValueError("Training configuration differs from the fixed five-epoch pilot")
    for sequence in sequences:
        if sequence["frame_ids"] != list(FRAME_IDS):
            raise ValueError("Each game must use the fixed 75 source frames at stride ten")


def freeze_protocol(root, inventory, external_manifest):
    root, external_manifest = Path(root), Path(external_manifest)
    external = json.loads(external_manifest.read_text())
    if external.get("schema") != "vision-benchmark/v1" or external.get("split") != "valid":
        raise ValueError("Expected the existing external validation development manifest")
    sequences = []
    for index, row in enumerate(inventory["selected"]):
        sequence = row["sequence"]
        path = root / "sources" / "train" / sequence / "Labels-GameState.json"
        if sha256(path) != row["sha256"]:
            raise ValueError("Selection annotation hash changed")
        labels = json.loads(path.read_text())
        validate_version(labels)
        info = labels["info"]
        if str(info["game_id"]) != str(row["game_id"]) or info["name"] != sequence:
            raise ValueError("Selected sequence identity differs from its official annotation")
        images = {int(Path(im["file_name"]).stem): im for im in labels["images"]}
        if any(
            fid not in images
            or not images[fid].get("is_labeled")
            or not images[fid].get("has_labeled_person")
            for fid in FRAME_IDS
        ):
            raise ValueError("Selected source frames lack official annotation coverage")
        sequences.append(
            {
                "sequence": sequence,
                "game_id": str(row["game_id"]),
                "partition": "train" if index < 2 else "valid",
                "frame_ids": list(FRAME_IDS),
                "annotation_path": str(path.relative_to(root)),
                "annotation_sha256": row["sha256"],
                "annotation_member": row["member"],
            }
        )
    protocol = {
        "schema": "detector-training-protocol/v1",
        "source_split": "train",
        "repository": "SoccerNet/SN-GSR-2024",
        "revision": DATASET_REVISION,
        "max_download_bytes": MAX_DOWNLOAD_BYTES,
        "sequences": sequences,
        "external_development_manifest_sha256": sha256(external_manifest),
        "excluded_external_game_ids": sorted(str(s["game_id"]) for s in external["sources"]),
        "selection": inventory.get(
            "selection_note",
            "First eligible distinct TRAIN games from "
            "bounded metadata-only discovery; no model scores used.",
        ),
        "internal_development": "75 frames from one distinct official TRAIN game; only these select checkpoints",
        "external_development_use": "Never training, checkpoint selection, early stopping, or tuning",
        "class_names": list(CLASSES),
        "training": RF_TRAINING,
        "ignore_policy": "Mask official ignore polygons, other objects, and explicitly ignored/difficult/crowd "
        "boxes in both formats. Omit boxes with >50% mask overlap; retain smaller occlusions.",
        "annotations": "Official public SoccerNet annotations, not new manual review by this project",
        "limitations": [
            "225 correlated frames from three games are a small pilot, not a generalization study",
            "Unknown public pretrained-checkpoint training overlap",
            "Masked ignore regions may introduce image artifacts",
            "Five epochs without search establish feasibility, not convergence",
        ],
    }
    validate_partition(protocol)
    write_new_json(root / "protocol.json", protocol)
    return protocol


def clipped_xywh(annotation, width, height):
    box = annotation.get("bbox_image")
    if not isinstance(box, dict):
        raise TypeError("Object lacks an image box")
    x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError("Invalid source box")
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Box lies outside source image")
    return [x0, y0, x1 - x0, y1 - y0]


def explicit_ignore(annotation):
    for fields in (annotation, annotation.get("attributes") or {}):
        for key in ("ignore", "ignored", "difficult", "iscrowd"):
            if str(fields.get(key, "")).lower() in {"1", "true", "yes"}:
                return True
    return False


def _mask_box(mask, box):
    x, y, w, h = box
    mask[math.floor(y) : math.ceil(y + h), math.floor(x) : math.ceil(x + w)] = 1


def training_annotations(image, annotations, categories):
    """Return shared labels and an explicit ignored-pixel mask for both exporters."""
    width, height = image["width"], image["height"]
    mask = np.zeros((height, width), dtype=np.uint8)
    dropped, candidates, polygons = [], [], []
    xs, ys = image.get("ignore_regions_x", []), image.get("ignore_regions_y", [])
    if xs and isinstance(xs[0], (int, float)):
        xs, ys = [xs], [ys]
    if len(xs) != len(ys):
        raise ValueError("Invalid ignore polygons")
    for x, y in zip(xs, ys, strict=True):
        if len(x) != len(y) or len(x) < 3:
            raise ValueError("Invalid ignore polygon")
        points = np.array(list(zip(x, y, strict=True)), dtype=np.float64)
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite ignore polygon")
        polygons.append(points.tolist())
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    for annotation in annotations:
        if annotation.get("supercategory") != "object":
            continue
        role = categories[annotation["category_id"]]
        box = clipped_xywh(annotation, width, height)
        reason = (
            "other_category"
            if role == "other"
            else "explicit_ignore"
            if explicit_ignore(annotation)
            else None
        )
        if reason:
            _mask_box(mask, box)
            dropped.append(
                {"source_annotation_id": annotation["id"], "reason": reason, "bbox_xywh": box}
            )
            continue
        if role not in CLASSES:
            raise ValueError(f"Unknown object class {role}")
        candidates.append(
            {
                "class_id": CLASSES.index(role),
                "role": role,
                "bbox_xywh": box,
                "source_annotation_id": annotation["id"],
            }
        )
    retained = []
    for candidate in candidates:
        x, y, w, h = candidate["bbox_xywh"]
        region = mask[math.floor(y) : math.ceil(y + h), math.floor(x) : math.ceil(x + w)]
        if region.size and region.mean() > 0.5:
            dropped.append({**candidate, "reason": "majority_in_ignore_region"})
        else:
            retained.append(candidate)
    # Omitted objects must not become unmasked false-negative background.
    for item in dropped:
        _mask_box(mask, item["bbox_xywh"])
    return (
        retained,
        mask,
        {
            "dropped_annotations": dropped,
            "ignore_polygons": polygons,
            "masked_pixels": int(mask.sum()),
        },
    )


def yolo_line(annotation, width, height):
    x, y, w, h = annotation["bbox_xywh"]
    return f"{annotation['class_id']} {(x + w / 2) / width:.10f} {(y + h / 2) / height:.10f} {w / width:.10f} {h / height:.10f}"


def export_corpus(root):
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Training manifest is already frozen")
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    validate_partition(protocol)
    downloads = json.loads((root / "download-summary.json").read_text())
    members = {row["relative_path"]: row for row in downloads["files"]}
    frames, files, census = [], [], Counter()
    categories_coco = [
        {"id": i + 1, "name": name, "supercategory": "soccer"} for i, name in enumerate(CLASSES)
    ]
    coco = {
        split: {
            "info": {"description": "SoccerNet TRAIN pilot; internal development is also TRAIN"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories_coco,
        }
        for split in ("train", "valid")
    }
    image_id, annotation_id = 0, 0
    for sequence in protocol["sequences"]:
        source_path = root / sequence["annotation_path"]
        if sha256(source_path) != sequence["annotation_sha256"]:
            raise ValueError("Source annotation changed after protocol freeze")
        source = json.loads(source_path.read_text())
        categories = {row["id"]: row["name"] for row in source["categories"]}
        annotations = {}
        for row in source["annotations"]:
            annotations.setdefault(row["image_id"], []).append(row)
        images = {int(Path(im["file_name"]).stem): im for im in source["images"]}
        split = sequence["partition"]
        for frame_id in sequence["frame_ids"]:
            image = images[frame_id]
            relative = f"sources/train/{sequence['sequence']}/img1/{Path(image['file_name']).name}"
            member = members[relative]
            source_image = root / relative
            if sha256(source_image) != member["sha256"]:
                raise ValueError("Downloaded training image changed")
            retained, mask, ignore = training_annotations(
                image, annotations.get(image["image_id"], []), categories
            )
            name = f"{sequence['sequence']}_{frame_id:06d}.jpg"
            coco_path = root / "coco" / split / name
            yolo_path = root / "yolo" / "images" / split / name
            coco_path.parent.mkdir(parents=True, exist_ok=True)
            yolo_path.parent.mkdir(parents=True, exist_ok=True)
            if coco_path.exists() or yolo_path.exists():
                raise FileExistsError("Partial export exists; use a new output corpus directory")
            if mask.any():
                pixels = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
                if pixels is None or pixels.shape[:2] != mask.shape:
                    raise ValueError("Source image resolution differs from annotations")
                pixels[mask.astype(bool)] = 0
                if not cv2.imwrite(str(coco_path), pixels, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError("Could not write masked training image")
            else:
                shutil.copyfile(source_image, coco_path)
            os.link(coco_path, yolo_path)
            label_path = root / "yolo" / "labels" / split / f"{Path(name).stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with label_path.open("x") as handle:
                handle.write(
                    "".join(
                        yolo_line(row, image["width"], image["height"]) + "\n" for row in retained
                    )
                )
            image_id += 1
            coco[split]["images"].append(
                {
                    "id": image_id,
                    "file_name": name,
                    "width": image["width"],
                    "height": image["height"],
                }
            )
            for row in retained:
                annotation_id += 1
                box = row["bbox_xywh"]
                coco[split]["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": row["class_id"] + 1,
                        "bbox": box,
                        "area": box[2] * box[3],
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                census[(split, row["role"])] += 1
            frames.append(
                {
                    "sequence": sequence["sequence"],
                    "game_id": sequence["game_id"],
                    "partition": split,
                    "frame_id": frame_id,
                    "source_image_sha256": member["sha256"],
                    "width": image["width"],
                    "height": image["height"],
                    "image_path": str(coco_path.relative_to(root)),
                    "image_sha256": sha256(coco_path),
                    "annotations": retained,
                    **ignore,
                }
            )
            for path in (coco_path, yolo_path, label_path):
                files.append({"path": str(path.relative_to(root)), "sha256": sha256(path)})
    for split, data in coco.items():
        path = root / "coco" / split / "_annotations.coco.json"
        write_new_json(path, data)
        files.append({"path": str(path.relative_to(root)), "sha256": sha256(path)})
    yolo_yaml = root / "yolo" / "data.yaml"
    with yolo_yaml.open("x") as handle:
        handle.write(
            "train: images/train\nval: images/valid\nnames:\n"
            + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
        )
    files.append({"path": str(yolo_yaml.relative_to(root)), "sha256": sha256(yolo_yaml)})
    manifest = {
        "schema": "detector-training/v1",
        "protocol_sha256": sha256(protocol_path),
        "source_split": "train",
        "frames": frames,
        "files": files,
        "class_names": list(CLASSES),
        "training": RF_TRAINING,
        "source_download_summary_sha256": sha256(root / "download-summary.json"),
        "census": {
            split: {name: census[(split, name)] for name in CLASSES} for split in ("train", "valid")
        },
        "frame_counts": dict(Counter(f["partition"] for f in frames)),
        "masked_frames": sum(f["masked_pixels"] > 0 for f in frames),
        "labels_are_model_input": True,
        "external_development_is_model_input": False,
    }
    write_new_json(manifest_path, manifest)
    return manifest


def roboflow_video_id(file_name):
    match = ROBOFLOW_IMAGE_NAME.match(file_name)
    if not match:
        raise ValueError(f"Unexpected Roboflow image name: {file_name}")
    return match.group("video")


def read_roboflow_export(source_dir):
    """Union of every Roboflow split; the upstream random split leaks videos, so it is discarded."""
    source_dir = Path(source_dir).resolve()
    images = {}
    for split in ("train", "valid", "test"):
        path = source_dir / split / "_annotations.coco.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        names = {row["id"]: row["name"] for row in data["categories"]}
        by_image = {}
        for row in data["annotations"]:
            by_image.setdefault(row["image_id"], []).append(row)
        for image in data["images"]:
            stem = image["file_name"].rsplit("_png.rf.", 1)[0]
            if stem in images:
                raise ValueError(f"Image appears in two Roboflow splits: {image['file_name']}")
            annotations = []
            for row in by_image.get(image["id"], []):
                name = names[row["category_id"]]
                if name not in CLASSES:
                    raise ValueError(f"Roboflow class {name!r} is not one of {CLASSES}")
                x, y, w, h = (float(v) for v in row["bbox"])
                clipped = clipped_xywh(
                    {"bbox_image": {"x": x, "y": y, "w": w, "h": h}},
                    image["width"],
                    image["height"],
                )
                annotations.append(
                    {
                        "class_id": CLASSES.index(name),
                        "role": name,
                        "bbox_xywh": clipped,
                        "source_annotation_id": str(row["id"]),
                        "source_split": split,
                    }
                )
            file_path = source_dir / split / image["file_name"]
            images[stem] = {
                "stem": stem,
                "video": roboflow_video_id(image["file_name"]),
                "file_name": image["file_name"],
                "source_split": split,
                "source_path": str(file_path),
                "source_image_sha256": sha256(file_path),
                "width": image["width"],
                "height": image["height"],
                "annotations": sorted(annotations, key=lambda a: a["source_annotation_id"]),
            }
    if not images:
        raise ValueError("No Roboflow COCO splits found")
    return images


def roboflow_partition(images, holdout_fraction=ROBOFLOW_HOLDOUT_FRACTION):
    """Video-disjoint split: walk video ids in descending order into valid until the fraction is met."""
    by_video = Counter(row["video"] for row in images.values())
    target = math.ceil(holdout_fraction * len(images))
    valid, held = [], 0
    for video in sorted(by_video, reverse=True):
        if held >= target:
            break
        valid.append(video)
        held += by_video[video]
    return {video: ("valid" if video in valid else "train") for video in by_video}


def freeze_roboflow_protocol(root, source_dir, external_manifest):
    root, source_dir, external_manifest = (
        Path(root),
        Path(source_dir).resolve(),
        Path(external_manifest),
    )
    source_record = json.loads((source_dir / "source.json").read_text())
    external = json.loads(external_manifest.read_text())
    if external.get("schema") != "vision-benchmark/v1" or external.get("split") != "valid":
        raise ValueError("Expected the existing external validation development manifest")
    images = read_roboflow_export(source_dir)
    partition = roboflow_partition(images)
    videos = {
        video: {
            "partition": partition[video],
            "images": sorted(stem for stem, row in images.items() if row["video"] == video),
        }
        for video in sorted(partition)
    }
    protocol = {
        "schema": ROBOFLOW_SCHEMA,
        "source_split": ROBOFLOW_SOURCE_SPLIT,
        "source": {
            "project_id": source_record["project_id"],
            "version_id": source_record["version_id"],
            "license": source_record.get("license"),
            "archive_sha256": source_record["archive_sha256"],
            "preprocessing": source_record.get("preprocessing"),
            "source_dir": str(source_dir),
            "source_json_sha256": sha256(source_dir / "source.json"),
        },
        "videos": videos,
        "images": {
            stem: {
                "video": row["video"],
                "file_name": row["file_name"],
                "source_split": row["source_split"],
                "source_image_sha256": row["source_image_sha256"],
            }
            for stem, row in sorted(images.items())
        },
        "partition_rule": (
            f"Video-disjoint. Roboflow's own random split is discarded. Video ids sorted descending "
            f"are held out for internal development until at least {ROBOFLOW_HOLDOUT_FRACTION:.0%} "
            "of images are held; no model scores used."
        ),
        "external_development_manifest_sha256": sha256(external_manifest),
        "external_development_source": "SoccerNet SN-GSR-2024 valid; a different video source from this corpus",
        "internal_development": "Held-out videos only select checkpoints",
        "external_development_use": "Never training, checkpoint selection, early stopping, or tuning",
        "class_names": list(CLASSES),
        "training": RF_TRAINING,
        "annotations": "Public Roboflow Universe annotations, not new manual review by this project",
        "limitations": [
            "Roboflow labels have no stated exhaustiveness guarantee, in particular for the ball",
            "Unknown public pretrained-checkpoint training overlap",
            "Five epochs without search establish feasibility, not convergence",
        ],
    }
    validate_roboflow_partition(protocol)
    write_new_json(root / "protocol.json", protocol)
    return protocol


def validate_roboflow_partition(protocol):
    if (
        protocol.get("schema") != ROBOFLOW_SCHEMA
        or protocol.get("source_split") != ROBOFLOW_SOURCE_SPLIT
    ):
        raise ValueError("Expected a Roboflow detector-training protocol")
    if protocol.get("training") != RF_TRAINING:
        raise ValueError("Training configuration differs from the fixed five-epoch pilot")
    videos, images = protocol["videos"], protocol["images"]
    partitions = Counter(v["partition"] for v in videos.values())
    if set(partitions) != {"train", "valid"}:
        raise ValueError("Expected both train and valid videos")
    seen = Counter()
    for video, row in videos.items():
        for stem in row["images"]:
            seen[stem] += 1
            if images[stem]["video"] != video or not stem.startswith(video + "_"):
                raise ValueError("Image assigned to a video it does not belong to")
    if seen.keys() != images.keys() or any(n != 1 for n in seen.values()):
        raise ValueError("Every image must belong to exactly one video partition")
    held = sum(len(v["images"]) for v in videos.values() if v["partition"] == "valid")
    if held < ROBOFLOW_HOLDOUT_FRACTION * len(images):
        raise ValueError("Internal development holdout is below the fixed fraction")


def export_roboflow_corpus(root):
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Training manifest is already frozen")
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    validate_roboflow_partition(protocol)
    source_dir = Path(protocol["source"]["source_dir"])
    if sha256(source_dir / "source.json") != protocol["source"]["source_json_sha256"]:
        raise ValueError("Roboflow source record changed after protocol freeze")
    images = read_roboflow_export(source_dir)
    if images.keys() != protocol["images"].keys():
        raise ValueError("Roboflow export images differ from the frozen protocol")
    frames, files, census = [], [], Counter()
    categories_coco = [
        {"id": i + 1, "name": name, "supercategory": "soccer"} for i, name in enumerate(CLASSES)
    ]
    coco = {
        split: {
            "info": {
                "description": f"Roboflow {protocol['source']['version_id']}; video-disjoint split"
            },
            "licenses": [{"id": 1, "name": protocol["source"].get("license") or "unstated"}],
            "images": [],
            "annotations": [],
            "categories": categories_coco,
        }
        for split in ("train", "valid")
    }
    image_id, annotation_id = 0, 0
    for video, row in protocol["videos"].items():
        split = row["partition"]
        for stem in row["images"]:
            image = images[stem]
            if image["source_image_sha256"] != protocol["images"][stem]["source_image_sha256"]:
                raise ValueError("Roboflow source image changed after protocol freeze")
            name = f"{stem}.jpg"
            coco_path = root / "coco" / split / name
            yolo_path = root / "yolo" / "images" / split / name
            coco_path.parent.mkdir(parents=True, exist_ok=True)
            yolo_path.parent.mkdir(parents=True, exist_ok=True)
            if coco_path.exists() or yolo_path.exists():
                raise FileExistsError("Partial export exists; use a new output corpus directory")
            shutil.copyfile(image["source_path"], coco_path)
            os.link(coco_path, yolo_path)
            label_path = root / "yolo" / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            retained = image["annotations"]
            with label_path.open("x") as handle:
                handle.write(
                    "".join(yolo_line(a, image["width"], image["height"]) + "\n" for a in retained)
                )
            image_id += 1
            coco[split]["images"].append(
                {
                    "id": image_id,
                    "file_name": name,
                    "width": image["width"],
                    "height": image["height"],
                }
            )
            for a in retained:
                annotation_id += 1
                box = a["bbox_xywh"]
                coco[split]["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": a["class_id"] + 1,
                        "bbox": box,
                        "area": box[2] * box[3],
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                census[(split, a["role"])] += 1
            frames.append(
                {
                    "sequence": video,
                    "game_id": video,
                    "partition": split,
                    "frame_id": stem,
                    "source_image_sha256": image["source_image_sha256"],
                    "width": image["width"],
                    "height": image["height"],
                    "image_path": str(coco_path.relative_to(root)),
                    "image_sha256": sha256(coco_path),
                    "annotations": retained,
                    "masked_pixels": 0,
                }
            )
            for path in (coco_path, yolo_path, label_path):
                files.append({"path": str(path.relative_to(root)), "sha256": sha256(path)})
    for split, data in coco.items():
        path = root / "coco" / split / "_annotations.coco.json"
        write_new_json(path, data)
        files.append({"path": str(path.relative_to(root)), "sha256": sha256(path)})
    yolo_yaml = root / "yolo" / "data.yaml"
    with yolo_yaml.open("x") as handle:
        handle.write(
            "train: images/train\nval: images/valid\nnames:\n"
            + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
        )
    files.append({"path": str(yolo_yaml.relative_to(root)), "sha256": sha256(yolo_yaml)})
    manifest = {
        "schema": "detector-training/v1",
        "protocol_sha256": sha256(protocol_path),
        "source_split": ROBOFLOW_SOURCE_SPLIT,
        "source": protocol["source"],
        "frames": frames,
        "files": files,
        "class_names": list(CLASSES),
        "training": RF_TRAINING,
        "census": {
            split: {name: census[(split, name)] for name in CLASSES} for split in ("train", "valid")
        },
        "frame_counts": dict(Counter(f["partition"] for f in frames)),
        "video_counts": {
            split: sum(1 for v in protocol["videos"].values() if v["partition"] == split)
            for split in ("train", "valid")
        },
        "masked_frames": 0,
        "labels_are_model_input": True,
        "external_development_is_model_input": False,
    }
    write_new_json(manifest_path, manifest)
    return manifest


def verify_exported_files(root, manifest):
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or sha256(path) != row["sha256"]:
            raise ValueError(f"Exported training file changed or escaped corpus: {row['path']}")
    for frame in manifest["frames"]:
        image = cv2.imread(str(root / frame["image_path"]), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (frame["height"], frame["width"]):
            raise ValueError("Training image dimensions differ from the frozen annotations")
    for split in ("train", "valid"):
        coco = json.loads((root / "coco" / split / "_annotations.coco.json").read_text())
        if [(c["id"], c["name"]) for c in coco["categories"]] != list(enumerate(CLASSES, 1)):
            raise ValueError("COCO category indices differ from the fixed four soccer classes")


def verify_roboflow_corpus(root, manifest, protocol):
    validate_roboflow_partition(protocol)
    if (
        manifest.get("schema") != "detector-training/v1"
        or manifest.get("source_split") != ROBOFLOW_SOURCE_SPLIT
    ):
        raise ValueError("Expected a Roboflow detector corpus")
    if manifest["protocol_sha256"] != sha256(root / "protocol.json"):
        raise ValueError("Training protocol changed after corpus freeze")
    expected = {
        (video, row["partition"], stem)
        for video, row in protocol["videos"].items()
        for stem in row["images"]
    }
    actual = {(f["sequence"], f["partition"], f["frame_id"]) for f in manifest["frames"]}
    if actual != expected or len(manifest["frames"]) != len(expected):
        raise ValueError("Exported frame identities differ from the frozen video partitions")
    expected_counts = Counter()
    for row in protocol["videos"].values():
        expected_counts[row["partition"]] += len(row["images"])
    if manifest["frame_counts"] != dict(expected_counts):
        raise ValueError("Unexpected corpus size")
    verify_exported_files(root, manifest)
    return manifest, protocol


def verify_corpus(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol.get("schema") == ROBOFLOW_SCHEMA:
        return verify_roboflow_corpus(root, manifest, protocol)
    validate_partition(protocol)
    if manifest.get("schema") != "detector-training/v1" or manifest.get("source_split") != "train":
        raise ValueError("Expected TRAIN-only detector corpus")
    if manifest["protocol_sha256"] != sha256(root / "protocol.json"):
        raise ValueError("Training protocol changed after corpus freeze")
    if manifest["frame_counts"] != {"train": 150, "valid": 75}:
        raise ValueError("Unexpected corpus size")
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or sha256(path) != row["sha256"]:
            raise ValueError(f"Exported training file changed or escaped corpus: {row['path']}")
    expected_frames = {
        (s["sequence"], s["game_id"], s["partition"], fid)
        for s in protocol["sequences"]
        for fid in s["frame_ids"]
    }
    actual_frames = {
        (f["sequence"], f["game_id"], f["partition"], f["frame_id"]) for f in manifest["frames"]
    }
    if actual_frames != expected_frames or len(manifest["frames"]) != len(expected_frames):
        raise ValueError("Exported frame identities differ from the frozen training partitions")
    for frame in manifest["frames"]:
        image = cv2.imread(str(root / frame["image_path"]), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (frame["height"], frame["width"]):
            raise ValueError("Training image dimensions differ from the frozen annotations")
    for split in ("train", "valid"):
        coco = json.loads((root / "coco" / split / "_annotations.coco.json").read_text())
        if [(c["id"], c["name"]) for c in coco["categories"]] != list(enumerate(CLASSES, 1)):
            raise ValueError("COCO category indices differ from the fixed four soccer classes")
    return manifest, protocol
