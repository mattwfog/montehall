"""Frozen, source-pixel detection development benchmark (never training input).

Inference consumes only manifest.json. Source annotations and normalized targets
are separate files consumed by this evaluator after predictions have been saved.
No optional ML dependency is needed for the primary precision/recall metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import statistics
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from soccerviz.datasets.soccernet_adapter import sha256, validate_version, write_json

CLASSES = ("person", "ball")
LABELS = {
    "person": "person",
    "player": "person",
    "goalkeeper": "person",
    "referee": "person",
    "ball": "ball",
    "sports ball": "ball",
    "sports_ball": "ball",
}


def canonical_label(label):
    if not isinstance(label, str) or label.strip().lower() not in LABELS:
        raise ValueError(f"Unsupported prediction label: {label!r}")
    return LABELS[label.strip().lower()]


def frame_key(frame):
    frame_id = frame.get("frame_id")
    if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 1:
        raise ValueError("frame_id must be a positive source-frame integer")
    sequence = frame.get("sequence")
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("sequence must be nonempty")
    return sequence, frame_id


def checked_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("bbox_xyxy requires four source-pixel values")
    values = [float(value) for value in box]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite bounding box")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("Bounding box must have positive width and height")
    return values


def checked_center(center):
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        raise ValueError("center_xy requires two source-pixel values")
    values = [float(value) for value in center]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite predicted center")
    return values


def source_box(annotation):
    box = annotation["bbox_image"]
    return checked_box([box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]])


def _polygons(image):
    xs, ys = image.get("ignore_regions_x", []), image.get("ignore_regions_y", [])
    if not xs and not ys:
        return []
    if xs and isinstance(xs[0], (int, float)):
        xs, ys = [xs], [ys]
    if len(xs) != len(ys) or any(len(x) != len(y) or len(x) < 3 for x, y in zip(xs, ys)):
        raise ValueError("Unsupported source ignore polygon layout")
    return [[list(point) for point in zip(x, y)] for x, y in zip(xs, ys)]


GSR_SPLITS = ("train", "valid", "test")


def sequence_splits(protocol):
    """Official split of every protocol sequence. A single-split corpus states `split`;
    a corpus drawn from several splits (the game-disjoint calibration evaluation uses
    GSR train and test games, evaluation only, never training) lists them in
    `sequence_splits` and sets `split` to the joined names."""
    explicit = protocol.get("sequence_splits", {})
    splits = {
        sequence: explicit.get(sequence, protocol["split"]) for sequence in protocol["sequences"]
    }
    bad = {sequence: split for sequence, split in splits.items() if split not in GSR_SPLITS}
    if bad:
        raise ValueError(f"Unsupported SoccerNet GSR split for {bad}")
    return splits


def prepare_manifest(root):
    """Freeze images and evaluation-only labels from verified downloader outputs."""
    root = Path(root).resolve()
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    splits = sequence_splits(protocol)
    if protocol["purpose"] != "development":
        raise ValueError("This experiment is restricted to development corpora")
    if (root / "manifest.json").exists():
        raise ValueError("Frozen manifest already exists; use a new benchmark directory")
    frames, targets, sources, annotation_sources = [], [], [], []
    census = Counter()
    for sequence in protocol["sequences"]:
        download_path = root / "sources" / f"gsr-{splits[sequence]}-{sequence}-download.json"
        download = json.loads(download_path.read_text())
        if (download["revision"], download["split"], download["sequence"]) != (
            protocol["revision"],
            splits[sequence],
            sequence,
        ):
            raise ValueError("Download does not match frozen dataset revision/split/sequence")
        members = {Path(row["path"]).name: row for row in download["files"]}
        label_member = members["Labels-GameState.json"]
        label_path = Path(label_member["path"])
        if sha256(label_path) != label_member["sha256"]:
            raise ValueError("Source annotation SHA256 mismatch")
        source = json.loads(label_path.read_text())
        validate_version(source)
        info = source["info"]
        fps = float(info["frame_rate"])
        if fps != 25 or info["name"] != sequence:
            raise ValueError("Expected original 25-Hz SoccerNet sequence")
        categories = {row["id"]: row["name"] for row in source["categories"]}
        annotations = {}
        for row in source["annotations"]:
            if row.get("supercategory") == "object":
                annotations.setdefault(row["image_id"], []).append(row)
        selected = set(
            range(
                protocol["first_frame_id"],
                protocol["first_frame_id"]
                + protocol["frames_per_sequence"] * protocol["frame_stride"],
                protocol["frame_stride"],
            )
        )
        found = set()
        for image in source["images"]:
            frame_id = int(Path(image["file_name"]).stem)
            if frame_id not in selected:
                continue
            if not image.get("is_labeled") or not image.get("has_labeled_person"):
                raise ValueError("Frozen frame lacks official person annotation coverage")
            member = members[Path(image["file_name"]).name]
            path = Path(member["path"])
            if sha256(path) != member["sha256"]:
                raise ValueError("Downloaded image SHA256 mismatch")
            # Decode pixels to validate the annotated resolution, not just the filename.
            from PIL import Image

            with Image.open(path) as decoded:
                if decoded.size != (image["width"], image["height"]):
                    raise ValueError("Source image dimensions differ from annotations")
            frames.append(
                {
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "image_path": str(path.resolve()),
                    "sha256": member["sha256"],
                    "width": image["width"],
                    "height": image["height"],
                    "fps": fps,
                    "timestamp_s": (frame_id - 1) / fps,
                }
            )
            target = {
                "sequence": sequence,
                "frame_id": frame_id,
                "image_id": image["image_id"],
                "detections": [],
                "ignored_boxes": [],
                "ignore_polygons": _polygons(image),
            }
            for annotation in annotations.get(image["image_id"], []):
                role = categories.get(annotation["category_id"])
                if role == "other":
                    target["ignored_boxes"].append(source_box(annotation))
                    continue
                if role not in LABELS:
                    raise ValueError(f"Unknown ground-truth object category: {role}")
                label = canonical_label(role)
                target["detections"].append(
                    {
                        "bbox_xyxy": source_box(annotation),
                        "label": label,
                        "role": role,
                        "source_annotation_id": annotation["id"],
                    }
                )
                census[(sequence, label)] += 1
            targets.append(target)
            found.add(frame_id)
        if found != selected:
            raise ValueError("Not every frozen source frame was found")
        sources.append(
            {
                "sequence": sequence,
                "split": splits[sequence],
                "repository": download["repository"],
                "revision": download["revision"],
                "archive_url": download["archive_url"],
                "archive_sha256_expected": download["archive_sha256_expected"],
                "archive_hash_verified": False,
                "game_id": info.get("game_id"),
                "source_clock": {
                    key: info.get(key)
                    for key in ("clip_start", "clip_stop", "game_time_start", "game_time_stop")
                },
            }
        )
        annotation_sources.append(
            {
                "sequence": sequence,
                "path": str(label_path.resolve()),
                "sha256": label_member["sha256"],
                "annotation_version": source["info"]["version"],
                "download_manifest": str(download_path),
                "download_manifest_sha256": sha256(download_path),
            }
        )
    manifest = {
        "schema": "vision-benchmark/v1",
        "purpose": "development",
        "split": protocol["split"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "clock": "Original 1-based source frame; timestamp_s=(frame_id-1)/fps, clip-relative.",
        "sources": sources,
        "frames": sorted(frames, key=frame_key),
        "annotations_are_model_input": False,
    }
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    write_json(
        root / "ground-truth.json",
        {
            "schema": "vision-ground-truth/v1",
            "manifest_sha256": sha256(manifest_path),
            "annotations_are_model_input": False,
            "sources": annotation_sources,
            "frames": sorted(targets, key=frame_key),
        },
    )
    summary = {
        "manifest_sha256": sha256(manifest_path),
        "frames": len(frames),
        "sample_interval_s": 1 / 25,
        "continuous_duration_s_per_sequence": protocol["frames_per_sequence"] / 25,
        "sequences": [
            {
                "sequence": s["sequence"],
                "game_id": s["game_id"],
                "frames": sum(f["sequence"] == s["sequence"] for f in frames),
                "annotations": {label: census[(s["sequence"], label)] for label in CLASSES},
            }
            for s in sources
        ],
        "limitations": [
            "Development subset, not an official split leaderboard result.",
            "Unknown public checkpoint pretraining overlap.",
            "No independent source flag guarantees exhaustive ball annotation.",
        ],
    }
    write_json(root / "corpus-summary.json", summary)
    return summary


def iou(left, right):
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )
    return intersection / union if union else 0.0


def center_distance(left, right):
    return math.hypot(
        (left[0] + left[2] - right[0] - right[2]) / 2, (left[1] + left[3] - right[1] - right[3]) / 2
    )


def size_bucket(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def _in_polygon(x, y, polygon):
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        previous = current
    return inside


def _ignored(box, target, threshold):
    if any(iou(box, other) >= threshold for other in target.get("ignored_boxes", [])):
        return True
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(_in_polygon(x, y, polygon) for polygon in target.get("ignore_polygons", []))


def _point_ignored(point, target):
    x, y = point
    return any(
        left <= x <= right and top <= y <= bottom
        for left, top, right, bottom in target.get("ignored_boxes", [])
    ) or any(_in_polygon(x, y, polygon) for polygon in target.get("ignore_polygons", []))


def _match(predictions, truth, threshold, *, centers=False, point_centers=False):
    remaining, matches, unmatched = set(range(len(truth))), [], []
    for prediction_index, pred in sorted(enumerate(predictions), key=lambda row: -row[1]["score"]):
        candidates = [
            (
                (
                    math.hypot(
                        pred["center_xy"][0] - sum(truth[index]["bbox_xyxy"][::2]) / 2,
                        pred["center_xy"][1] - sum(truth[index]["bbox_xyxy"][1::2]) / 2,
                    )
                    if point_centers
                    else center_distance(pred["bbox_xyxy"], truth[index]["bbox_xyxy"])
                )
                if centers
                else -iou(pred["bbox_xyxy"], truth[index]["bbox_xyxy"]),
                index,
            )
            for index in sorted(remaining)
        ]
        value, best = min(candidates) if candidates else (math.inf, None)
        passes = value <= threshold if centers else value <= -threshold
        if best is not None and passes:
            matches.append((prediction_index, best, value if centers else -value))
            remaining.remove(best)
        else:
            unmatched.append(prediction_index)
    return matches, unmatched, remaining


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _class_metrics(
    keys, predictions, targets, label, protocol, *, centers=False, point_centers=False
):
    counts = Counter()
    sizes = {name: Counter() for name in ("small", "medium", "large")}
    errors = []
    for key in keys:
        truth = [row for row in targets[key]["detections"] if row["label"] == label]
        preds = [
            row
            for row in predictions.get(key, [])
            if row["label"] == label and row["score"] >= protocol["score_threshold"]
        ]
        threshold = protocol["ball_center_threshold_px"] if centers else protocol["iou_threshold"]
        matches, unmatched, missed = _match(
            preds, truth, threshold, centers=centers, point_centers=point_centers
        )
        false_positives = [
            index
            for index in unmatched
            if not (
                _point_ignored(preds[index]["center_xy"], targets[key])
                if point_centers
                else _ignored(preds[index]["bbox_xyxy"], targets[key], protocol["iou_threshold"])
            )
        ]
        counts.update(
            gt=len(truth),
            predictions=len(preds),
            tp=len(matches),
            fn=len(missed),
            fp=len(false_positives),
            ignored=len(unmatched) - len(false_positives),
        )
        for row in truth:
            sizes[size_bucket(row["bbox_xyxy"])]["gt"] += 1
        for _, index, value in matches:
            sizes[size_bucket(truth[index]["bbox_xyxy"])]["tp"] += 1
            if centers:
                errors.append(value)
        if not point_centers:
            for index in false_positives:
                sizes[size_bucket(preds[index]["bbox_xyxy"])]["unmatched_predicted_boxes"] += 1
    result = {
        **dict(counts),
        "precision": _ratio(counts["tp"], counts["tp"] + counts["fp"]),
        "recall": _ratio(counts["tp"], counts["gt"]),
        "false_positives_per_frame": _ratio(counts["fp"], len(keys)),
        "size_buckets": {
            name: {
                "gt": row["gt"],
                "tp": row["tp"],
                "fn": row["gt"] - row["tp"],
                "recall": _ratio(row["tp"], row["gt"]),
                "unmatched_predicted_boxes": row["unmatched_predicted_boxes"],
            }
            for name, row in sizes.items()
        },
    }
    if centers:
        ordered = sorted(errors)
        result["matched_center_error_px"] = {
            "mean": statistics.mean(errors) if errors else None,
            "median": statistics.median(errors) if errors else None,
            "p95": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)] if ordered else None,
        }
    if point_centers:
        for row in result["size_buckets"].values():
            row.pop("unmatched_predicted_boxes")
    return result


def _coco_ap(keys, predictions, targets, manifest_frames, candidate_score_floor=None):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return {
            "available": False,
            "map": None,
            "reason": "pycocotools is not installed; precision/recall are not mAP.",
        }
    if any(targets[key].get("ignore_polygons") for key in keys):
        return {
            "available": False,
            "map": None,
            "reason": "Source ignore polygons cannot be represented by this COCO bbox adapter.",
        }
    categories = [{"id": i + 1, "name": label} for i, label in enumerate(CLASSES)]
    label_id = {row["name"]: row["id"] for row in categories}
    images, annotations, detections = [], [], []
    for image_id, key in enumerate(keys, 1):
        image = manifest_frames[key]
        images.append({"id": image_id, "width": image["width"], "height": image["height"]})
        for row in targets[key]["detections"]:
            x, y, right, bottom = row["bbox_xyxy"]
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": label_id[row["label"]],
                    "bbox": [x, y, right - x, bottom - y],
                    "area": (right - x) * (bottom - y),
                    "iscrowd": 0,
                }
            )
        for box in targets[key].get("ignored_boxes", []):
            x, y, right, bottom = box
            for category in categories:
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": category["id"],
                        "bbox": [x, y, right - x, bottom - y],
                        "area": (right - x) * (bottom - y),
                        "iscrowd": 1,
                    }
                )
        for row in predictions.get(key, []):
            x, y, right, bottom = row["bbox_xyxy"]
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": label_id[row["label"]],
                    "bbox": [x, y, right - x, bottom - y],
                    "score": row["score"],
                }
            )
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = {
            "info": {},
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        coco.createIndex()
        if detections:
            detected = coco.loadRes(detections)
        else:
            detected = COCO()
            detected.dataset = {**coco.dataset, "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(coco, detected, "bbox")
        evaluator.params.imgIds = list(range(1, len(keys) + 1))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "available": True,
        "pycocotools_version": version("pycocotools"),
        "candidate_score_floor": candidate_score_floor,
        "score_truncated_input": (
            candidate_score_floor > 0 if candidate_score_floor is not None else None
        ),
        "map": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]),
        "by_class": {
            label: {"map": float(values[values >= 0].mean()) if (values >= 0).any() else None}
            for label, values in (
                (label, evaluator.eval["precision"][:, :, index, 0, -1])
                for index, label in enumerate(CLASSES)
            )
        },
        "protocol": "COCO bbox IoU .50:.95, maxDets100, all submitted scores; crowd ignore convention.",
        "limitation": "AP uses submitted predictions only; candidates removed during inference cannot be recovered. Compare identical candidate score floors.",
    }


def evaluate(manifest_path, predictions_path, ground_truth_path=None, *, verify_images=True):
    manifest_path, predictions_path = Path(manifest_path), Path(predictions_path)
    ground_truth_path = Path(ground_truth_path or manifest_path.parent / "ground-truth.json")
    manifest = json.loads(manifest_path.read_text())
    envelope = json.loads(predictions_path.read_text())
    ground_truth = json.loads(ground_truth_path.read_text())
    manifest_hash = sha256(manifest_path)
    if manifest.get("schema") != "vision-benchmark/v1":
        raise ValueError("Unsupported benchmark manifest schema")
    if envelope.get("schema") not in {"vision-predictions/v1", "vision-center-predictions/v1"}:
        raise ValueError("Unsupported predictions schema")
    point_centers = envelope["schema"] == "vision-center-predictions/v1"
    if ground_truth.get("schema") != "vision-ground-truth/v1":
        raise ValueError("Unsupported ground-truth schema")
    if (
        envelope.get("manifest_sha256") != manifest_hash
        or ground_truth.get("manifest_sha256") != manifest_hash
    ):
        raise ValueError("Manifest SHA256 provenance mismatch")
    if not envelope.get("backend") or "checkpoint_hashes" not in envelope:
        raise ValueError("Predictions require backend and checkpoint_hashes provenance")
    protocol_path = Path(manifest["protocol_path"])
    if sha256(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("Frozen protocol SHA256 mismatch")
    protocol = json.loads(protocol_path.read_text())
    supplemental_protocols = {}
    for name in ("center", "coco"):
        supplemental_path = manifest_path.parent / f"{name}-protocol.json"
        if supplemental_path.exists():
            supplemental = json.loads(supplemental_path.read_text())
            if supplemental.get("manifest_sha256") != manifest_hash:
                raise ValueError("Supplemental protocol manifest provenance mismatch")
            if name == "center" and any(
                supplemental[key] != protocol[key]
                for key in ("score_threshold", "ball_center_threshold_px")
            ):
                raise ValueError("Center protocol differs from frozen operating thresholds")
            supplemental_protocols[name] = sha256(supplemental_path)
    frames = {frame_key(row): row for row in manifest["frames"]}
    targets = {frame_key(row): row for row in ground_truth["frames"]}
    if not frames or len(frames) != len(manifest["frames"]):
        raise ValueError("Empty or duplicate manifest frames")
    if set(frames) != set(targets) or len(targets) != len(ground_truth["frames"]):
        raise ValueError("Ground-truth frame set differs from the full frozen manifest")
    for source in ground_truth.get("sources", []):
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("Source annotation SHA256 mismatch")
    if verify_images:
        for frame in frames.values():
            if sha256(frame["image_path"]) != frame["sha256"]:
                raise ValueError("Image SHA256 mismatch")
    predictions = {}
    for frame in envelope.get("frames", []):
        key = frame_key(frame)
        if key not in frames or key in predictions:
            raise ValueError("Unexpected or duplicate prediction frame")
        normalized = []
        for row in frame.get("detections", []):
            score = float(row["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Prediction score must be finite and within [0,1]")
            label = canonical_label(row["label"])
            if point_centers and label != "ball":
                raise ValueError("Center-only predictions support the ball class only")
            geometry = (
                {"center_xy": checked_center(row["center_xy"])}
                if point_centers
                else {"bbox_xyxy": checked_box(row["bbox_xyxy"])}
            )
            normalized.append(
                {
                    **row,
                    "label": label,
                    **geometry,
                    "score": score,
                }
            )
        predictions[key] = normalized
    keys = sorted(frames)
    candidate_score_floor = (
        envelope["backend"].get("confidence_threshold")
        if isinstance(envelope["backend"], dict)
        else None
    )

    def metrics(selected):
        if point_centers:
            return {
                "frames": len(selected),
                "classes": None,
                "ball_center": _class_metrics(
                    selected,
                    predictions,
                    targets,
                    "ball",
                    protocol,
                    centers=True,
                    point_centers=True,
                ),
            }
        return {
            "frames": len(selected),
            "classes": {
                label: _class_metrics(selected, predictions, targets, label, protocol)
                for label in CLASSES
            },
            "ball_center": _class_metrics(
                selected, predictions, targets, "ball", protocol, centers=True
            ),
        }

    return {
        "schema": "vision-benchmark-results/v1",
        "purpose": "development",
        "manifest_sha256": manifest_hash,
        "protocol_sha256": manifest["protocol_sha256"],
        "supplemental_protocol_sha256": supplemental_protocols,
        "predictions_schema": envelope["schema"],
        "prediction_geometry": "center-only" if point_centers else "bbox_xyxy",
        "predictions_sha256": sha256(predictions_path),
        "ground_truth_sha256": sha256(ground_truth_path),
        "backend": envelope["backend"],
        "checkpoint_hashes": envelope["checkpoint_hashes"],
        "thresholds": {
            key: protocol[key]
            for key in ("score_threshold", "iou_threshold", "ball_center_threshold_px")
        },
        "submitted_frames": len(predictions),
        "missing_prediction_frames": len(frames.keys() - predictions.keys()),
        "image_hashes_verified": verify_images,
        "overall": metrics(keys),
        "by_sequence": {
            sequence: metrics([key for key in keys if key[0] == sequence])
            for sequence in sorted({key[0] for key in keys})
        },
        "coco_bbox": (
            {
                "available": False,
                "map": None,
                "reason": "Center-only predictions have no boxes; box AP is undefined.",
            }
            if point_centers
            else _coco_ap(keys, predictions, targets, frames, candidate_score_floor)
        ),
        "limitations": [
            "Validation development subset; not a final test or official split score.",
            "Unsubmitted frames count as empty predictions.",
            "Annotated-ball recall; no separate source ball-completeness flag.",
            "Unknown public-pretraining overlap.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", type=Path, required=True)
    score = subparsers.add_parser("evaluate")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--ground-truth", type=Path)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "prepare":
        print(json.dumps(prepare_manifest(args.root), indent=2))
    else:
        if args.output.exists():
            raise FileExistsError("Evaluation output already exists; choose a new result path")
        result = evaluate(args.manifest, args.predictions, args.ground_truth)
        write_json(args.output, result)
        print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
