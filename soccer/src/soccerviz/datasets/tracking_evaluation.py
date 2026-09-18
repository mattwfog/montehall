"""Reviewed-frame pixel-space tracking evaluation via official TrackEval metrics."""

from __future__ import annotations

import importlib
import importlib.metadata
from collections import defaultdict
from contextlib import contextmanager
from threading import RLock

import numpy as np

from soccerviz.harness.engine import digest

TRACKEVAL_REQUIREMENT = (
    "trackeval @ git+https://github.com/JonathonLuiten/TrackEval.git@"
    "12c8791b303e0a0b50f753af204249e622d0281a"
)


# Pinned upstream uses removed np.float/np.int aliases. Scope compatibility to the two
# metric modules while holding a lock; never monkey-patch the shared NumPy module.
_METRIC_LOCK = RLock()


class _LegacyNumpy:
    float = float
    int = int

    def __getattr__(self, name):
        return getattr(np, name)


@contextmanager
def _metric_compatibility():
    modules = [
        importlib.import_module(f"trackeval.metrics.{name}") for name in ("hota", "identity")
    ]
    with _METRIC_LOCK:
        originals = [module.np for module in modules]
        try:
            for module in modules:
                module.np = _LegacyNumpy()
            yield
        finally:
            for module, original in zip(modules, originals, strict=True):
                module.np = original


def pixel_iou(left, right):
    left = np.asarray(left, dtype=float).reshape(-1, 4)
    right = np.asarray(right, dtype=float).reshape(-1, 4)
    for boxes in (left, right):
        if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("Bounding boxes must be finite with positive width and height")
    overlap = np.maximum(
        0,
        np.minimum(left[:, None, 2:], right[None, :, 2:])
        - np.maximum(left[:, None, :2], right[None, :, :2]),
    )
    intersection = overlap.prod(axis=2)
    union = (
        (left[:, 2:] - left[:, :2]).prod(axis=1)[:, None]
        + (right[:, 2:] - right[:, :2]).prod(axis=1)[None, :]
        - intersection
    )
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def sequence_data(ground_truth, predictions, frame_ids):
    """Adapt already scoped boxes, assigning consecutive IDs as TrackEval requires."""
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Duplicate evaluation frames")
    ids = set(frame_ids)
    grouped = []
    for rows in (ground_truth, predictions):
        groups, identities, seen = defaultdict(list), {}, set()
        for row in rows:
            if row["frame_id"] not in ids:
                raise ValueError("Box outside reviewed evaluation frames")
            identity = str(row["track_id"])
            pair = (row["frame_id"], identity)
            if pair in seen:
                raise ValueError("Duplicate identity in one frame")
            seen.add(pair)
            identities.setdefault(identity, len(identities))
            groups[row["frame_id"]].append((identities[identity], row["bbox"]))
        grouped.append((groups, identities))
    data = {
        "num_timesteps": len(frame_ids),
        "num_gt_dets": len(ground_truth),
        "num_tracker_dets": len(predictions),
        "num_gt_ids": len(grouped[0][1]),
        "num_tracker_ids": len(grouped[1][1]),
        "gt_ids": [],
        "tracker_ids": [],
        "similarity_scores": [],
    }
    for frame_id in frame_ids:
        gt, pred = grouped[0][0][frame_id], grouped[1][0][frame_id]
        data["gt_ids"].append(np.array([x[0] for x in gt], dtype=int))
        data["tracker_ids"].append(np.array([x[0] for x in pred], dtype=int))
        data["similarity_scores"].append(pixel_iou([x[1] for x in gt], [x[1] for x in pred]))
    return data


def evaluate_tracking(reviewed: dict, predictions: list[dict], *, source_sha256: str) -> dict:
    """Score only frame/label pairs explicitly declared exhaustive by a human reviewer.

    Prediction rows: frame_id, track_id, label, bbox=[x0,y0,x1,y1]. Scores are 0..1.
    """
    if reviewed.get("schema") != "reviewed-tracking/v1":
        raise ValueError("Expected reviewed tracking import")
    if source_sha256 != reviewed["source_sha256"]:
        raise ValueError("Prediction and ground-truth video hashes differ")
    attestation = reviewed["review"]
    if not all(attestation.get(k) for k in ("reviewer", "reason", "annotations_sha256")):
        raise ValueError("Missing human review provenance")
    coverage = defaultdict(set)
    for entry in attestation["reviewed_frames"]:
        if entry.get("exhaustive") is not True:
            raise ValueError("Evaluation requires exhaustive annotation")
        for label in entry["labels"]:
            coverage[label].add(entry["frame_id"])
    gt = reviewed["ground_truth"]
    for row in gt:
        if row.get("reviewed") is not True or row["frame_id"] not in coverage[row["label"]]:
            raise ValueError("Ground truth outside explicitly reviewed coverage")
    report = {
        "schema": "tracking-evaluation/v1",
        "source_sha256": source_sha256,
        "ground_truth_id": digest(reviewed),
        "predictions_id": digest(predictions),
        "status": "awaiting_human_review" if not coverage else "evaluated",
        "coordinate_space": "original video pixels",
        "metric_scale": "0..1",
        "identity_iou_threshold": 0.5,
        "classes": {},
        "limitations": [
            "Scores cover reviewed samples only; gaps are not interpolated",
            "No MOTChallenge distractor preprocessing; classes scored separately",
        ],
    }
    if not coverage:
        return report
    try:
        from trackeval.metrics import HOTA, Identity
    except ImportError as error:
        raise RuntimeError(
            f"Install optional official TrackEval: {TRACKEVAL_REQUIREMENT}"
        ) from error
    report["numpy_compatibility"] = "metric-module-only np.float=float and np.int=int aliases"
    report["trackeval_version"] = importlib.metadata.version("trackeval")
    distribution = importlib.metadata.distribution("trackeval")
    report["trackeval_direct_url"] = distribution.read_text("direct_url.json")
    for label, frames in sorted(coverage.items()):
        selected_gt = [r for r in gt if r["label"] == label]
        selected_pred = [r for r in predictions if r["label"] == label and r["frame_id"] in frames]
        data = sequence_data(selected_gt, selected_pred, sorted(frames))
        with _metric_compatibility():
            hota = HOTA().eval_sequence(data)
            identity = Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(data)
        report["classes"][label] = {
            "frames": len(frames),
            "ground_truth_boxes": len(selected_gt),
            "predicted_boxes": len(selected_pred),
            **{k: float(np.mean(hota[k])) for k in ("HOTA", "DetA", "AssA", "LocA")},
            **{k: float(identity[k]) for k in ("IDF1", "IDP", "IDR")},
            **{k: int(identity[k]) for k in ("IDTP", "IDFP", "IDFN")},
            "hota_by_iou": np.asarray(hota["HOTA"]).tolist(),
        }
    return report


def prediction_rows(evidence: dict) -> list[dict]:
    """Normalize the existing video tracklets without changing their detections."""
    return [
        {
            "frame_id": row["frame_id"],
            "track_id": f"{row.get('shot_id', 0)}:{row['tracklet_id']}",
            "label": row["role_hypothesis"],
            "bbox": [row[k] for k in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")],
        }
        for row in evidence["tables"]["detections"]
    ]
