"""Official SoccerNet GS-HOTA and MOT metric workers; no ground-truth prediction input."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from soccerviz.datasets.soccernet_adapter import (
    import_tracking,
    read_mot,
    sha256,
    validate_version,
    write_json,
)
from soccerviz.datasets.tracking_evaluation import _metric_compatibility, sequence_data

GSR_COMMIT = "9c25232f6f2b56c9f203f1eb55784ff1e97df683"


def metric_provenance():
    for name in ("trackeval", "sn-trackeval"):
        try:
            dist = importlib.metadata.distribution(name)
            return {
                "distribution": name,
                "version": dist.version,
                "direct_url": json.loads(dist.read_text("direct_url.json") or "{}"),
            }
        except importlib.metadata.PackageNotFoundError:
            pass
    raise RuntimeError("Official TrackEval evaluator is not installed")


def metric_summary(data):
    from trackeval.metrics import HOTA, Identity

    with _metric_compatibility():
        hota = HOTA().eval_sequence(data)
        identity = Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(data)
    return {
        **{k: float(np.mean(hota[k])) for k in ("HOTA", "DetA", "AssA", "LocA")},
        **{k: float(identity[k]) for k in ("IDF1", "IDP", "IDR")},
        "frames": data["num_timesteps"],
        "ground_truth_boxes": data["num_gt_dets"],
        "predicted_boxes": data["num_tracker_dets"],
        "hota_by_threshold": hota["HOTA"].tolist(),
    }


def evaluate_gsr(labels, predictions, split, sequence, frame_ids=None):
    """Use the official SoccerNet dataset preprocessing and similarity unchanged.

    Pixel metrics disable attributes; GS-HOTA uses all attributes and 5m tolerance.
    Missing pitch geometry means GS-HOTA is unavailable, never silently filtered.
    """
    from trackeval.datasets.soccernet_gs import SoccerNetGS

    distribution = importlib.metadata.distribution("sn-trackeval")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != GSR_COMMIT:
        raise RuntimeError("Install envs/soccernet/requirements.txt to use pinned official GS-HOTA")
    if split not in {"train", "valid", "test", "challenge"}:
        raise ValueError("Unknown official SoccerNet split")
    gt = json.loads(Path(labels).read_text())
    validate_version(gt)
    if gt["info"].get("name", sequence) != sequence:
        raise ValueError("Sequence name differs from source annotations")
    if (
        Path(labels).parent.parent.name in {"train", "valid", "test", "challenge"}
        and Path(labels).parent.parent.name != split
    ):
        raise ValueError("Requested split differs from source annotation path")
    pred_file = json.loads(Path(predictions).read_text())
    if not isinstance(pred_file, dict) or "predictions" not in pred_file:
        raise ValueError("Expected official SoccerNet predictions JSON")
    frames = {int(Path(im["file_name"]).stem): im for im in gt["images"]}
    selected = sorted(frames) if frame_ids is None else frame_ids
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(f not in frames for f in selected)
    ):
        raise ValueError("Evaluation frame IDs must be a nonempty unique subset of source frames")
    gt["images"] = [frames[f] for f in sorted(selected)]
    if not any(
        all(
            im.get(k, False)
            for k in ("has_labeled_pitch", "has_labeled_camera", "has_labeled_person")
        )
        for im in gt["images"]
    ):
        raise ValueError("No officially scored frames in requested sample")
    image_ids = {im["image_id"] for im in gt["images"]}
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in image_ids]
    pred = copy.deepcopy(pred_file)
    all_image_ids = {im["image_id"] for im in frames.values()}
    if any(a["image_id"] not in all_image_ids for a in pred["predictions"]):
        raise ValueError(
            "Prediction image ID not in original source; supply explicit frame mapping"
        )
    pred["predictions"] = [a for a in pred["predictions"] if a["image_id"] in image_ids]
    missing_pitch, seen = 0, set()
    for a in pred["predictions"]:
        key = (a["image_id"], a["track_id"])
        if key in seen:
            raise ValueError("Duplicate prediction identity in source frame")
        seen.add(key)
        box = a.get("bbox_image")
        if (
            not box
            or not all(math.isfinite(float(box[k])) for k in ("x", "y", "w", "h"))
            or box["w"] <= 0
            or box["h"] <= 0
        ):
            raise ValueError("Invalid prediction image box")
        if a.get("attributes", {}).get("role") != "ball":
            pitch = a.get("bbox_pitch")
            if not pitch or not all(
                k in pitch and pitch[k] is not None and math.isfinite(float(pitch[k]))
                for k in (
                    "x_bottom_left",
                    "y_bottom_left",
                    "x_bottom_middle",
                    "y_bottom_middle",
                    "x_bottom_right",
                    "y_bottom_right",
                )
            ):
                missing_pitch += 1
    result = {
        "schema": "soccernet-evaluation/v1",
        "kind": "gsr",
        "split": split,
        "sequence": sequence,
        "annotation_version": str(gt["info"]["version"]),
        "labels_sha256": sha256(labels),
        "predictions_sha256": sha256(predictions),
        "frame_ids": sorted(selected),
        "scored_frame_ids": [
            f
            for f in sorted(selected)
            if all(
                frames[f].get(k, False)
                for k in ("has_labeled_pitch", "has_labeled_camera", "has_labeled_person")
            )
        ],
        "prediction_provenance": pred_file.get("provenance"),
        "metric_scale": "0..1",
        "evaluator": {
            "distribution": "sn-trackeval",
            "version": distribution.version,
            "direct_url": direct_url,
        },
        "gs_hota": None,
        "missing_prediction_pitch": missing_pitch,
        "limitations": [
            "Subset score, not the full official split leaderboard.",
            "Ball excluded following the official GSR evaluator.",
            "No team or jersey labels are inferred from ground truth.",
        ],
    }
    with tempfile.TemporaryDirectory(prefix="soccernet-gs-") as temporary:
        root = Path(temporary)
        write_json(root / "gt" / split / sequence / "Labels-GameState.json", gt)
        tracker_dir = root / "trackers" / f"SoccerNetGS-{split}" / "candidate" / "data"
        write_json(tracker_dir / f"{sequence}.json", pred)
        base = {
            "GT_FOLDER": str(root / "gt"),
            "TRACKERS_FOLDER": str(root / "trackers"),
            "TRACKERS_TO_EVAL": ["candidate"],
            "SPLIT_TO_EVAL": split,
            "SEQ_INFO": {sequence: len(selected)},
            "PRINT_CONFIG": False,
        }
        for space in ["image", "pitch"]:
            if space == "pitch" and missing_pitch:
                result["gs_hota_unavailable_reason"] = (
                    "Predictions lack metric pitch geometry; image metrics remain valid. No prediction was silently discarded to improve GS-HOTA."
                )
                continue
            config = {
                **base,
                "EVAL_SPACE": space,
                "EVAL_SIMILARITY_METRIC": "iou" if space == "image" else "gaussian",
                "USE_ROLES": space == "pitch",
                "USE_TEAMS": space == "pitch",
                "USE_JERSEY_NUMBERS": space == "pitch",
                "EVAL_DIST_TOL": 5,
            }
            dataset = SoccerNetGS(config)
            raw = dataset.get_raw_seq_data("candidate", sequence)
            data = dataset.get_preprocessed_seq_data(raw, "person")
            summary = metric_summary(data)
            if space == "pitch":
                summary["GS-HOTA"] = summary.pop("HOTA")
                summary["distance_tolerance_m"] = 5
                summary["attributes"] = ["role", "team", "jersey"]
            result["image" if space == "image" else "gs_hota"] = summary
    return result


def track_public_detections(sequence_dir, split, out, max_age_s=0.8):
    """Association-only diagnostic: official det files may contain oracle boxes."""
    from soccerviz.core.identity import TrackletAssociator

    benchmark = import_tracking(sequence_dir, split)
    detection_path = Path(sequence_dir) / "det/det.txt"
    detections = read_mot(detection_path)
    grouped = defaultdict(list)
    for row in detections:
        grouped[row["frame_id"]].append(row)
    tracker, predictions = TrackletAssociator(max_age_s), []
    for frame in benchmark["frames"]:
        rows = grouped[frame["frame_id"]]
        boxes = np.array([r["bbox"] for r in rows]).reshape(-1, 4)
        ids = tracker.update(boxes, np.zeros(len(boxes), dtype=int), frame["timestamp_s"])
        predictions.extend(
            {"frame_id": frame["frame_id"], "track_id": int(identity), "bbox": row["bbox"]}
            for row, identity in zip(rows, ids, strict=True)
        )
    # Exact source box comparison makes the association-only nature machine-readable.
    signature = lambda rows: sorted((r["frame_id"], *r["bbox"]) for r in rows)
    oracle = signature(benchmark["ground_truth"]) == signature(detections)
    report = {
        "schema": "soccernet-predictions/v1",
        "kind": "tracking",
        "split": split,
        "sequence": benchmark["sequence"],
        "detector": "official provided detections",
        "tracker": "soccerviz.core.identity.TrackletAssociator",
        "max_age_s": max_age_s,
        "detections_sha256": sha256(detection_path),
        "oracle_boxes": oracle,
        "scope": "association_only; not video detection or held-out accuracy",
        "predictions": predictions,
    }
    write_json(out, report)
    return {
        "schema": "soccernet-association-run/v1",
        "predictions_path": str(Path(out).resolve()),
        "predictions_sha256": sha256(out),
        "oracle_boxes": oracle,
        "predictions": len(predictions),
        "tracklets": len({r["track_id"] for r in predictions}),
        "split": split,
        "sequence": benchmark["sequence"],
    }


def evaluate_mot(sequence_dir, split, predictions, frame_ids=None):
    benchmark = import_tracking(sequence_dir, split)
    if not benchmark["labels_available"]:
        raise ValueError("Ground-truth labels unavailable; cannot evaluate challenge data locally")
    pred = json.loads(Path(predictions).read_text())
    if pred.get("sequence") != benchmark["sequence"] or pred.get("split") != split:
        raise ValueError("Prediction sequence/split provenance mismatch")
    selected = [f["frame_id"] for f in benchmark["frames"]] if frame_ids is None else frame_ids
    all_frames = {f["frame_id"] for f in benchmark["frames"]}
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= all_frames:
        raise ValueError("Invalid evaluation source frames")
    rows = pred["predictions"]
    if any(r["frame_id"] not in all_frames for r in rows):
        raise ValueError("Prediction frame outside source sequence")
    data = sequence_data(
        [r for r in benchmark["ground_truth"] if r["frame_id"] in selected],
        [r for r in rows if r["frame_id"] in selected],
        selected,
    )
    return {
        "schema": "soccernet-evaluation/v1",
        "kind": "tracking",
        "split": split,
        "sequence": benchmark["sequence"],
        "source_files": benchmark["sources"],
        "evaluator": metric_provenance(),
        "predictions_sha256": sha256(predictions),
        "frame_ids": selected,
        "oracle_boxes_declared": pred.get("oracle_boxes"),
        "oracle_boxes": sorted(
            (r["frame_id"], *r["bbox"]) for r in rows if r["frame_id"] in selected
        )
        == sorted(
            (r["frame_id"], *r["bbox"])
            for r in benchmark["ground_truth"]
            if r["frame_id"] in selected
        ),
        "metric_scale": "0..1",
        "metrics": metric_summary(data),
        "limitations": [
            "Official provided boxes association diagnostic if oracle_boxes=true; not detector accuracy.",
            "All provided SoccerNet tracking objects scored as one class, no additional MOT distractor removal.",
        ],
    }


def run(request):
    if request.get("schema_version", 1) != 1:
        raise ValueError("Unsupported request schema_version")
    operation = request["operation"]
    function = {
        "gsr": evaluate_gsr,
        "tracking": evaluate_mot,
        "track-public-detections": track_public_detections,
    }.get(operation)
    if not function:
        raise ValueError(f"Unknown evaluation operation: {operation}")
    return function(
        **{k: v for k, v in request.items() if k not in {"operation", "schema_version"}}
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args()
    write_json(args.response, run(json.loads(args.request.read_text())))


if __name__ == "__main__":
    main()
