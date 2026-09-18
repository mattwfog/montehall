"""Replay evaluated detector boxes through one unchanged anonymous tracker."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from soccerviz.candidates.vision_benchmark import (
    _ignored,
    _match,
    canonical_label,
    checked_box,
    frame_key,
)
from soccerviz.core.assets import sha256
from soccerviz.core.identity import TrackletAssociator
from soccerviz.datasets.tracking_evaluation import _metric_compatibility, sequence_data


def ignored_prediction_indices(rows, target):
    """Keep valid matches before applying annotation ignore regions to unmatched boxes."""
    truth = [row for row in target["detections"] if canonical_label(row["label"]) == "person"]
    _, unmatched, _ = _match(rows, truth, 0.5)
    return {i for i in unmatched if _ignored(rows[i]["bbox_xyxy"], target, 0.5)}


def run(manifest_path, predictions_path, detection_report_path, out):
    from trackeval.metrics import HOTA, Identity

    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads(manifest_path.read_text())
    predictions = json.loads(predictions_path.read_text())
    detection_report = json.loads(detection_report_path.read_text())
    if (
        predictions["manifest_sha256"] != sha256(manifest_path)
        or detection_report["manifest_sha256"] != sha256(manifest_path)
        or detection_report["predictions_sha256"] != sha256(predictions_path)
        or not detection_report["image_hashes_verified"]
    ):
        raise ValueError("Tracking requires a verified matching detection evaluation")
    ground_path = manifest_path.parent / "ground-truth.json"
    if detection_report["ground_truth_sha256"] != sha256(ground_path):
        raise ValueError("Ground truth changed since detector evaluation")
    ground = json.loads(ground_path.read_text())
    targets = {frame_key(row): row for row in ground["frames"]}
    source_annotations = {}
    for source in ground["sources"]:
        if sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError("Original annotations changed")
        data = json.loads(Path(source["path"]).read_text())
        for annotation in data["annotations"]:
            key = (source["sequence"], str(annotation["id"]))
            if key in source_annotations:
                raise ValueError("Duplicate original annotation identity within a sequence")
            source_annotations[key] = annotation
    prediction_lookup = {frame_key(row): row for row in predictions["frames"]}
    threshold = detection_report["thresholds"]["score_threshold"]
    hota_metric, identity_metric = HOTA(), Identity({"PRINT_CONFIG": False})
    hota_results, identity_results, counts = {}, {}, {}
    tracked = []
    for sequence in sorted({row["sequence"] for row in manifest["frames"]}):
        frames = sorted(
            [row for row in manifest["frames"] if row["sequence"] == sequence],
            key=lambda row: row["frame_id"],
        )
        tracker, truth_rows, prediction_rows = TrackletAssociator(), [], []
        for frame in frames:
            key = frame_key(frame)
            target = targets[key]
            rows = [
                row
                for row in prediction_lookup.get(key, {}).get("detections", [])
                if canonical_label(row["label"]) == "person" and row["score"] >= threshold
            ]
            # The tracker sees detections only; evaluator ignore regions are applied afterward.
            boxes = [checked_box(row["bbox_xyxy"]) for row in rows]
            identities = tracker.update(
                boxes, np.zeros(len(boxes), dtype=int), frame["timestamp_s"]
            )
            ignored = ignored_prediction_indices(rows, target)
            for index, (row, identity) in enumerate(zip(rows, identities, strict=True)):
                if index in ignored:
                    continue
                item = {
                    "frame_id": frame["frame_id"],
                    "track_id": int(identity),
                    "bbox": row["bbox_xyxy"],
                }
                prediction_rows.append(item)
                tracked.append({"sequence": sequence, **item})
            for row in target["detections"]:
                if row["label"] == "person":
                    annotation = source_annotations[(sequence, str(row["source_annotation_id"]))]
                    truth_rows.append(
                        {
                            "frame_id": frame["frame_id"],
                            "track_id": str(annotation["track_id"]),
                            "bbox": row["bbox_xyxy"],
                        }
                    )
        data = sequence_data(truth_rows, prediction_rows, [row["frame_id"] for row in frames])
        with _metric_compatibility():
            hota_results[sequence] = hota_metric.eval_sequence(data)
            identity_results[sequence] = identity_metric.eval_sequence(data)
        counts[sequence] = {
            "frames": len(frames),
            "ground_truth_boxes": len(truth_rows),
            "predicted_boxes": len(prediction_rows),
            "predicted_tracks": len({row["track_id"] for row in prediction_rows}),
        }

    def summary(hota, identity):
        return {
            **{key: float(np.mean(hota[key])) for key in ("HOTA", "DetA", "AssA", "LocA")},
            **{key: float(identity[key]) for key in ("IDF1", "IDP", "IDR")},
        }

    with _metric_compatibility():
        overall = summary(
            hota_metric.combine_sequences(hota_results),
            identity_metric.combine_sequences(identity_results),
        )
    report = {
        "schema": "detector-tracking-comparison/v1",
        "manifest_sha256": sha256(manifest_path),
        "predictions_sha256": sha256(predictions_path),
        "detection_report_sha256": sha256(detection_report_path),
        "tracker": "Existing TrackletAssociator, max_age_s=0.8, person class only, no appearance",
        "tracker_code_sha256": sha256(Path(__file__).parents[2] / "src/soccerviz/core/identity.py"),
        "comparison_code_sha256": sha256(Path(__file__)),
        "score_threshold": threshold,
        "trackeval_version": importlib.metadata.version("trackeval"),
        "trackeval_source": json.loads(
            importlib.metadata.distribution("trackeval").read_text("direct_url.json") or "{}"
        ),
        "overall": overall,
        "by_sequence": {
            key: {**counts[key], **summary(hota_results[key], identity_results[key])}
            for key in counts
        },
        "limitations": [
            "Three short validation sequences; not the full official benchmark.",
            "Person roles collapsed equally for both detectors; no team or jersey inference.",
            "Appearance disabled equally; not the original complete video pipeline.",
            "Tracker resets at each source sequence; no cross-camera identity claim.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tracks_path = out.with_name(out.stem + "-tracks.json")
    if tracks_path.exists():
        raise FileExistsError(tracks_path)
    tracks_path.write_text(json.dumps(tracked, indent=2) + "\n")
    report["tracks_sha256"] = sha256(tracks_path)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--detection-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.predictions, args.detection_report, args.out)
