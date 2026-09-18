"""Video annotation handoff, measurable detection metrics, and artifact lineage checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from soccerviz.core.assets import sha256
from soccerviz.core.data import write_json
from soccerviz.core.identity import box_iou


def detection_metrics(predictions, ground_truth, threshold=0.5):
    """Class-aware IoU matching on explicitly reviewed frames; not COCO mAP."""
    tp, fp, fn = 0, 0, 0
    matches = []
    reviewed = sorted(set(ground_truth["reviewed_frames"]))
    for frame in reviewed:
        predicted = [p for p in predictions if p["frame_id"] == frame]
        truth = [p for p in ground_truth["objects"] if p["frame_id"] == frame]
        used_p, used_t = set(), set()
        if predicted and truth:
            overlap = box_iou(
                np.array([p["bbox"] for p in predicted]), np.array([p["bbox"] for p in truth])
            )
            compatible = np.array([[p["class"] == t["class"] for t in truth] for p in predicted])
            score = np.where(compatible & (overlap >= threshold), overlap, -1e6)
            ii, jj = linear_sum_assignment(-score)
            for i, j in zip(ii, jj, strict=True):
                if score[i, j] >= threshold:
                    used_p.add(i)
                    used_t.add(j)
                    matches.append(
                        {
                            "frame_id": frame,
                            "predicted_id": predicted[i].get("tracklet_id"),
                            "true_id": truth[j].get("track_id"),
                            "iou": float(overlap[i, j]),
                        }
                    )
        tp += len(used_p)
        fp += len(predicted) - len(used_p)
        fn += len(truth) - len(used_t)
    return {
        "reviewed_frames": len(reviewed),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_at_iou_0_5": tp / (tp + fp) if tp + fp else None,
        "recall_at_iou_0_5": tp / (tp + fn) if tp + fn else None,
        "matches": matches,
        "metric": "single-threshold precision/recall, not mAP or HOTA",
    }


def export_annotation_task(video_run: Path):
    detections = pd.read_parquet(video_run / "detections.parquet")
    frames = []
    for path in sorted((video_run / "frames").glob("*.jpg")):
        if "annotated" in path.stem:
            continue
        frame = int(path.stem)
        subset = detections[detections.frame_id == frame]
        suggestions = [
            {
                "bbox": [r.bbox_x0, r.bbox_y0, r.bbox_x1, r.bbox_y1],
                "class": r.role_hypothesis,
                "tracklet_id": r.tracklet_id,
            }
            for r in subset.itertuples()
        ]
        frames.append(
            {
                "frame_id": frame,
                "image": f"frames/{path.name}",
                "status": "unreviewed",
                "model_suggestions": suggestions,
                "ground_truth": [],
            }
        )
    task = {
        "schema_version": "0.2",
        "source_sha256": json.loads((video_run / "report.json").read_text())["source_sha256"],
        "instructions": "Review images, correct/add/remove boxes and labels, then mark individual frames reviewed. Model suggestions are not ground truth.",
        "frames": frames,
    }
    path = video_run / "annotation-task.json"
    if not path.exists():
        write_json(path, task)
    return {"annotation_frames": len(frames), "ground_truth_status": "requires review"}


def evaluate_annotations(video_run: Path, annotations: Path):
    task = json.loads(annotations.read_text())
    source = json.loads((video_run / "report.json").read_text())["source_sha256"]
    if task.get("source_sha256") != source:
        raise ValueError("Annotations refer to a different source video")
    frames, objects = [], []
    for frame in task["frames"]:
        if frame["status"] == "reviewed":
            frames.append(frame["frame_id"])
            objects.extend(
                [{**obj, "frame_id": frame["frame_id"]} for obj in frame["ground_truth"]]
            )
    if not frames:
        raise ValueError(
            "No reviewed annotation frames; model suggestions cannot serve as ground truth"
        )
    predictions = pd.read_parquet(video_run / "detections.parquet")
    records = [
        {
            "frame_id": r.frame_id,
            "bbox": [r.bbox_x0, r.bbox_y0, r.bbox_x1, r.bbox_y1],
            "class": r.role_hypothesis,
            "tracklet_id": r.tracklet_id,
        }
        for r in predictions.itertuples()
    ]
    report = detection_metrics(records, {"reviewed_frames": frames, "objects": objects})
    write_json(video_run / "annotation-evaluation.json", report)
    return report


def audit_video_contract(video_run: Path):
    frames = pd.read_parquet(video_run / "frames.parquet")
    detections = pd.read_parquet(video_run / "detections.parquet")
    state = pd.read_parquet(video_run / "state.parquet")
    if not frames.timestamp_s.is_monotonic_increasing or frames.timestamp_s.duplicated().any():
        raise ValueError("Presentation timestamps must increase strictly")
    if detections.detection_id.duplicated().any():
        raise ValueError("Detection evidence IDs must be unique")
    known = set(frames.frame_id)
    if not set(detections.frame_id) <= known or not set(state.frame_id) <= known:
        raise ValueError("Orphan observations")
    projected = state.status.isin(["projected_observation", "projected_hypothesis"])
    if not np.isfinite(state.loc[projected, ["x_m", "y_m"]].to_numpy()).all():
        raise ValueError("A projected observation must have finite coordinates")
    if not state.loc[projected, "calibration_accepted"].all():
        raise ValueError("Rejected calibration cannot produce a pitch observation")
    primitive_columns = [
        "detection_id",
        "frame_id",
        "timestamp_s",
        "role_hypothesis",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "confidence",
        "source",
    ]
    # Track continuity, team classification, and downstream tactical scores live in separate tables.
    detections[primitive_columns].to_parquet(video_run / "primitive-evidence.parquet", index=False)
    manifest = {
        "schema_version": "0.2",
        "clock": "source PTS seconds",
        "primitive_rows": len(detections),
        "projected_rows": int(projected.sum()),
        "downstream_verdicts_in_primitive_stream": False,
        "files": {
            path.name: {"sha256": sha256(path), "rows": len(pd.read_parquet(path))}
            for path in sorted(video_run.glob("*.parquet"))
        },
        "unknowns": [
            "Named identity",
            "Physical-position uncertainty",
            "Ground-truth tracking accuracy",
        ],
    }
    write_json(video_run / "stage-manifest.json", manifest)
    return manifest
