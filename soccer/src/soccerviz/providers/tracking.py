"""Replay frozen detections through TrackLab's packaged OC-SORT implementation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from soccerviz.core.assets import sha256
from soccerviz.core.data import write_json
from soccerviz.core.identity import TrackletAssociator
from soccerviz.datasets.tracking_evaluation import prediction_rows

BOX = ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]


def validate_inputs(frames: pd.DataFrame, detections: pd.DataFrame) -> None:
    required = {"frame_id", "timestamp_s", "shot_id"}
    if not required.issubset(frames) or not (
        required | set(BOX) | {"detection_id", "confidence", "role_hypothesis"}
    ).issubset(detections):
        raise ValueError("Frozen frame/detection schema is incomplete")
    if frames.empty or frames.frame_id.duplicated().any():
        raise ValueError("Frames must be nonempty with unique IDs")
    if not np.isfinite(frames.timestamp_s).all() or not (np.diff(frames.timestamp_s) > 0).all():
        raise ValueError("Source timestamps must be finite and strictly increasing")
    if detections.detection_id.duplicated().any():
        raise ValueError("Detection IDs must be unique")
    clock = frames.set_index("frame_id")
    if not detections.frame_id.isin(clock.index).all():
        raise ValueError("Detection references an unknown frame")
    for column in ["timestamp_s", "shot_id"]:
        if not np.array_equal(detections[column], clock.loc[detections.frame_id, column]):
            raise ValueError(f"Detection {column} disagrees with frame clock")
    values = detections[BOX + ["confidence"]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, 2:4] <= values[:, :2]).any():
        raise ValueError("Detection boxes/confidences must be finite with positive dimensions")
    if ((values[:, 4] <= 0) | (values[:, 4] > 1)).any():
        raise ValueError("Confidence must be in (0, 1]")


def track_frozen(
    frames: pd.DataFrame,
    detections: pd.DataFrame,
    backend: str = "tracklab-ocsort",
    max_age_s: float = 0.8,
) -> pd.DataFrame:
    """Return every input detection with new anonymous IDs; never synthesize boxes.

    OC-SORT uses a unit-step motion model. Reject irregular sampling inside a shot
    instead of pretending source frame ordinals are elapsed time. Missing-detection
    frames still advance its state. Each role and shot has independent association.
    """
    validate_inputs(frames, detections)
    if backend not in {"tracklab-ocsort", "soccerviz"}:
        raise ValueError(f"Unknown backend: {backend}")
    if max_age_s <= 0:
        raise ValueError("max_age_s must be positive")
    if backend == "tracklab-ocsort":
        try:
            import torch
            from oc_sort.ocsort import OCSort
        except ImportError as exc:
            raise RuntimeError(
                "Install the isolated TrackLab worker dependencies; see docs/guides/data-tracking-adapters.md"
            ) from exc
    result = detections.copy().reset_index(drop=True)
    result["original_tracklet_id"] = result.get(
        "tracklet_id", pd.Series(index=result.index, dtype=float)
    )
    result["tracklet_id"] = -1
    result["tracker_backend"] = backend
    result["identity_status"] = "anonymous_tracklet"
    # Existing team votes were accumulated along the original tracker. Reusing
    # those labels after changing association would silently leak old identity.
    for column in ["team_cluster", "team_color_margin", "player_identity", "jersey_hypothesis"]:
        if column in result:
            result[column] = None
    next_id = 1
    for shot, shot_frames in frames.groupby("shot_id", sort=False):
        steps = np.diff(shot_frames.timestamp_s.to_numpy(float))
        dt = float(np.median(steps)) if len(steps) else 0.2
        if (
            backend == "tracklab-ocsort"
            and len(steps)
            and not np.allclose(steps, dt, atol=1e-6, rtol=1e-4)
        ):
            raise ValueError(
                "OC-SORT requires regular sampling within each shot; resample explicitly first"
            )
        subset = result[result.shot_id == shot]
        for role in sorted(subset.role_hypothesis.unique()):
            tracker = (
                OCSort(
                    det_thresh=0.0,
                    max_age=max(1, round(max_age_s / dt)),
                    min_hits=0,
                    iou_threshold=0.3,
                    delta_t=1,
                )
                if backend == "tracklab-ocsort"
                else TrackletAssociator(max_age_s)
            )
            mapping = {}
            for frame in shot_frames.itertuples():
                selected = result[
                    (result.frame_id == frame.frame_id) & (result.role_hypothesis == role)
                ]
                boxes = selected[BOX].to_numpy(float)
                if backend == "tracklab-ocsort":
                    tensor = torch.as_tensor(
                        np.column_stack(
                            [boxes, selected.confidence, np.zeros(len(selected)), selected.index]
                        ),
                        dtype=torch.float64,
                    )
                    tracked = tracker.update(tensor, None)
                    pairs = [(int(row[7]), int(row[4])) for row in tracked]
                else:
                    ids = tracker.update(boxes, np.zeros(len(boxes), int), frame.timestamp_s)
                    pairs = list(zip(selected.index, ids, strict=True))
                for index, local_id in pairs:
                    if index not in selected.index:
                        raise ValueError("Upstream tracker returned an unknown detection index")
                    if local_id not in mapping:
                        mapping[local_id] = next_id
                        next_id += 1
                    result.at[index, "tracklet_id"] = mapping[local_id]
    if (result.tracklet_id < 0).any():
        raise ValueError(
            "Tracker omitted input detections; refusing an unequal-detection comparison"
        )
    return result


def diagnostics(detections: pd.DataFrame) -> dict:
    lengths = detections.groupby("tracklet_id").size()
    return {
        "detections": len(detections),
        "tracklets": len(lengths),
        "singleton_tracklets": int((lengths == 1).sum()),
        "median_detections_per_tracklet": float(lengths.median()) if len(lengths) else None,
        "ground_truth_available": False,
        "hota": None,
        "idf1": None,
        "interpretation": "Unlabeled continuity diagnostics; fewer tracks does not establish better accuracy",
    }


def compare_trackers(video_dir: Path, out: Path, max_age_s: float = 0.8) -> dict:
    """Write two harness-importable video artifact directories and comparison.json."""
    video_dir, out = Path(video_dir), Path(out)
    if out.exists():
        raise ValueError(f"Output already exists: {out}")
    frames = pd.read_parquet(video_dir / "frames.parquet")
    original = pd.read_parquet(video_dir / "detections.parquet")
    validate_inputs(frames, original)
    summary = {
        "schema_version": 1,
        "source_dir": str(video_dir.resolve()),
        "frozen_detection_sha256": sha256(video_dir / "detections.parquet"),
        "frame_sha256": sha256(video_dir / "frames.parquet"),
        "max_age_s": max_age_s,
        "comparison": {},
        "limitations": [
            "No reviewed identity labels; no accuracy ranking",
            "Frozen artifacts do not contain shirt-color vectors, so both trackers run without appearance",
            "Existing track-voted team hypotheses cleared; recompute team evidence before tactics",
        ],
    }
    source_report = json.loads((video_dir / "report.json").read_text())
    if not source_report.get("source_sha256"):
        raise ValueError("Source report must identify the video SHA-256")
    # Compute both before writing to avoid half-complete success artifacts.
    candidates = {
        backend: track_frozen(frames, original, backend, max_age_s)
        for backend in ["soccerviz", "tracklab-ocsort"]
    }
    for backend, table in candidates.items():
        target = out / backend
        target.mkdir(parents=True)
        for path in video_dir.glob("*.parquet"):
            if path.name in {
                "frames.parquet",
                "landmarks.parquet",
                "calibration.parquet",
                "ball_candidates.parquet",
            }:
                shutil.copyfile(path, target / path.name)
        table.to_parquet(target / "detections.parquet", index=False)
        state = pd.read_parquet(video_dir / "state.parquet")
        ids = table.set_index("detection_id").tracklet_id
        player_mask = state.evidence_id.isin(ids.index)
        state.loc[player_mask, "tracklet_id"] = state.loc[player_mask, "evidence_id"].map(ids)
        state.loc[player_mask, "team_cluster"] = None
        state.to_parquet(target / "state.parquet", index=False)
        report = dict(source_report)
        report["anonymous_tracklets"] = int(table.tracklet_id.nunique())
        report["team_assignment_coverage"] = 0.0
        report["tracking_adapter"] = {
            "backend": backend,
            "max_age_s": max_age_s,
            "source_report_sha256": sha256(video_dir / "report.json"),
            "frozen_detection_sha256": summary["frozen_detection_sha256"],
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    ["tracklab", "torch", "numpy", "filterpy"]
                    if backend == "tracklab-ocsort"
                    else ["numpy", "scipy"]
                )
            },
        }
        prediction_path = target / "predictions.json"
        write_json(
            prediction_path,
            {
                "schema_version": 1,
                "source_sha256": source_report["source_sha256"],
                "frozen_detection_sha256": summary["frozen_detection_sha256"],
                "backend": backend,
                "predictions": prediction_rows(
                    {"tables": {"detections": table.to_dict("records")}}
                ),
            },
        )
        report["tracking_adapter"]["predictions_path"] = str(prediction_path.resolve())
        report["tracking_adapter"]["predictions_sha256"] = sha256(prediction_path)
        report["accuracy"] = {key: None for key in source_report.get("accuracy", {})}
        write_json(target / "report.json", report)
        summary["comparison"][backend] = {
            **diagnostics(table),
            "predictions_path": str(prediction_path.resolve()),
            "predictions_sha256": sha256(prediction_path),
        }
    write_json(out / "comparison.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if request.get("schema_version", 1) != 1:
        raise ValueError("Unsupported worker request schema_version")
    report = compare_trackers(
        Path(request["video_dir"]), Path(request["out"]), request.get("max_age_s", 0.8)
    )
    write_json(args.response, report)


if __name__ == "__main__":
    main()
