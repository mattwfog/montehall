"""Map actual video-model outputs to SoccerNet coordinates without reading target labels."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

from soccerviz.core.assets import sha256
from soccerviz.core.data import write_json
from soccerviz.core.geometry import project


def prepare_video(sequence, out, frame_numbers, source_fps=25):
    """Encode explicit uniformly sampled source images losslessly; retain the exact frame map."""
    sequence, out = Path(sequence), Path(out)
    if out.exists():
        raise ValueError("Choose a new sample directory")
    numbers = list(frame_numbers)
    steps = np.diff(numbers)
    if len(numbers) < 2 or not (steps > 0).all() or not (steps == steps[0]).all():
        raise ValueError("At least two increasing uniformly spaced source frames are required")
    labels = json.loads((sequence / "Labels-GameState.json").read_text())
    # Only public image metadata is accessed. Object annotations are never used.
    metadata = {int(Path(row["file_name"]).stem): row for row in labels["images"]}
    fps = Fraction(source_fps, int(steps[0]))
    files = []
    for ordinal, number in enumerate(numbers):
        info = metadata[number]
        file = sequence / "img1" / info["file_name"]
        if not file.is_file():
            raise ValueError(f"Source image missing: {file}")
        files.append(
            {
                "video_frame": ordinal,
                "source_frame": number,
                "image_id": str(info["image_id"]),
                "timestamp_s": (number - numbers[0]) / source_fps,
                "source_timestamp_s": (number - 1) / source_fps,
                "width": info["width"],
                "height": info["height"],
                "path": str(file.resolve()),
                "sha256": sha256(file),
            }
        )
    out.mkdir(parents=True)
    video = out / "source.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264rgb", rate=fps)
        stream.width, stream.height = files[0]["width"], files[0]["height"]
        stream.pix_fmt = "rgb24"
        stream.options = {"crf": "0", "preset": "fast"}
        for row in files:
            pixels = cv2.imread(row["path"])
            if pixels is None or pixels.shape[:2] != (stream.height, stream.width):
                raise ValueError("Source image dimensions disagree")
            frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    report = {
        "schema": "soccernet-sampled-video/v1",
        "sequence": sequence.name,
        "source_metadata_sha256": sha256(sequence / "Labels-GameState.json"),
        "video": str(video.resolve()),
        "source_sha256": sha256(video),
        "source_fps": source_fps,
        "sample_fps": float(fps),
        "frames": files,
        "policy": "Lossless encoding of selected source images; no interpolation or annotation overlays",
    }
    write_json(out / "frame-map.json", report)
    return report


def export_predictions(video_run, frame_map, out):
    video_run, frame_map, out = Path(video_run), Path(frame_map), Path(out)
    if out.exists():
        raise ValueError("Choose a new prediction file")
    mapping = json.loads(frame_map.read_text())
    report = json.loads((video_run / "report.json").read_text())
    if report["source_sha256"] != mapping["source_sha256"]:
        raise ValueError("Inference source does not match sampled video")
    frames = pd.read_parquet(video_run / "frames.parquet")
    detections = pd.read_parquet(video_run / "detections.parquet")
    state = pd.read_parquet(video_run / "state.parquet")
    calibration = pd.read_parquet(video_run / "calibration.parquet")
    if frames.frame_id.duplicated().any() or frames.source_frame.duplicated().any():
        raise ValueError("Duplicate inference frame identity")
    if state.evidence_id.duplicated().any() or calibration.frame_id.duplicated().any():
        raise ValueError("Duplicate state evidence or calibration frame")
    for key in ("video_frame", "image_id", "source_frame"):
        values = [row[key] for row in mapping["frames"]]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate frame-map {key}")
    calibration_lookup = {row["frame_id"]: row for row in calibration.to_dict("records")}
    source_frames = {row["video_frame"]: row for row in mapping["frames"]}
    frame_lookup = {}
    for row in frames.to_dict("records"):
        source = source_frames[int(row["source_frame"])]
        if abs(float(row["timestamp_s"]) - source["timestamp_s"]) > 0.001:
            raise ValueError("Sampled video timestamps differ from source-frame map")
        if (int(row["width"]), int(row["height"])) != (source["width"], source["height"]):
            raise ValueError("Inference pixel dimensions changed")
        frame_lookup[row["frame_id"]] = source
    state_lookup = {row["evidence_id"]: row for row in state.to_dict("records")}
    categories = {"player": 1, "goalkeeper": 2, "referee": 3, "other": 7}
    identities, predictions = {}, []
    for row in detections.to_dict("records"):
        source = frame_lookup[row["frame_id"]]
        role = row["role_hypothesis"]
        if role not in categories:
            raise ValueError(f"Unsupported GSR object role: {role}")
        identity = (int(row.get("shot_id", 0)), int(row["tracklet_id"]))
        identities.setdefault(identity, len(identities) + 1)
        point = state_lookup.get(row["detection_id"], {})
        x, y = point.get("x_m"), point.get("y_m")
        pitch = None
        if (
            x is not None
            and y is not None
            and np.isfinite([x, y]).all()
            and point.get("calibration_accepted")
        ):
            matrix = calibration_lookup[row["frame_id"]].get("homography")
            if matrix is not None:
                bottom = project(
                    np.array([[row["bbox_x0"], row["bbox_y1"]], [row["bbox_x1"], row["bbox_y1"]]]),
                    np.asarray(matrix).reshape(3, 3),
                )
                if np.isfinite(bottom).all():
                    pitch = {
                        "x_bottom_middle": float(x) - 52.5,
                        "y_bottom_middle": float(y) - 34,
                        "x_bottom_left": float(bottom[0, 0]) - 52.5,
                        "y_bottom_left": float(bottom[0, 1]) - 34,
                        "x_bottom_right": float(bottom[1, 0]) - 52.5,
                        "y_bottom_right": float(bottom[1, 1]) - 34,
                    }
        predictions.append(
            {
                "image_id": source["image_id"],
                "track_id": identities[identity],
                "supercategory": "object",
                "category_id": categories[role],
                "bbox_image": {
                    "x": row["bbox_x0"],
                    "y": row["bbox_y0"],
                    "w": row["bbox_x1"] - row["bbox_x0"],
                    "h": row["bbox_y1"] - row["bbox_y0"],
                },
                "bbox_pitch": pitch,
                "attributes": {"role": role, "team": None, "jersey": None},
            }
        )
    result = {
        "predictions": predictions,
        "provenance": {
            "schema": "soccernet-model-predictions/v1",
            "sequence": mapping["sequence"],
            "sample_video_sha256": mapping["source_sha256"],
            "frame_map_sha256": sha256(frame_map),
            "inference_report_sha256": sha256(video_run / "report.json"),
            "detections_sha256": sha256(video_run / "detections.parquet"),
            "frames_sha256": sha256(video_run / "frames.parquet"),
            "state_sha256": sha256(video_run / "state.parquet"),
            "calibration_sha256": sha256(video_run / "calibration.parquet"),
            "frame_ids": [row["source_frame"] for row in frame_lookup.values()],
            "input_kind": "actual_model_predictions",
            "model_sha256": report.get("model_sha256"),
            "limitations": [
                "Anonymous team clusters have no verified SoccerNet left/right mapping; team is null",
                "No jersey classifier output; jersey is null",
                "Only calibration-accepted 105x68 projected footpoints enter pitch metrics",
                "Detector pretraining overlap with SoccerNet is unknown; this is a development benchmark",
            ],
        },
    }
    write_json(out, result)
    return {
        "predictions": len(predictions),
        "frames": len(frame_lookup),
        "path": str(out.resolve()),
        "sha256": sha256(out),
        "provenance": result["provenance"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--sequence", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--frames", type=int, default=50)
    prepare.add_argument("--stride", type=int, default=5)
    export = sub.add_parser("export")
    export.add_argument("--video-run", type=Path, required=True)
    export.add_argument("--frame-map", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_video(
            args.sequence, args.out, range(1, 1 + args.frames * args.stride, args.stride)
        )
    else:
        result = export_predictions(args.video_run, args.frame_map, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
