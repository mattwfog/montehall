"""Selectable video perception → guarded pitch replay → reviewable shared state."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from soccerviz.core.assets import sha256
from soccerviz.core.data import write_json
from soccerviz.core.geometry import project
from soccerviz.core.identity import TrackletAssociator
from soccerviz.vision.checkpoint_licence import licence_report, require_shipping
from soccerviz.vision.pipeline import cut_distance, shirt_color, video_frames
from soccerviz.vision.video_safety import BallTrack, CameraGuard, SafetyConfig, jersey_consensus

MODEL_ROOT = Path("/opt/soccerviz-models")
UPSTREAM_ROOT = Path("/opt/soccerviz-upstreams")
PRESETS = {
    "rf-soccer": ("rf", "rf-soccer.pth", "soccer", 576),
    "rf-general": ("rf", "rf-detr-medium.pth", "coco", 960),
    "yolo26-soccer": ("yolo", "yolo26-soccer.pt", "soccer", 576),
    "yolo26-general": ("yolo", "yolo26m.pt", "coco", 640),
    "legacy-soccer": ("yolo", "football-player-detection.pt", "soccer", 1280),
}
CALIBRATION_PROTOCOL = {
    "pnl": {"keypoint_threshold": 0.3434, "line_threshold": 0.7867, "refine_lines": True},
    "yolo": {"imgsz": 640, "detection_confidence": 0.25, "landmark_confidence": 0.5},
    "heatmap": {"input": 960, "landmark_confidence": 0.5},
}
CALIBRATIONS = tuple(CALIBRATION_PROTOCOL)


def make_models(preset, calibration, model_root, upstream_root, device):
    from soccerviz.candidates.calibration_model import PnLCalibrationModel, YoloCalibrationModel
    from soccerviz.candidates.detector_backends import RFDETRDetector, UltralyticsDetector
    from soccerviz.candidates.pitch_keypoints import HeatmapCalibrationModel

    kind, filename, mode, resolution = PRESETS[preset]
    if kind == "rf":
        detector = RFDETRDetector(
            checkpoint=model_root / filename,
            mode=mode,
            device=device,
            resolution=resolution,
            threshold=0.25,
        )
    else:
        detector = UltralyticsDetector(
            model_root / filename,
            mode=mode,
            device=device,
            resolution=resolution,
            threshold=0.25,
            ball_checkpoint=model_root / "football-ball-detection.pt"
            if preset == "legacy-soccer"
            else None,
            tiled_ball=preset == "legacy-soccer",
        )
    if calibration == "pnl":
        camera = PnLCalibrationModel(
            upstream_root / "pnlcalib",
            model_root / "pnl-keypoints.pt",
            model_root / "pnl-lines.pt",
            CALIBRATION_PROTOCOL,
            device,
        )
    elif calibration == "heatmap":
        camera = HeatmapCalibrationModel(
            model_root / "pitch-heatmap.pt", CALIBRATION_PROTOCOL, device
        )
    else:
        camera = YoloCalibrationModel(
            model_root / "football-pitch-detection.pt", CALIBRATION_PROTOCOL, device
        )
    return detector, camera


def render_frame(image, detections, state, frame):
    """Faithful overlays use source-pixel boxes and a separate guarded pitch map."""
    left = cv2.resize(image, (960, 540))
    sx, sy = 960 / image.shape[1], 540 / image.shape[0]
    if frame.get("ball_center_xy") is not None:
        x, y = frame["ball_center_xy"]
        point = (int(x * sx), int(y * sy))
        color = (0, 230, 255) if frame["ball_status"] == "detected" else (170, 170, 170)
        cv2.circle(left, point, 8, color, 2 if frame["ball_status"] == "detected" else 1)
        cv2.putText(left, frame["ball_status"], (point[0] + 10, point[1]), 0, 0.4, color, 1)
    right = np.full((540, 600, 3), (34, 54, 34), np.uint8)
    cv2.rectangle(right, (30, 65), (570, 415), (160, 185, 160), 2)
    cv2.line(right, (300, 65), (300, 415), (160, 185, 160), 1)
    colors = {0: (230, 150, 40), 1: (65, 105, 240)}
    for row in detections:
        color = colors.get(row.get("team_cluster"), (180, 180, 180))
        a = (int(row["bbox_x0"] * sx), int(row["bbox_y0"] * sy))
        b = (int(row["bbox_x1"] * sx), int(row["bbox_y1"] * sy))
        cv2.rectangle(left, a, b, color, 1)
        number = row.get("jersey_hypothesis")
        label = f"T{row['tracklet_id']}" + (f" #{int(number)}?" if number is not None else "")
        cv2.putText(left, label, (a[0], max(14, a[1] - 3)), 0, 0.4, color, 1)
    for row in state:
        if row.get("x_m") is None:
            continue
        ball = row["entity"] == "ball_candidate"
        color = (0, 230, 255) if ball else colors.get(row.get("team_cluster"), (180, 180, 180))
        p = (round(30 + row["x_m"] / 105 * 540), round(65 + row["y_m"] / 68 * 350))
        cv2.circle(
            right,
            p,
            5 if ball else 4,
            color,
            1 if row["status"] in {"estimated", "tentative"} else -1,
        )
    cv2.putText(
        right,
        f"{frame['timestamp_s']:.2f}s | shot {frame['shot_id']}",
        (25, 30),
        0,
        0.65,
        (235, 235, 235),
        1,
    )
    for index, line in enumerate(
        [
            "Camera: " + ("usable hypothesis" if frame["calibration_accepted"] else "REJECTED"),
            "Ball: " + frame["ball_status"],
            "Blue/red: uniform clusters | gray: unknown",
            "Hollow ball: uncertain | #?: jersey hypothesis",
            "Positions and identity remain unverified.",
        ]
    ):
        cv2.putText(right, line, (20, 445 + 19 * index), 0, 0.43, (230, 230, 230), 1)
    return np.concatenate([left, right], axis=1)


def run_workflow(
    source,
    out,
    *,
    preset="rf-soccer",
    calibration="pnl",
    seconds=20.0,
    hz=5.0,
    jersey=True,
    model_root=MODEL_ROOT,
    upstream_root=UPSTREAM_ROOT,
    device="cuda:0",
    detector=None,
    camera=None,
    shipping=False,
):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError("Select a new workflow output directory")
    if not source.is_file() or not 0 < seconds <= 600 or not 0 < hz <= 25:
        raise ValueError("Provide a video, 0<seconds<=600 and 0<hz<=25")
    if preset not in PRESETS or calibration not in CALIBRATIONS:
        raise ValueError("Unknown detector or camera preset")
    detector, camera = (
        (detector, camera)
        if detector is not None and camera is not None
        else make_models(preset, calibration, Path(model_root), Path(upstream_root), device)
    )
    licence = licence_report(model_hashes(detector, camera))
    if shipping:
        require_shipping(licence)
    out.mkdir(parents=True)
    (out / "frames").mkdir()
    (out / "jersey/crops").mkdir(parents=True)
    safety = SafetyConfig()
    guard, ball_tracker, tracker = CameraGuard(safety), BallTrack(safety), TrackletAssociator()
    tables = {
        name: []
        for name in (
            "frames",
            "detections",
            "state",
            "calibration",
            "ball_candidates",
            "ball_tracks",
        )
    }
    started = time.perf_counter()
    source_hash = sha256(source)
    last_hist, previous_time, shot = None, None, 0
    colors, team_model, team_votes, previous_centers, crop_times, crops = [], None, {}, {}, {}, []
    write_json(
        out / "configuration.json",
        {
            "schema": "video-workflow-config/v1",
            "preset": preset,
            "calibration": calibration,
            "seconds": seconds,
            "sample_hz": hz,
            "jersey": jersey,
            "shipping_required": shipping,
            "licence": licence,
            "source_sha256": source_hash,
            "safety": asdict(safety),
            "detector": detector.metadata,
            "camera": camera.metadata,
        },
    )
    for ordinal, timestamp, pts, time_base, image in video_frames(source, hz, seconds):
        fid = len(tables["frames"])
        cut_score, last_hist = cut_distance(image, last_hist)
        cut = cut_score > 0.55 or (previous_time is not None and timestamp - previous_time > 0.6)
        previous_time = timestamp
        if cut:
            shot += 1
            guard.reset()
            ball_tracker.reset()
            tracker.reset()
            colors, team_model, team_votes, previous_centers = [], None, {}, {}
        raw_detections, _ = detector.predict(image)
        people = [r for r in raw_detections if r["label"] == "person"]
        balls = [r for r in raw_detections if r["label"] == "ball"]
        boxes = np.array([r["bbox_xyxy"] for r in people], float).reshape(-1, 4)
        roles = [r.get("role", "person") for r in people]
        appearance = [
            shirt_color(image, box) if role == "player" else None
            for box, role in zip(boxes, roles, strict=True)
        ]
        tracks = tracker.update(boxes, np.zeros(len(boxes), int), timestamp, appearance)
        centers = {int(t): (b[:2] + b[2:]) / 2 for t, b in zip(tracks, boxes, strict=True)}
        moves = [v - previous_centers[k] for k, v in centers.items() if k in previous_centers]
        camera_shift = np.median(moves, axis=0) if len(moves) >= 4 else np.zeros(2)
        previous_centers = centers
        feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
        raw = camera.predict(image)
        fit = guard.check(raw, feet, timestamp)
        matrix = None
        if fit["accepted"]:
            matrix = np.array([[1, 0, 52.5], [0, 1, 34], [0, 0, 1]]) @ np.asarray(fit["homography"])
        tables["calibration"].append(
            {
                "frame_id": fid,
                "timestamp_s": timestamp,
                "shot_id": shot,
                "accepted": fit["accepted"],
                "reason": fit["reason"],
                "homography": matrix.ravel().tolist() if matrix is not None else None,
                "raw_homography": np.asarray(raw["homography"]).ravel().tolist()
                if raw.get("homography") is not None
                else None,
                "raw_accepted": bool(raw["accepted"]),
                "diagnostics_json": json.dumps(fit["diagnostics"]),
            }
        )
        projected = project(feet, matrix) if matrix is not None else np.full((len(feet), 2), np.nan)
        for index, (box, role, track, color) in enumerate(
            zip(boxes, roles, tracks, appearance, strict=True)
        ):
            if color is not None and team_model is None:
                colors.append(color)
                if len(colors) >= 80:
                    team_model = KMeans(2, n_init=10, random_state=22).fit(colors)
            team, margin = None, None
            if color is not None and team_model is not None:
                distances = np.linalg.norm(team_model.cluster_centers_ - color, axis=1)
                margin = float((distances.max() - distances.min()) / max(distances.max(), 1e-6))
                if margin >= 0.15:
                    votes = team_votes.setdefault(int(track), Counter())
                    votes[int(np.argmin(distances))] += 1
                    team = votes.most_common(1)[0][0]
            record = {
                "frame_id": fid,
                "source_frame": ordinal,
                "timestamp_s": timestamp,
                "shot_id": shot,
                "detection_id": f"f{fid}-d{index}",
                "tracklet_id": int(track),
                "role_hypothesis": role,
                "confidence": people[index]["score"],
                **dict(
                    zip(("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"), box.tolist(), strict=True)
                ),
                "team_cluster": team,
                "team_color_margin": margin,
                "jersey_hypothesis": None,
                "jersey_status": "unknown",
                "player_identity": None,
                "source": "video_detection",
            }
            tables["detections"].append(record)
            xy = projected[index]
            valid = np.isfinite(xy).all() and (-3 <= xy[0] <= 108 and -3 <= xy[1] <= 71)
            tables["state"].append(
                {
                    "frame_id": fid,
                    "timestamp_s": timestamp,
                    "tracklet_id": int(track),
                    "entity": role,
                    "team_cluster": team,
                    "x_m": float(xy[0]) if valid else None,
                    "y_m": float(xy[1]) if valid else None,
                    "status": "projected_observation" if valid else "unavailable",
                    "calibration_accepted": fit["accepted"],
                    "position_sigma_m": None,
                    "evidence_id": record["detection_id"],
                }
            )
            key = (shot, int(track))
            if (
                jersey
                and role in {"player", "goalkeeper"}
                and len(crops) < 300
                and timestamp - crop_times.get(key, -1e6) >= 1
            ):
                x0, y0, x1, y1 = np.round(box).astype(int)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
                crop = image[y0:y1, x0:x1]
                if crop.shape[0] >= 24 and crop.shape[1] >= 10:
                    cid = record["detection_id"]
                    path = out / "jersey/crops" / f"{cid}.png"
                    cv2.imwrite(str(path), crop)
                    crops.append(
                        {
                            "crop_id": cid,
                            "crop_path": f"crops/{cid}.png",
                            "crop_sha256": sha256(path),
                            "crop_height": crop.shape[0],
                            "crop_width": crop.shape[1],
                            "detection_id": cid,
                            "frame_id": fid,
                            "timestamp_s": timestamp,
                            "shot_id": shot,
                            "tracklet_id": int(track),
                        }
                    )
                    crop_times[key] = timestamp
        ball = ball_tracker.update(balls, timestamp, image.shape, camera_shift)
        tables["ball_tracks"].append(
            {
                "frame_id": fid,
                "timestamp_s": timestamp,
                "shot_id": shot,
                **ball,
                "camera_shift_xy": camera_shift.tolist(),
            }
        )
        for rank, candidate in enumerate(balls):
            tables["ball_candidates"].append(
                {
                    "frame_id": fid,
                    "timestamp_s": timestamp,
                    "rank": rank,
                    "confidence": candidate["score"],
                    **dict(zip(("x0", "y0", "x1", "y1"), candidate["bbox_xyxy"], strict=True)),
                    "selected": rank == ball["selected_index"],
                    "eligible_for_state": rank == ball["selected_index"]
                    and ball["eligible_for_state"],
                }
            )
        xy = (
            project([ball["center_xy"]], matrix)[0]
            if matrix is not None and ball["center_xy"] is not None
            else [np.nan, np.nan]
        )
        valid = np.isfinite(xy).all() and -3 <= xy[0] <= 108 and -3 <= xy[1] <= 71
        tables["state"].append(
            {
                "frame_id": fid,
                "timestamp_s": timestamp,
                "tracklet_id": -1,
                "entity": "ball_candidate",
                "team_cluster": None,
                "x_m": float(xy[0]) if valid else None,
                "y_m": float(xy[1]) if valid else None,
                "status": (
                    ball["status"]
                    if ball["status"] in {"estimated", "tentative"}
                    else "projected_hypothesis"
                )
                if valid
                else "unavailable",
                "ball_status": ball["status"],
                "ball_center_xy": ball["center_xy"],
                "calibration_accepted": fit["accepted"],
                "position_sigma_m": None,
                "evidence_id": f"f{fid}-ball{ball['selected_index']}"
                if ball["selected_index"] is not None
                else f"f{fid}-ball-estimate",
                "eligible_for_state": ball["eligible_for_state"],
            }
        )
        image_path = out / "frames" / f"{fid:05d}.jpg"
        cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tables["frames"].append(
            {
                "frame_id": fid,
                "source_frame": ordinal,
                "timestamp_s": timestamp,
                "pts": pts,
                "time_base": time_base,
                "shot_id": shot,
                "cut_score": cut_score,
                "width": image.shape[1],
                "height": image.shape[0],
                "players_detected": len(people),
                "ball_center_xy": ball["center_xy"],
                "ball_candidates": len(balls),
                "ball_status": ball["status"],
                "calibration_accepted": fit["accepted"],
                "calibration_reason": fit["reason"],
                "image_path": f"frames/{fid:05d}.jpg",
                "image_sha256": sha256(image_path),
            }
        )
        if fid % 25 == 0:
            print(
                json.dumps(
                    {
                        "frame": fid,
                        "time": timestamp,
                        "camera": fit["reason"],
                        "ball": ball["status"],
                    }
                ),
                flush=True,
            )
    if not tables["frames"]:
        raise ValueError("No decoded frames")
    jersey_rows = []
    if crops:
        from soccerviz.candidates.jersey_model import infer

        write_json(
            out / "jersey/crop-manifest.json",
            {
                "schema": "jersey-crops/v1",
                "source_sha256": source_hash,
                "ground_truth_used_in_sampling": False,
                "crops": crops,
            },
        )
        predictions = infer(
            out / "jersey/crop-manifest.json",
            Path(upstream_root) / "uncertainty-jnr",
            Path(model_root) / "jersey-vitb.pt",
            out / "jersey/inference",
            4,
            device,
        )
        indexed = {r["crop_id"]: r for r in predictions["predictions"]}
        jersey_rows = jersey_consensus(
            [
                {
                    **r,
                    "number": indexed[r["crop_id"]]["jersey_number"],
                    "accepted": indexed[r["crop_id"]]["status"] == "provisional",
                }
                for r in crops
            ]
        )
        histories = {}
        for row in jersey_rows:
            histories.setdefault((row["shot_id"], row["tracklet_id"]), []).append(row)
        for row in tables["detections"]:
            history = [
                r
                for r in histories.get((row["shot_id"], row["tracklet_id"]), [])
                if r["timestamp_s"] <= row["timestamp_s"]
            ]
            if history:
                row.update({k: history[-1][k] for k in ("jersey_hypothesis", "jersey_status")})
        write_json(
            out / "jersey/track-evidence.json",
            {
                "rows": jersey_rows,
                "policy": "Two agreeing accepted frames; any accepted conflict causes abstention; causal, shot-local.",
            },
        )
    with av.open(str(out / "preview.mp4"), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(str(hz)))
        stream.width, stream.height, stream.pix_fmt = 1560, 540, "yuv420p"
        stream.options = {"crf": "24", "preset": "fast"}
        for frame in tables["frames"]:
            fid = frame["frame_id"]
            pixels = render_frame(
                cv2.imread(str(out / frame["image_path"])),
                [r for r in tables["detections"] if r["frame_id"] == fid],
                [r for r in tables["state"] if r["frame_id"] == fid],
                frame,
            )
            video = av.VideoFrame.from_ndarray(pixels, format="bgr24")
            video.pts = round((frame["timestamp_s"] - tables["frames"][0]["timestamp_s"]) * 1000)
            video.time_base = Fraction(1, 1000)
            for packet in stream.encode(video):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    # Fixed empty-table columns are essential for existing harness imports.
    empty = {
        "detections": ["frame_id", "timestamp_s", "detection_id", "tracklet_id", "role_hypothesis"],
        "ball_candidates": [
            "frame_id",
            "timestamp_s",
            "rank",
            "confidence",
            "x0",
            "y0",
            "x1",
            "y1",
            "selected",
            "eligible_for_state",
        ],
    }
    for name, rows in tables.items():
        pd.DataFrame(rows, columns=None if rows else empty.get(name)).to_parquet(
            out / f"{name}.parquet", index=False
        )
    models = model_hashes(detector, camera)
    if crops:
        models["jersey"] = sha256(Path(model_root) / "jersey-vitb.pt")
    licence = licence_report(models)
    if shipping:
        require_shipping(licence)
    report = {
        "schema": "video-workflow/v1",
        "source_sha256": source_hash,
        "model_sha256": models,
        "licence": licence,
        "preset": preset,
        "calibration": calibration,
        "sample_hz": hz,
        "frames": len(tables["frames"]),
        "detections": len(tables["detections"]),
        "anonymous_tracklets": len({r["tracklet_id"] for r in tables["detections"]}),
        "span_s": tables["frames"][-1]["timestamp_s"] - tables["frames"][0]["timestamp_s"],
        "frames_with_ball_candidate": sum(r["ball_candidates"] > 0 for r in tables["frames"]),
        "ball_status_counts": dict(Counter(r["status"] for r in tables["ball_tracks"])),
        "accepted_calibration_frames": sum(r["accepted"] for r in tables["calibration"]),
        "camera_reasons": dict(Counter(r["reason"] for r in tables["calibration"])),
        "jersey_crops": len(crops),
        "jersey_supported_readings": sum(r["jersey_hypothesis"] is not None for r in jersey_rows),
        "cuts_detected": shot,
        "elapsed_s": time.perf_counter() - started,
        "safety": asdict(safety),
        "code_sha256": {
            p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("video_safety.py")]
        },
        "limitations": [
            "Heuristic guards, team clusters and temporal estimates are not calibrated truth.",
            "Generic person detectors cannot supply soccer roles; team shape abstains without roles.",
            "Camera cuts use a histogram heuristic; named and cross-cut identity remain unknown.",
            "Ball positions assume the ground plane; estimates never enter possession inference.",
        ],
    }
    if sha256(source) != source_hash:
        raise ValueError("Source video changed during analysis")
    write_json(out / "report.json", report)
    write_json(
        out / "files.json",
        {str(p.relative_to(out)): sha256(p) for p in out.rglob("*") if p.is_file()},
    )
    return report


def model_hashes(detector, camera):
    """Role→SHA256 of every checkpoint the detector and camera adapters loaded."""
    models = {
        "detector_" + k: v["sha256"] for k, v in detector.metadata.get("checkpoints", {}).items()
    }
    models.update(
        {"camera_" + k: v["sha256"] for k, v in camera.metadata.get("checkpoints", {}).items()}
    )
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preset", choices=PRESETS, default="rf-soccer")
    parser.add_argument("--calibration", choices=CALIBRATIONS, default="pnl")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--hz", type=float, default=5)
    parser.add_argument("--no-jersey", action="store_true")
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--upstream-root", type=Path, default=UPSTREAM_ROOT)
    parser.add_argument(
        "--shipping",
        action="store_true",
        help="Refuse to run unless every loaded checkpoint is cleared for shipping",
    )
    args = vars(parser.parse_args())
    args["jersey"] = not args.pop("no_jersey")
    print(json.dumps(run_workflow(**args), indent=2))


if __name__ == "__main__":
    main()
