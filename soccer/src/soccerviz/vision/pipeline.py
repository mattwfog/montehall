"""PTS-preserving video → detections → anonymous tracklets → pitch observations.

This first adapter uses the public Roboflow soccer YOLO checkpoints. All model
outputs remain hypotheses; coverage and self-consistency are not accuracy scores.
"""

from __future__ import annotations

import importlib.metadata
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from soccerviz.core.assets import sha256
from soccerviz.core.data import write_json
from soccerviz.core.geometry import calibrate, pitch_landmarks, project
from soccerviz.core.identity import TrackletAssociator


def video_frames(path: Path, hz=5.0, seconds=20.0):
    last_bucket, first, last = None, None, -np.inf
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for ordinal, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError("Video frame lacks PTS; cannot establish reliable timestamps")
            timestamp = float(frame.pts * frame.time_base)
            if timestamp <= last:
                raise ValueError("Non-monotonic presentation timestamps")
            last = timestamp
            if first is None:
                first = timestamp
            if timestamp - first >= seconds:
                break
            bucket = int(np.floor((timestamp - first) * hz + 1e-7))
            if bucket != last_bucket:
                last_bucket = bucket
                yield (
                    ordinal,
                    timestamp,
                    int(frame.pts),
                    str(frame.time_base),
                    frame.to_ndarray(format="bgr24"),
                )


def shirt_color(image, box):
    x0, y0, x1, y1 = box
    x0, x1 = max(0, int(x0 + (x1 - x0) * 0.2)), min(image.shape[1], int(x0 + (x1 - x0) * 0.8))
    y0, y1 = max(0, int(y0 + (y1 - y0) * 0.2)), min(image.shape[0], int(y0 + (y1 - y0) * 0.55))
    crop = image[y0:y1, x0:x1]
    if crop.size < 30:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 60)
    pixels = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[~grass]
    return np.median(pixels, axis=0) if len(pixels) >= 10 else None


def cut_distance(image, previous):
    hsv = cv2.cvtColor(cv2.resize(image, (160, 90)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return (
        float(cv2.compareHist(hist, previous, cv2.HISTCMP_BHATTACHARYYA))
        if previous is not None
        else 0.0
    ), hist


def ball_candidates(model, image, device):
    height, width = image.shape[:2]
    candidates = []
    # Four overlapping tiles retain resolution for a small ball.
    for y0, y1 in [(0, int(height * 0.6)), (int(height * 0.4), height)]:
        for x0, x1 in [(0, int(width * 0.6)), (int(width * 0.4), width)]:
            result = model.predict(
                image[y0:y1, x0:x1], imgsz=640, conf=0.25, device=device, verbose=False
            )[0]
            if result.boxes is None:
                continue
            for box, confidence in zip(
                result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), strict=True
            ):
                box += [x0, y0, x0, y0]
                candidates.append((box, float(confidence)))
    # Keep one observation hypothesis for the ball, without interpolating misses.
    return sorted(candidates, key=lambda x: x[1], reverse=True)


def run_video(source: Path, models: Path, out: Path, seconds=20.0, hz=5.0, device="cuda:0"):
    from ultralytics import YOLO

    if seconds <= 0 or hz <= 0:
        raise ValueError("seconds and hz must be positive")
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    completion = out / "report.json"
    if completion.exists():
        raise ValueError(f"Completed run already exists at {out}; select a new output directory")
    started = time.monotonic()
    names = [
        "football-player-detection.pt",
        "football-pitch-detection.pt",
        "football-ball-detection.pt",
    ]
    detector, pitch_model, ball_model = [YOLO(str(models / name)) for name in names]
    tracker, colors, team_model = TrackletAssociator(), [], None
    frames, detections, landmarks, calibrations, states, candidates_log = [], [], [], [], [], []
    last_hist, shot, team_counts = None, 0, {}
    encoder = av.open(str(out / "preview.mp4"), mode="w")
    output_stream = encoder.add_stream("libx264", rate=Fraction(str(hz)))
    output_stream.width, output_stream.height = 960, 540
    output_stream.pix_fmt = "yuv420p"
    output_stream.options = {"crf": "23"}
    first_time = None
    try:
        for ordinal, timestamp, pts, time_base, image in video_frames(source, hz, seconds):
            if first_time is None:
                first_time = timestamp
            frame_id = len(frames)
            cut_score, last_hist = cut_distance(image, last_hist)
            if cut_score > 0.55:
                shot += 1
                tracker.reset()
            result = detector.predict(image, imgsz=1280, conf=0.3, device=device, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidence = result.boxes.conf.cpu().numpy()
            role_names = [str(result.names[c]).lower() for c in classes]
            keep = np.array([role != "ball" for role in role_names], dtype=bool)
            boxes, classes, confidence = boxes[keep], classes[keep], confidence[keep]
            role_names = [role for role, take in zip(role_names, keep, strict=True) if take]
            appearance = [
                shirt_color(image, box) if role == "player" else None
                for box, role in zip(boxes, role_names, strict=True)
            ]
            track_ids = tracker.update(boxes, classes, timestamp, appearance)
            keypoints = pitch_model.predict(
                image, imgsz=640, conf=0.25, device=device, verbose=False
            )[0].keypoints
            if keypoints is not None and len(keypoints.xy):
                points = keypoints.xy[0].cpu().numpy()
                point_conf = (
                    keypoints.conf[0].cpu().numpy()
                    if keypoints.conf is not None
                    else np.zeros(len(points))
                )
            else:
                points, point_conf = np.zeros((32, 2)), np.zeros(32)
            if len(points) != 32:
                raise ValueError(
                    "Pitch model keypoint count differs from configured landmark order"
                )
            valid = (point_conf >= 0.5) & (points > 0).all(axis=1)
            fit = calibrate(points[valid], pitch_landmarks()[valid])
            calibrations.append(
                {
                    "frame_id": frame_id,
                    "timestamp_s": timestamp,
                    "shot_id": shot,
                    **fit.diagnostics(),
                    "homography": fit.matrix.ravel().tolist() if fit.accepted else None,
                }
            )
            for k in np.flatnonzero(valid):
                landmarks.append(
                    {
                        "frame_id": frame_id,
                        "keypoint": int(k),
                        "u": float(points[k, 0]),
                        "v": float(points[k, 1]),
                        "confidence": float(point_conf[k]),
                    }
                )
            feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
            projected = (
                project(feet, fit.matrix) if fit.accepted else np.full((len(feet), 2), np.nan)
            )
            annotated = image.copy()
            for j, (box, role, tid) in enumerate(zip(boxes, role_names, track_ids, strict=True)):
                color = appearance[j]
                if color is not None and team_model is None:
                    colors.append(color)
                    if len(colors) >= 80:
                        team_model = KMeans(2, n_init=10, random_state=22).fit(colors)
                team, margin = None, None
                if color is not None and team_model is not None:
                    distances = np.linalg.norm(team_model.cluster_centers_ - color, axis=1)
                    raw = int(np.argmin(distances))
                    margin = float((distances.max() - distances.min()) / max(distances.max(), 1e-6))
                    if margin >= 0.15:
                        team_counts.setdefault(int(tid), Counter())[raw] += 1
                        team = team_counts[int(tid)].most_common(1)[0][0]
                record = {
                    "detection_id": f"f{frame_id}-d{j}",
                    "frame_id": frame_id,
                    "source_frame": ordinal,
                    "timestamp_s": timestamp,
                    "shot_id": shot,
                    "tracklet_id": int(tid),
                    "role_hypothesis": role,
                    "bbox_x0": float(box[0]),
                    "bbox_y0": float(box[1]),
                    "bbox_x1": float(box[2]),
                    "bbox_y1": float(box[3]),
                    "confidence": float(confidence[j]),
                    "team_cluster": team,
                    "team_color_margin": margin,
                    "player_identity": None,
                    "jersey_hypothesis": None,
                    "identity_status": "anonymous_tracklet",
                    "source": "video_detection",
                }
                detections.append(record)
                xy = projected[j]
                plausible = bool(np.isfinite(xy).all() and -3 <= xy[0] <= 108 and -3 <= xy[1] <= 71)
                states.append(
                    {
                        "frame_id": frame_id,
                        "timestamp_s": timestamp,
                        "tracklet_id": int(tid),
                        "entity": role,
                        "team_cluster": team,
                        "x_m": float(xy[0]) if plausible else None,
                        "y_m": float(xy[1]) if plausible else None,
                        "status": "projected_observation" if plausible else "unavailable",
                        "calibration_accepted": fit.accepted,
                        "metric_geometry": "assumed_105x68",
                        "position_sigma_m": None,
                        "evidence_id": record["detection_id"],
                    }
                )
                rgb = (
                    (255, 140, 70)
                    if team == 0
                    else (80, 100, 255)
                    if team == 1
                    else (210, 210, 210)
                )
                x0, y0, x1, y1 = box.astype(int)
                cv2.rectangle(annotated, (x0, y0), (x1, y1), rgb, 2)
                cv2.putText(
                    annotated,
                    f"T{tid} {role} C{team if team is not None else '?'}",
                    (x0, max(15, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    rgb,
                    1,
                )
            balls = ball_candidates(ball_model, image, device)
            for k, (box, conf) in enumerate(balls):
                candidates_log.append(
                    {
                        "frame_id": frame_id,
                        "timestamp_s": timestamp,
                        "rank": k,
                        "confidence": conf,
                        "x0": float(box[0]),
                        "y0": float(box[1]),
                        "x1": float(box[2]),
                        "y1": float(box[3]),
                        "selected": k == 0,
                    }
                )
            if balls:
                box, conf = balls[0]
                center = (box[:2] + box[2:]) / 2
                xy = project(center[None], fit.matrix)[0] if fit.accepted else [np.nan, np.nan]
                plausible = bool(np.isfinite(xy).all() and -3 <= xy[0] <= 108 and -3 <= xy[1] <= 71)
                states.append(
                    {
                        "frame_id": frame_id,
                        "timestamp_s": timestamp,
                        "tracklet_id": -1,
                        "entity": "ball_candidate",
                        "team_cluster": None,
                        "x_m": float(xy[0]) if plausible else None,
                        "y_m": float(xy[1]) if plausible else None,
                        "status": "projected_hypothesis" if plausible else "unavailable",
                        "calibration_accepted": fit.accepted,
                        "metric_geometry": "ground_plane_assumption",
                        "position_sigma_m": None,
                        "evidence_id": f"f{frame_id}-ball0",
                    }
                )
                cv2.circle(annotated, tuple(center.astype(int)), 10, (0, 255, 255), 2)
            frames.append(
                {
                    "frame_id": frame_id,
                    "source_frame": ordinal,
                    "timestamp_s": timestamp,
                    "pts": pts,
                    "time_base": time_base,
                    "shot_id": shot,
                    "cut_score": cut_score,
                    "players_detected": len(boxes),
                    "ball_candidates": len(balls),
                    "calibration_accepted": fit.accepted,
                    "width": image.shape[1],
                    "height": image.shape[0],
                }
            )
            cv2.putText(
                annotated,
                f"{timestamp:.2f}s | calibration: {fit.reason} | ball candidates: {len(balls)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if frame_id % max(1, int(hz * 5)) == 0:
                cv2.imwrite(str(out / "frames" / f"{frame_id:05d}.jpg"), image)
                cv2.imwrite(str(out / "frames" / f"{frame_id:05d}-annotated.jpg"), annotated)
            encoded = av.VideoFrame.from_ndarray(cv2.resize(annotated, (960, 540)), format="bgr24")
            encoded.pts = round((timestamp - first_time) * 1000)
            encoded.time_base = Fraction(1, 1000)
            for packet in output_stream.encode(encoded):
                encoder.mux(packet)
            if frame_id % 25 == 0:
                print(
                    f"Video frame {frame_id}: {timestamp:.2f}s, {len(boxes)} detections, calibration {fit.reason}",
                    flush=True,
                )
        for packet in output_stream.encode():
            encoder.mux(packet)
    finally:
        encoder.close()
    if not frames:
        raise ValueError("No decoded frames")
    tables = {
        "frames": frames,
        "detections": detections,
        "landmarks": landmarks,
        "calibration": calibrations,
        "state": states,
        "ball_candidates": candidates_log,
    }
    for name, rows in tables.items():
        path = out / f"{name}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        temporary.replace(path)
    observation = pd.DataFrame(detections)
    summary = {
        "source_sha256": sha256(source),
        "model_sha256": {n: sha256(models / n) for n in names},
        "sample_hz": hz,
        "frames": len(frames),
        "span_s": frames[-1]["timestamp_s"] - frames[0]["timestamp_s"],
        "detections": len(detections),
        "anonymous_tracklets": int(observation.tracklet_id.nunique()),
        "role_counts": dict(Counter(observation.role_hypothesis)),
        "team_assignment_coverage": float(observation.team_cluster.notna().mean()),
        "frames_with_ball_candidate": sum(f["ball_candidates"] > 0 for f in frames),
        "accepted_calibration_frames": sum(f["calibration_accepted"] for f in frames),
        "cuts_detected": shot,
        "elapsed_s": time.monotonic() - started,
        "versions": {
            name: importlib.metadata.version(name) for name in ["ultralytics", "av", "numpy"]
        },
        "accuracy": {
            "detection_map": None,
            "tracking_idf1": None,
            "team_accuracy": None,
            "pitch_position_error_m": None,
            "ball_precision": None,
        },
        "limitations": [
            "Inference diagnostics only; no labeled video ground truth yet",
            "Team colors are anonymous clusters, not identified home/away teams",
            "Tracklet continuity is local image-space association and resets at detected cuts",
            "Pitch dimensions assumed 105x68; residuals measure model self-consistency, not true accuracy",
            "Aerial ball projection violates the ground-plane assumption",
            "Jersey OCR and named-player identity abstain until evidence exists",
        ],
    }
    write_json(completion, summary)
    return summary


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/demo/2e57b9_0.mp4"))
    parser.add_argument("--models", type=Path, default=Path("data/demo"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/video/demo-v1"))
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--hz", type=float, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(
        json.dumps(
            run_video(args.source, args.models, args.out, args.seconds, args.hz, args.device),
            indent=2,
        )
    )
