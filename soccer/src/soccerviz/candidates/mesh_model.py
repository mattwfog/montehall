"""SAM 3D Body full-body mesh recovery on frozen detector person boxes (CUDA worker).

Upstream is imported unmodified from a pinned checkout and hard-codes CUDA in its
inference path, so this backend runs only in the Spark worker. The gated
checkpoint's hashes are not published by Meta; the first verified load records
them beside the snapshot and every later load must match. Outputs are the 70
MHR keypoints in source pixels and camera-frame metres, camera translation and
focal length, plus the rig parameters that regenerate the mesh; vertices are
optional because 18k vertices per person do not belong in JSON.
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from soccerviz.candidates.pose_model import (
    _load_partial,
    _read_verified,
    frame_person_boxes,
    inside_box,
    save_json,
)
from soccerviz.core.assets import sha256

UPSTREAM = "https://github.com/facebookresearch/sam-3d-body"
UPSTREAM_COMMIT = "b5c765a0d89d789985e186d396315e7590887b94"
UPSTREAM_PACKAGE_SHA256 = "dfd0eeaac76c8da4fda281e61b67bc881046b4d904c560b47897e790dfd0f956"
BACKENDS = {
    "sam-3d-body-dinov3": {
        "repo_id": "facebook/sam-3d-body-dinov3",
        "revision": "main",
        "files": {
            "checkpoint": "model.ckpt",
            "rig": "assets/mhr_model.pt",
            "config": "model_config.yaml",
        },
        "expected_bytes": {"model.ckpt": 2109129346, "assets/mhr_model.pt": 696110248},
        "license": "SAM License (Meta, 2025-11-19); gated, manual approval",
        "family": "SAM 3D Body, DINOv3 backbone, Momentum Human Rig",
    },
    "sam-3d-body-vith": {
        "repo_id": "facebook/sam-3d-body-vith",
        "revision": "main",
        "files": {
            "checkpoint": "model.ckpt",
            "rig": "assets/mhr_model.pt",
            "config": "model_config.yaml",
        },
        "expected_bytes": {},
        "license": "SAM License (Meta, 2025-11-19); gated, manual approval",
        "family": "SAM 3D Body, ViT-H backbone, Momentum Human Rig",
    },
}
MHR70_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe_tip", "left_small_toe_tip", "left_heel",
    "right_big_toe_tip", "right_small_toe_tip", "right_heel",
    "right_thumb_tip", "right_thumb_first_joint", "right_thumb_second_joint",
    "right_thumb_third_joint", "right_index_tip", "right_index_first_joint",
    "right_index_second_joint", "right_index_third_joint", "right_middle_tip",
    "right_middle_first_joint", "right_middle_second_joint", "right_middle_third_joint",
    "right_ring_tip", "right_ring_first_joint", "right_ring_second_joint",
    "right_ring_third_joint", "right_pinky_tip", "right_pinky_first_joint",
    "right_pinky_second_joint", "right_pinky_third_joint", "right_wrist",
    "left_thumb_tip", "left_thumb_first_joint", "left_thumb_second_joint",
    "left_thumb_third_joint", "left_index_tip", "left_index_first_joint",
    "left_index_second_joint", "left_index_third_joint", "left_middle_tip",
    "left_middle_first_joint", "left_middle_second_joint", "left_middle_third_joint",
    "left_ring_tip", "left_ring_first_joint", "left_ring_second_joint",
    "left_ring_third_joint", "left_pinky_tip", "left_pinky_first_joint",
    "left_pinky_second_joint", "left_pinky_third_joint", "left_wrist",
    "left_olecranon", "right_olecranon", "left_cubital_fossa", "right_cubital_fossa",
    "left_acromion", "right_acromion", "neck",
)  # fmt: skip
# Offside law: head, body and feet count; hands and arms do not.
FOOT_KEYPOINTS = (
    "left_ankle", "right_ankle", "left_big_toe_tip", "left_small_toe_tip", "left_heel",
    "right_big_toe_tip", "right_small_toe_tip", "right_heel",
)  # fmt: skip
HIP_KEYPOINTS = ("left_hip", "right_hip")
ARM_KEYPOINTS = frozenset(
    n
    for n in MHR70_KEYPOINTS
    if n in ("left_elbow", "right_elbow", "left_wrist", "right_wrist")
    or any(
        part in n for part in ("thumb", "index", "middle", "ring", "pinky", "olecranon", "cubital")
    )
)
PROJECTION_TOLERANCE_PX = 0.5


def package_fingerprint(upstream):
    """Aggregate hash of the upstream Python package, AppleDouble sidecars excluded."""
    import hashlib

    root = Path(upstream) / "sam_3d_body"
    files = sorted(p for p in root.rglob("*.py") if not p.name.startswith("._"))
    if not files:
        raise ValueError(f"No upstream package at {root}")
    payload = "".join(f"{p.relative_to(root)} {sha256(p)}\n" for p in files)
    return hashlib.sha256(payload.encode()).hexdigest()


def record_or_verify_hashes(snapshot, spec, manifest_path):
    """Meta publishes no checksums; pin the first verified download and refuse drift."""
    observed = {}
    for name in spec["files"].values():
        path = Path(snapshot) / name
        size = path.stat().st_size
        expected = spec["expected_bytes"].get(name)
        if expected is not None and size != expected:
            raise ValueError(f"{name} is {size} bytes; the hub listed {expected}")
        observed[name] = {"sha256": sha256(path), "bytes": size}
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text())["files"]
        if recorded != observed:
            raise ValueError(f"Gated checkpoint files changed since first load: {manifest_path}")
        return observed, "verified_against_first_load"
    save_json(
        manifest_path,
        {
            "schema": "gated-checkpoint-hashes/v1",
            "repo_id": spec["repo_id"],
            "files": observed,
            "verification": "Locally computed on first load; no independently published checksum",
        },
    )
    return observed, "recorded_on_first_load"


def project_keypoints(keypoints_3d, cam_t, focal_length, width, height):
    """Perspective projection the upstream applies (principal point at image centre)."""
    points = np.asarray(keypoints_3d, dtype=float) + np.asarray(cam_t, dtype=float)
    depth = points[:, 2:3]
    if (depth <= 0).any():
        raise ValueError("Keypoint behind the camera")
    xy = points[:, :2] * float(focal_length) / depth
    return xy + np.array([width / 2.0, height / 2.0])


def validate_person(person, width, height):
    kp2 = np.asarray(person["keypoints_2d_xy"], dtype=float)
    kp3 = np.asarray(person["keypoints_3d_xyz_m"], dtype=float)
    cam_t = np.asarray(person["cam_t_m"], dtype=float)
    n = len(MHR70_KEYPOINTS)
    if kp2.shape != (n, 2) or kp3.shape != (n, 3) or cam_t.shape != (3,):
        raise ValueError("Expected 70 MHR keypoints in 2D and 3D and a 3-vector camera translation")
    if not (np.isfinite(kp2).all() and np.isfinite(kp3).all() and np.isfinite(cam_t).all()):
        raise ValueError("Nonfinite mesh output")
    focal = float(person["focal_length_px"])
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("Focal length must be positive and finite")
    reprojected = project_keypoints(kp3, cam_t, focal, width, height)
    error = float(np.abs(reprojected - kp2).max())
    if error > PROJECTION_TOLERANCE_PX:
        raise ValueError(f"2D keypoints disagree with projected 3D keypoints by {error:.2f} px")
    return kp2, kp3, cam_t


def body_extremes(keypoints_2d, keypoints_3d):
    """Offside-eligible points and the lowest foot in the image; arms are excluded by law."""
    kp2 = np.asarray(keypoints_2d, dtype=float)
    kp3 = np.asarray(keypoints_3d, dtype=float)
    eligible = [
        {"name": name, "xy": kp2[i].tolist(), "xyz_m": kp3[i].tolist()}
        for i, name in enumerate(MHR70_KEYPOINTS)
        if name not in ARM_KEYPOINTS
    ]
    feet = [e for e in eligible if e["name"] in FOOT_KEYPOINTS]
    return {
        "lowest_foot_image": max(feet, key=lambda e: e["xy"][1]),
        "eligible_points": eligible,
        "rule": "head, body and feet only; hands, wrists, elbows and forearm landmarks excluded",
        "note": "Which point is furthest forward depends on attacking direction and the "
        "pitch homography; this only names the candidates",
    }


def _as_int_or_list(value):
    """Upstream configs carry IMAGE_SIZE either as one int or as a [height, width] list."""
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return int(value)


class SAM3DBodyBackend:
    """Pinned upstream code plus a gated Hugging Face snapshot, loaded offline."""

    def __init__(
        self, name, cache_dir, *, device="cuda:0", upstream="artifacts/upstreams/sam-3d-body"
    ):
        if name not in BACKENDS:
            raise ValueError(f"Unknown mesh backend {name!r}; known: {sorted(BACKENDS)}")
        if not str(device).startswith("cuda"):
            raise ValueError("Upstream SAM 3D Body hard-codes CUDA tensors; use a cuda device")
        upstream = Path(upstream).resolve()
        fingerprint = package_fingerprint(upstream)
        if fingerprint != UPSTREAM_PACKAGE_SHA256:
            raise ValueError(
                f"Upstream package differs from pinned commit {UPSTREAM_COMMIT}: "
                f"expected {UPSTREAM_PACKAGE_SHA256}, observed {fingerprint}"
            )
        import torch
        from huggingface_hub import snapshot_download

        spec = BACKENDS[name]
        snapshot = Path(
            snapshot_download(
                spec["repo_id"],
                revision=spec["revision"],
                cache_dir=str(cache_dir),
                local_files_only=True,
            )
        )
        hashes, hash_status = record_or_verify_hashes(
            snapshot, spec, Path(cache_dir) / f"{name}.hashes.json"
        )
        sys.path.insert(0, str(upstream))
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        model, config = load_sam_3d_body(
            str(snapshot / spec["files"]["checkpoint"]),
            device=device,
            mhr_path=str(snapshot / spec["files"]["rig"]),
        )
        self.estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=config)
        self.torch, self.device, self.name = torch, device, name
        self.last_timing_ms = {}
        self.last_vertices = None
        self.metadata = {
            "name": name,
            **{k: v for k, v in spec.items() if k != "expected_bytes"},
            "snapshot_path": str(snapshot),
            "file_hashes": hashes,
            "file_hash_status": hash_status,
            "upstream": UPSTREAM,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_package_sha256": UPSTREAM_PACKAGE_SHA256,
            "keypoints": list(MHR70_KEYPOINTS),
            "inference_type": "full",
            "input_size": _as_int_or_list(config.MODEL.IMAGE_SIZE),
            "camera": "default intrinsics from image size; no FOV estimator, no mask conditioning",
            "precision": "float32",
            "device": str(device),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("torch", "torchvision", "pytorch-lightning", "roma", "numpy")
            },
        }

    def predict(self, image_bgr, boxes_xyxy):
        """One dict per box, in box order; keypoints in source pixels and camera metres."""
        if not boxes_xyxy:
            self.last_vertices = None
            return []
        height, width = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        started = time.perf_counter()
        outputs = self.estimator.process_one_image(
            rgb, bboxes=np.asarray(boxes_xyxy, dtype=np.float32), inference_type="full"
        )
        self.torch.cuda.synchronize(self.device)
        finished = time.perf_counter()
        if len(outputs) != len(boxes_xyxy):
            raise ValueError("Mesh output count differs from box count")
        people, vertices = [], []
        for out in outputs:
            person = {
                "keypoints_2d_xy": np.asarray(out["pred_keypoints_2d"], dtype=float).tolist(),
                "keypoints_3d_xyz_m": np.asarray(out["pred_keypoints_3d"], dtype=float).tolist(),
                "cam_t_m": np.asarray(out["pred_cam_t"], dtype=float).reshape(3).tolist(),
                "focal_length_px": float(np.asarray(out["focal_length"]).reshape(-1)[0]),
                "rig_params": {
                    key: np.asarray(out[source], dtype=float).reshape(-1).tolist()
                    for key, source in (
                        ("global_rot", "global_rot"),
                        ("body_pose", "body_pose_params"),
                        ("shape", "shape_params"),
                        ("scale", "scale_params"),
                        ("hand_pose", "hand_pose_params"),
                    )
                },
            }
            validate_person(person, width, height)
            people.append(person)
            vertices.append(np.asarray(out["pred_vertices"], dtype=np.float16))
        self.last_vertices = np.stack(vertices) if vertices else None
        self.last_timing_ms = {"total": (finished - started) * 1000}
        return people


def infer(
    manifest_path,
    predictions_path,
    out,
    *,
    backend_name="sam-3d-body-dinov3",
    cache_dir="artifacts/models/hf",
    device="cuda:0",
    upstream="artifacts/upstreams/sam-3d-body",
    limit=None,
    prefixes=(),
    save_vertices=False,
    backend_factory=SAM3DBodyBackend,
):
    """Frame by frame; each frame lands in frames.jsonl as it finishes and reruns resume."""
    out = Path(out)
    if (out / "predictions.json").exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {out / 'predictions.json'}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    manifest = json.loads(Path(manifest_path).read_text())
    predictions = json.loads(Path(predictions_path).read_text())
    if manifest.get("schema") != "vision-benchmark/v1":
        raise ValueError("Expected a vision-benchmark/v1 manifest")
    if predictions.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("Detector predictions are not bound to this frozen manifest")
    rows = frame_person_boxes(manifest, predictions)
    selected = rows[:limit] if limit else rows
    out.mkdir(parents=True, exist_ok=True)
    partial_path = out / "frames.jsonl"
    done = _load_partial(partial_path)
    backend = backend_factory(backend_name, cache_dir, device=device, upstream=upstream)
    started = time.perf_counter()
    timings = []
    with partial_path.open("a") as partial:
        for number, row in enumerate(selected, start=1):
            frame = row["frame"]
            key = frame["sequence"], frame["frame_id"]
            if key in done:
                continue
            image = _read_verified(frame, prefixes)
            people = backend.predict(image, [p["bbox_xyxy"] for p in row["people"]])
            record = _frame_record(frame, row["people"], people)
            if save_vertices and backend.last_vertices is not None:
                path = out / "vertices" / f"{frame['sequence']}-{frame['frame_id']:06d}.npz"
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, vertices=backend.last_vertices)
                record["vertices_path"] = str(path.relative_to(out))
            partial.write(json.dumps(record, allow_nan=False) + "\n")
            partial.flush()
            done[key] = record
            timings.append(backend.last_timing_ms.get("total", 0.0))
            if number == 1 or number % 25 == 0 or number == len(selected):
                print(f"mesh: {number}/{len(selected)} frames", flush=True)
    frames = [done[(r["frame"]["sequence"], r["frame"]["frame_id"])] for r in selected]
    output = {
        "schema": "mesh-predictions/v1",
        "backend": backend.metadata,
        "manifest_sha256": sha256(manifest_path),
        "detector_predictions_sha256": sha256(predictions_path),
        "detector_backend": predictions["backend"],
        "box_source": "predicted_detector_person_boxes",
        "ground_truth_used_in_inference": False,
        "complete_manifest": len(frames) == len(rows),
        "adapter_sha256": sha256(__file__),
        "output_coordinates": "2D: original-image pixels; 3D: camera frame, metres, after cam_t",
        "vertices_saved": save_vertices,
        "runtime": {
            "session_s": time.perf_counter() - started,
            "frames_inferred_this_session": len(timings),
            "median_total_ms_per_frame": float(np.median(timings)) if timings else None,
            "resumed_frames": len(frames) - len(timings),
        },
        "frames": frames,
    }
    save_json(out / "predictions.json", output)
    return output


def _frame_record(frame, people, meshes):
    if len(meshes) != len(people):
        raise ValueError("Backend returned a mesh count that differs from the box count")
    return {
        "sequence": frame["sequence"],
        "frame_id": frame["frame_id"],
        "width": frame["width"],
        "height": frame["height"],
        "timestamp_s": frame["timestamp_s"],
        "image_sha256": frame["sha256"],
        "people": [
            {
                "detection_index": person["detection_index"],
                "detector_score": person["detector_score"],
                "predicted_role": person["predicted_role"],
                "bbox_xyxy": person["bbox_xyxy"],
                **mesh,
            }
            for person, mesh in zip(people, meshes, strict=True)
        ],
    }


def summarize(predictions_path, out):
    """Sanity rates on saved mesh output. No 3D labels exist: this is not accuracy."""
    data = json.loads(Path(predictions_path).read_text())
    if data.get("schema") != "mesh-predictions/v1":
        raise ValueError("Expected mesh-predictions/v1")
    names = list(MHR70_KEYPOINTS)
    feet = [names.index(n) for n in FOOT_KEYPOINTS]
    hips = [names.index(n) for n in HIP_KEYPOINTS]
    counts = Counter()
    depths, heights = [], []
    for frame in data["frames"]:
        for person in frame["people"]:
            kp2, kp3, cam_t = validate_person(person, frame["width"], frame["height"])
            counts["people"] += 1
            counts["by_role", person.get("predicted_role")] += 1
            counts["keypoints"] += len(kp2)
            counts["keypoints_inside_box"] += sum(inside_box(p, person["bbox_xyxy"]) for p in kp2)
            if max(kp2[i][1] for i in feet) > max(kp2[i][1] for i in hips):
                counts["feet_below_hips_image"] += 1
            depths.append(float(cam_t[2]))
            heights.append(float(kp3[:, 1].max() - kp3[:, 1].min()))
    people = counts["people"]
    report = {
        "schema": "mesh-summary/v1",
        "kind": "sanity_rates_not_accuracy",
        "predictions_sha256": sha256(predictions_path),
        "backend": data["backend"]["name"],
        "frames": len(data["frames"]),
        "people": people,
        "people_by_role": {
            str(k[1]): v for k, v in counts.items() if isinstance(k, tuple) and k[0] == "by_role"
        },
        "keypoints_inside_box_rate": counts["keypoints_inside_box"] / counts["keypoints"]
        if counts["keypoints"]
        else None,
        "feet_below_hips_image_rate": counts["feet_below_hips_image"] / people if people else None,
        "reprojection_consistency": f"every person within {PROJECTION_TOLERANCE_PX} px",
        "camera_depth_m": _quantiles(depths),
        "keypoint_vertical_extent_m": _quantiles(heights),
        "limitations": [
            "No keypoint or mesh ground truth in the frozen benchmark; rates check plausibility only",
            "Boxes are detector predictions, so mesh errors include detector box errors",
            "Camera intrinsics are the upstream default from image size, not calibrated",
            "Depth and extent are metres in a model-scaled camera frame, not measured pitch units",
            "No temporal smoothing, no identity, no pitch projection in this worker",
        ],
    }
    save_json(out, report)
    return report


def _quantiles(values):
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }
