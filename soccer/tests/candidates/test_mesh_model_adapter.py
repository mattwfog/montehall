from typing import ClassVar

import numpy as np
import pytest

from soccerviz.candidates.mesh_model import (
    ARM_KEYPOINTS,
    FOOT_KEYPOINTS,
    MHR70_KEYPOINTS,
    body_extremes,
    infer,
    package_fingerprint,
    project_keypoints,
    record_or_verify_hashes,
    summarize,
    validate_person,
)
from tests.candidates.test_pose_model_adapter import benchmark

WIDTH, HEIGHT = 60, 40


def standing_person(box_xyxy, focal=100.0, depth=5.0):
    """A 1.7 m figure placed so its projection fills the box; 2D derived from 3D."""
    x0, y0, x1, y1 = box_xyxy
    names = list(MHR70_KEYPOINTS)
    kp3 = np.zeros((len(names), 3))
    for i, name in enumerate(names):
        if name in FOOT_KEYPOINTS:
            kp3[i, 1] = 0.85
        elif "hip" in name:
            kp3[i, 1] = 0.0
        elif name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
            kp3[i, 1] = -0.8
        else:
            kp3[i, 1] = -0.3
        kp3[i, 0] = 0.1 if name.startswith("left") else -0.1
    # Choose cam_t so the hips land at the box centre and feet sit at the box bottom.
    centre_x, centre_y = (x0 + x1) / 2, (y0 + y1) / 2
    cam_t = np.array(
        [(centre_x - WIDTH / 2) * depth / focal, (centre_y - HEIGHT / 2) * depth / focal, depth]
    )
    kp2 = project_keypoints(kp3, cam_t, focal, WIDTH, HEIGHT)
    return {
        "keypoints_2d_xy": kp2.tolist(),
        "keypoints_3d_xyz_m": kp3.tolist(),
        "cam_t_m": cam_t.tolist(),
        "focal_length_px": focal,
        "rig_params": {
            "global_rot": [0, 0, 0],
            "body_pose": [0.0] * 4,
            "shape": [0.0],
            "scale": [1.0],
            "hand_pose": [],
        },
    }


def test_projection_matches_upstream_convention():
    kp3 = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
    xy = project_keypoints(kp3, [0.0, 0.0, 2.0], 10.0, WIDTH, HEIGHT)
    assert xy.tolist() == [[30.0, 20.0], [35.0, 30.0]]
    with pytest.raises(ValueError, match="behind"):
        project_keypoints(kp3, [0.0, 0.0, -1.0], 10.0, WIDTH, HEIGHT)


def test_validate_person_rejects_inconsistent_2d_and_bad_focal():
    person = standing_person([10, 5, 18, 35])
    validate_person(person, WIDTH, HEIGHT)
    shifted = {**person, "keypoints_2d_xy": (np.asarray(person["keypoints_2d_xy"]) + 2).tolist()}
    with pytest.raises(ValueError, match="disagree"):
        validate_person(shifted, WIDTH, HEIGHT)
    with pytest.raises(ValueError, match="Focal"):
        validate_person({**person, "focal_length_px": 0.0}, WIDTH, HEIGHT)


def test_arm_exclusion_covers_hands_wrists_and_forearm_landmarks():
    assert {"left_wrist", "right_wrist", "left_elbow", "right_elbow"} <= ARM_KEYPOINTS
    assert all(n in ARM_KEYPOINTS for n in MHR70_KEYPOINTS if "tip" in n and "toe" not in n)
    assert not ({"nose", "neck", "left_heel", "right_big_toe_tip", "left_acromion"} & ARM_KEYPOINTS)
    person = standing_person([10, 5, 18, 35])
    result = body_extremes(person["keypoints_2d_xy"], person["keypoints_3d_xyz_m"])
    assert result["lowest_foot_image"]["name"] in FOOT_KEYPOINTS
    assert not {e["name"] for e in result["eligible_points"]} & ARM_KEYPOINTS


def test_package_fingerprint_changes_with_source(tmp_path):
    root = tmp_path / "sam_3d_body"
    root.mkdir()
    (root / "a.py").write_text("one")
    before = package_fingerprint(tmp_path)
    (root / "a.py").write_text("two")
    assert package_fingerprint(tmp_path) != before
    with pytest.raises(ValueError, match="No upstream package"):
        package_fingerprint(tmp_path / "missing")


def test_gated_hashes_are_recorded_once_and_verified_afterwards(tmp_path):
    snapshot = tmp_path / "snap"
    (snapshot / "assets").mkdir(parents=True)
    for name in ("model.ckpt", "assets/mhr_model.pt", "model_config.yaml"):
        (snapshot / name).write_bytes(name.encode())
    spec = {
        "repo_id": "x/y",
        "files": {
            "checkpoint": "model.ckpt",
            "rig": "assets/mhr_model.pt",
            "config": "model_config.yaml",
        },
        "expected_bytes": {"model.ckpt": len(b"model.ckpt")},
    }
    manifest = tmp_path / "hashes.json"
    _, status = record_or_verify_hashes(snapshot, spec, manifest)
    assert status == "recorded_on_first_load"
    _, status = record_or_verify_hashes(snapshot, spec, manifest)
    assert status == "verified_against_first_load"
    (snapshot / "model.ckpt").write_bytes(b"model.ckp!")
    with pytest.raises(ValueError, match="changed since first load"):
        record_or_verify_hashes(snapshot, spec, manifest)
    with pytest.raises(ValueError, match="hub listed"):
        record_or_verify_hashes(snapshot, {**spec, "expected_bytes": {"model.ckpt": 1}}, manifest)


class FakeBackend:
    calls: ClassVar[list[int]] = []

    def __init__(self, name, cache_dir, *, device, upstream):
        self.metadata = {"name": name}
        self.last_timing_ms = {"total": 1.0}
        self.last_vertices = None

    def predict(self, image, boxes):
        FakeBackend.calls.append(int(image[0, 0, 0]))
        self.last_vertices = np.zeros((len(boxes), 4, 3), dtype=np.float16)
        return [standing_person(box) for box in boxes]


def test_infer_persists_resumes_saves_vertices_and_summarizes(tmp_path):
    manifest, detections = benchmark(tmp_path)
    out = tmp_path / "mesh"
    FakeBackend.calls = []
    partial = infer(
        manifest, detections, out, limit=2, backend_factory=FakeBackend, save_vertices=True
    )
    assert partial["complete_manifest"] is False
    assert FakeBackend.calls == [1, 2]
    (out / "predictions.json").unlink()
    full = infer(manifest, detections, out, backend_factory=FakeBackend, save_vertices=True)
    assert FakeBackend.calls == [1, 2, 3]
    assert full["complete_manifest"] is True and full["runtime"]["resumed_frames"] == 2
    assert (out / full["frames"][0]["vertices_path"]).exists()
    assert len(full["frames"][0]["people"][0]["keypoints_2d_xy"]) == 70
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        infer(manifest, detections, out, backend_factory=FakeBackend)
    report = summarize(out / "predictions.json", tmp_path / "summary.json")
    assert report["kind"] == "sanity_rates_not_accuracy"
    assert report["people"] == 6
    assert report["feet_below_hips_image_rate"] == 1.0
    assert report["camera_depth_m"]["median"] == 5.0
    assert report["keypoint_vertical_extent_m"]["median"] == pytest.approx(1.65)


def test_backend_refuses_non_cuda_device():
    from soccerviz.candidates.mesh_model import SAM3DBodyBackend

    with pytest.raises(ValueError, match="CUDA"):
        SAM3DBodyBackend("sam-3d-body-dinov3", "unused", device="mps")
