"""Heatmap encode/decode, mirror pairs, export loading and the calibration adapter contract."""

from pathlib import Path

import numpy as np
import pytest

from soccerviz.candidates import pitch_keypoints as pk
from soccerviz.core.geometry import pitch_landmarks

EXPORT = Path("artifacts/public-data/roboflow/football-field-detection-f07vi-v18-coco")


def test_heatmap_roundtrip_is_within_half_a_heatmap_pixel():
    points = np.zeros((pk.NUM_KEYPOINTS, 3), dtype=np.float32)
    points[3] = (123.0, 456.0, 2)
    points[17] = (900.0, 40.0, 2)
    points[31] = (5.0, 955.0, 1)
    heatmaps, weights = pk.encode_heatmaps(points, 960, 960)
    assert weights.sum() == 3 and heatmaps.shape == (32, pk.HEATMAP_SIZE, pk.HEATMAP_SIZE)
    decoded, confidence = pk.decode_heatmaps(heatmaps * 30.0, 960, 960)  # sharp logits
    for k in (3, 17, 31):
        assert np.linalg.norm(decoded[k] - points[k, :2]) <= 0.5 * pk.STRIDE
        assert confidence[k] > 0.9
    assert confidence[0] < 0.01  # a flat map has no mass near its arbitrary peak


def test_heatmap_loss_rewards_the_target_location():
    torch = pytest.importorskip("torch")
    points = np.zeros((pk.NUM_KEYPOINTS, 3), dtype=np.float32)
    points[5] = (400.0, 300.0, 2)
    heatmaps, weights = pk.encode_heatmaps(points, 960, 960)
    targets, w = torch.from_numpy(heatmaps)[None], torch.from_numpy(weights)[None]
    optimal = np.log(heatmaps + 1e-6).astype(np.float32)  # softmax(log t) == t
    visible_only = {"invisible_weight": 0.0}
    good = pk.heatmap_loss(torch.from_numpy(optimal)[None], targets, w, **visible_only)
    flat = pk.heatmap_loss(torch.zeros_like(targets), targets, w, **visible_only)
    rolled = torch.from_numpy(np.roll(optimal, 40, axis=2))[None]
    wrong = pk.heatmap_loss(rolled, targets, w, **visible_only)
    assert good < 5.0 < flat < wrong  # target entropy about 4.4 nats; flat is log(240*240)


def test_heatmap_loss_drives_invisible_keypoints_flat():
    """An unseen landmark must not be allowed a sharp peak: that is the v18 failure."""
    torch = pytest.importorskip("torch")
    points = np.zeros((pk.NUM_KEYPOINTS, 3), dtype=np.float32)
    points[5] = (400.0, 300.0, 2)
    heatmaps, weights = pk.encode_heatmaps(points, 960, 960)
    targets, w = torch.from_numpy(heatmaps)[None], torch.from_numpy(weights)[None]
    logits = torch.from_numpy(np.log(heatmaps + 1e-6).astype(np.float32))[None].clone()
    flat_elsewhere = pk.heatmap_loss(logits, targets, w)
    spiked = logits.clone()
    spiked[0, 9] = torch.from_numpy(30.0 * heatmaps[5])  # invisible keypoint 9 copies a sharp peak
    # About 20 nats on the phantom keypoint alone, averaged over all 32 keypoints.
    assert (pk.heatmap_loss(spiked, targets, w) - flat_elsewhere) * pk.NUM_KEYPOINTS > 10.0
    assert pk.heatmap_loss(spiked, targets, w, invisible_weight=0.0) == pytest.approx(
        pk.heatmap_loss(logits, targets, w, invisible_weight=0.0)
    )
    _, confidence = pk.decode_heatmaps(spiked[0].numpy(), 960, 960)
    # The phantom the loss now penalises would pass predict()'s confidence cut.
    assert confidence[9] > pk.DEFAULT_PROTOCOL["landmark_confidence"]


def test_invisible_and_out_of_frame_points_get_no_target():
    points = np.zeros((pk.NUM_KEYPOINTS, 3), dtype=np.float32)
    points[0] = (100.0, 100.0, 0)
    points[1] = (2000.0, 100.0, 2)
    _, weights = pk.encode_heatmaps(points, 960, 960)
    assert weights.sum() == 0


def test_mirror_pairs_are_geometric_involutions():
    pairs = pk.mirror_pairs()
    table = pitch_landmarks()
    assert len(pairs) == 14 and len({i for pair in pairs for i in pair}) == 28
    for i, j in pairs:
        assert table[i, 0] == pytest.approx(105.0 - table[j, 0]) and table[i, 1] == table[j, 1]
    fixed = set(range(32)) - {i for pair in pairs for i in pair}
    assert fixed == {13, 14, 15, 16}  # the four halfway-line landmarks map to themselves


def test_preprocess_shape_and_normalisation():
    image = np.full((540, 960, 3), 128, dtype=np.uint8)
    tensor = pk.preprocess(image, 256)
    assert tensor.shape == (3, 256, 256) and tensor.dtype == np.float32
    assert abs(float(tensor[0].mean()) - (128 / 255 - 0.485) / 0.229) < 1e-4


def test_load_split_reads_the_real_export():
    if not EXPORT.exists():
        pytest.skip("Roboflow v18 export not downloaded")
    records = pk.load_split(EXPORT, "valid")
    assert len(records) == 34
    assert all(r["keypoints"].shape == (32, 3) for r in records)
    assert all(Path(r["image_path"]).exists() for r in records[:3])


class FakeNet:
    """Returns the heatmaps of a known pitch view so the adapter must recover the homography."""

    def __init__(self, heatmaps):
        import torch

        self.heatmaps = torch.from_numpy(heatmaps)

    def __call__(self, images):
        return self.heatmaps[None].to(images.device)

    def eval(self):
        return self

    def to(self, device):
        return self


def test_adapter_recovers_a_synthetic_homography():
    torch = pytest.importorskip("torch")
    matrix = np.array([[9.0, -1.5, 120.0], [0.4, 6.5, 40.0], [0.0, 0.002, 1.0]])
    world = pitch_landmarks()
    homogeneous = np.column_stack([world, np.ones(32)]) @ matrix.T
    image_points = homogeneous[:, :2] / homogeneous[:, 2:]
    points = np.column_stack([image_points, np.full(32, 2.0)]).astype(np.float32)
    points[[5, 16, 29], 2] = 0  # a few landmarks off screen
    heatmaps, _ = pk.encode_heatmaps(points, 960, 960)
    model = pk.HeatmapCalibrationModel.__new__(pk.HeatmapCalibrationModel)
    model.torch, model.device, model.net = torch, "cpu", FakeNet(heatmaps * 30.0)
    model.protocol = dict(pk.DEFAULT_PROTOCOL)
    model.num_primitives = 0  # the vertex-only checkpoint path
    from soccerviz.candidates.calibration_model import centered_homography

    model.centered_homography = centered_homography
    result = model.predict(np.zeros((960, 960, 3), dtype=np.uint8))
    assert result["accepted"] and result["reason"] == "accepted"
    assert 20 <= result["diagnostics"]["landmarks_above_threshold"] <= 29  # some land off-frame
    assert result["diagnostics"]["median_residual_m"] < 0.5
    confidence = result["diagnostics"]["landmark_confidence"]
    assert len(confidence) == 32 and all(confidence[k] < 0.01 for k in (5, 16, 29))


def test_primitive_targets_roundtrip_through_decode():
    """Rendered lines decode to points that lie on those lines, and empty maps to nothing."""
    from soccerviz.core.geometry import project
    from soccerviz.core.pitch_template import NUM_LINES, line_coefficients

    matrix = np.array([[9.0, -1.5, 120.0], [0.4, 6.5, 40.0], [0.0, 0.002, 1.0]])
    world = pitch_landmarks()
    homogeneous = np.column_stack([world, np.ones(32)]) @ matrix.T
    image_points = homogeneous[:, :2] / homogeneous[:, 2:]
    keypoints = np.column_stack([image_points, np.full(32, 2.0)]).astype(np.float32)
    samples = pk.primitive_samples(keypoints, 1280, 720)
    assert samples is not None and samples.shape[1] == 3
    targets = pk.encode_primitive_heatmaps(samples, 1280, 720)
    assert targets.shape == (pk.NUM_PRIMITIVES, pk.HEATMAP_SIZE, pk.HEATMAP_SIZE)
    assert targets.max() == 1.0 and (targets >= 0).all()
    logits = np.log(targets + 1e-6) - np.log(1 - targets + 1e-6)
    primitives, pixels = pk.decode_primitives(logits, 1280, 720)
    assert len(pixels) == pk.NUM_PRIMITIVES and len(primitives) >= 10
    ground = np.linalg.inv(matrix)
    lines = line_coefficients()
    for primitive in primitives:
        if primitive.kind != "line":
            continue
        distance = np.abs(
            project(primitive.image_points, ground) @ lines[primitive.index, :2]
            + lines[primitive.index, 2]
        )
        assert np.median(distance) < 0.6, (primitive.index, np.median(distance))
    assert all(p.index < NUM_LINES or p.kind == "conic" for p in primitives)
    assert pk.decode_primitives(np.full_like(logits, -10.0), 1280, 720)[0] == []


def test_primitive_samples_need_a_sound_truth():
    keypoints = np.zeros((32, 3), dtype=np.float32)
    keypoints[:3, 2] = 2  # three vertices cannot fit a homography
    assert pk.primitive_samples(keypoints, 960, 960) is None
    assert pk.encode_primitive_heatmaps(None, 960, 960).max() == 0.0


def test_attach_primitive_samples_on_the_real_export():
    if not EXPORT.exists():
        pytest.skip("Roboflow v18 export not downloaded")
    records = pk.attach_primitive_samples(pk.load_split(EXPORT, "valid"))
    assert len(records) == 34 and all(r["primitives"] is not None for r in records)
    assert all(len(r["primitives"]) > 50 for r in records)


def test_primitive_loss_prefers_the_target_and_masks_unfit_images():
    torch = pytest.importorskip("torch")
    targets = np.zeros((2, pk.NUM_PRIMITIVES, 24, 24), dtype=np.float32)
    targets[0, 4, 12, :] = 1.0  # the halfway line across image 0
    t = torch.from_numpy(targets)
    good = torch.from_numpy(np.where(targets > 0.5, 8.0, -8.0).astype(np.float32))
    bad = -good
    mask = torch.tensor([1.0, 0.0])
    assert pk.primitive_loss(good, t, mask) < 0.01 < pk.primitive_loss(bad, t, mask)
    assert pk.primitive_loss(bad, t, torch.tensor([0.0, 0.0])) == 0.0


def test_checkpoint_primitives_defaults_to_vertex_only():
    assert pk.checkpoint_primitives({"model": {}}) == 0
    assert pk.checkpoint_primitives({"model": {}, "num_primitives": 20}) == 20


def test_adapter_solves_a_halfway_view_from_lines_and_circle():
    """Five vertices in frame: the vertex-only fit refuses, the line-aware fit accepts."""
    torch = pytest.importorskip("torch")
    from soccerviz.candidates.calibration_model import centered_homography
    from soccerviz.core.geometry import project
    from soccerviz.core.pitch_template import NUM_LINES, line_names

    matrix = np.array([[9.0, -1.5, 120.0], [0.4, 6.5, 40.0], [0.0, 0.002, 1.0]])
    world = pitch_landmarks()
    homogeneous = np.column_stack([world, np.ones(32)]) @ matrix.T
    image_points = homogeneous[:, :2] / homogeneous[:, 2:]
    keypoints = np.column_stack([image_points, np.zeros(32)]).astype(np.float32)
    keypoints[[13, 14, 15, 30, 31], 2] = 2  # halfway top, circle top/bottom/left/right
    heatmaps, _ = pk.encode_heatmaps(keypoints, 960, 960)
    samples = pk.primitive_samples(
        np.column_stack([image_points, np.full(32, 2.0)]).astype(np.float32), 960, 960
    )
    keep = np.isin(samples[:, 2], [line_names().index("halfway"), NUM_LINES])
    targets = pk.encode_primitive_heatmaps(samples[keep], 960, 960)
    logits = np.log(targets + 1e-6) - np.log(1 - targets + 1e-6)
    maps = np.concatenate([heatmaps * 30.0, logits]).astype(np.float32)

    def adapter(num_primitives):
        model = pk.HeatmapCalibrationModel.__new__(pk.HeatmapCalibrationModel)
        model.torch, model.device, model.net = torch, "cpu", FakeNet(maps[: 32 + num_primitives])
        model.protocol = dict(pk.DEFAULT_PROTOCOL)
        model.centered_homography = centered_homography
        model.num_primitives = num_primitives
        return model

    vertex_only = adapter(0).predict(np.zeros((960, 960, 3), dtype=np.uint8))
    assert not vertex_only["accepted"] and vertex_only["reason"] == "fewer_than_six_landmarks"
    result = adapter(pk.NUM_PRIMITIVES).predict(np.zeros((960, 960, 3), dtype=np.uint8))
    assert result["accepted"], result["diagnostics"]
    assert sorted(result["diagnostics"]["primitives_detected"]) == [4, NUM_LINES]
    assert result["diagnostics"]["inlier_primitives"] == 7
    ground = np.linalg.inv(matrix)
    centred = centered_homography(ground)
    grid = np.array([[x, y] for x in range(40, 66, 5) for y in range(0, 69, 17)], float)
    predicted = project(project(grid, matrix), np.asarray(result["homography"]))
    truth = project(project(grid, matrix), centred)
    assert np.median(np.linalg.norm(predicted - truth, axis=1)) < 0.5
