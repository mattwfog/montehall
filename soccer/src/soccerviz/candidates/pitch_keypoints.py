"""Permissive pitch calibrator: timm HRNet heatmaps of vertices, lines and conics → homography.

Trained on the CC BY 4.0 Roboflow Universe football-field-detection export with
timm (Apache-2.0) and ImageNet HRNet-W32 weights, so the camera half of the video
workflow no longer depends on PnLCalib (GPL-2.0, SoccerNet weights). Keypoint
index j is vertex j+1 of `geometry.pitch_landmarks()`; see that docstring for the
trap in the export's keypoint names.

Line and conic channels (`pitch_template.primitive_names()`) are supervised from each
training image's own fitted homography, so they cost no labels. A checkpoint records
`num_primitives`; 0 is the vertex-only model, whose fit goes through
`geometry.calibrate`, and 20 the line-aware one, fitted by
`primitive_calibration.calibrate_primitives` so a halfway-line view is solvable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from soccerviz.core import geometry, pitch_template
from soccerviz.core.assets import sha256
from soccerviz.core.geometry import calibrate, pitch_landmarks
from soccerviz.core.primitive_calibration import Primitive, calibrate_primitives

NUM_KEYPOINTS = 32
INPUT_SIZE = 960  # square network input; the v18 export is already stretched to 960x960
STRIDE = 4
HEATMAP_SIZE = INPUT_SIZE // STRIDE
SIGMA = 2.0  # heatmap pixels
BACKBONE = "hrnet_w32"
PRETRAINED_TAG = "hrnet_w32.ms_in1k"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_PROTOCOL = {"input": INPUT_SIZE, "landmark_confidence": 0.5, "primitive_threshold": 0.5}
INVISIBLE_WEIGHT = 1.0  # loss weight of the uniform target on invisible keypoints
NUM_PRIMITIVES = pitch_template.NUM_PRIMITIVES
PRIMITIVE_SIGMA = 2.0  # heatmap pixels, half-width of a rendered line
PRIMITIVE_POSITIVE_WEIGHT = 10.0  # line pixels are a few percent of a map
PRIMITIVE_LOSS_WEIGHT = 10.0  # BCE means are ~0.05 at convergence against ~4.4 nats of keypoint CE
PRIMITIVE_MIN_PIXELS = 8
PRIMITIVE_MAX_POINTS = 80
GROUND_TRUTH_MAX_RESIDUAL_M = 1.0
PRIMITIVE_LINK_PX = 12  # consecutive samples further apart than this are not joined


def load_split(export_dir, split):
    """Records of one COCO split: image path and (32, 3) keypoints in source pixels."""
    folder = Path(export_dir) / split
    data = json.loads((folder / "_annotations.coco.json").read_text())
    category = next(c for c in data["categories"] if c.get("keypoints"))
    if len(category["keypoints"]) != NUM_KEYPOINTS:
        raise ValueError(
            f"{folder}: expected {NUM_KEYPOINTS} keypoints, got {len(category['keypoints'])}"
        )
    images = {image["id"]: image for image in data["images"]}
    records = []
    for annotation in data["annotations"]:
        if annotation["category_id"] != category["id"]:
            continue
        image = images[annotation["image_id"]]
        points = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(NUM_KEYPOINTS, 3)
        records.append(
            {
                "image_path": str(folder / image["file_name"]),
                "width": image["width"],
                "height": image["height"],
                "keypoints": points,
            }
        )
    if not records:
        raise ValueError(f"{folder}: no pitch annotations")
    return records


def mirror_pairs():
    """Index pairs swapped by a horizontal image flip: vertex mirrored across the halfway line."""
    table = pitch_landmarks()
    mirrored = np.column_stack([table[:, 0].max() - table[:, 0], table[:, 1]])
    pairs = []
    for i, point in enumerate(mirrored):
        j = int(np.argmin(np.linalg.norm(table - point, axis=1)))
        if i < j:
            pairs.append((i, j))
    return pairs


def encode_heatmaps(points, width, height, size=HEATMAP_SIZE, sigma=SIGMA):
    """Gaussian target per keypoint in heatmap coordinates; weight 0 for invisible ones."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    heatmaps = np.zeros((NUM_KEYPOINTS, size, size), dtype=np.float32)
    weights = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
    for k, (x, y, visible) in enumerate(points):
        if visible <= 0:
            continue
        cx, cy = x * size / width, y * size / height
        if not (0 <= cx < size and 0 <= cy < size):
            continue
        heatmaps[k] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
        weights[k] = 1.0
    return heatmaps, weights


def decode_heatmaps(logits, width, height, radius=3):
    """Peak per keypoint with quarter-pixel refinement, in source pixels, plus confidence.

    Logits are one spatial distribution per keypoint (trained with `heatmap_loss`);
    confidence is the softmax probability mass within `radius` heatmap pixels of the
    peak, so a sharp prediction scores near 1 and a diffuse one near 0.
    """
    size = logits.shape[-1]
    flat = logits.reshape(NUM_KEYPOINTS, -1).astype(np.float64)
    index = flat.argmax(axis=1)
    shifted = np.exp(flat - flat.max(axis=1, keepdims=True))
    probabilities = (shifted / shifted.sum(axis=1, keepdims=True)).reshape(
        NUM_KEYPOINTS, size, size
    )
    px, py = (index % size).astype(np.float32), (index // size).astype(np.float32)
    confidence = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
    for k in range(NUM_KEYPOINTS):
        x, y = int(px[k]), int(py[k])
        if 0 < x < size - 1:
            px[k] += 0.25 * np.sign(logits[k, y, x + 1] - logits[k, y, x - 1])
        if 0 < y < size - 1:
            py[k] += 0.25 * np.sign(logits[k, y + 1, x] - logits[k, y - 1, x])
        window = probabilities[
            k, max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1
        ]
        confidence[k] = window.sum()
    points = np.column_stack([px * width / size, py * height / size])
    return points, confidence


def heatmap_loss(logits, targets, weights, invisible_weight=INVISIBLE_WEIGHT):
    """Spatial softmax cross-entropy per keypoint.

    Visible keypoints (weight 1) are trained against the normalised Gaussian target.
    Invisible and out-of-frame keypoints (weight 0) are trained against the uniform
    distribution, scaled by `invisible_weight`, so their peak carries no probability
    mass and the `decode_heatmaps` confidence separates seen from unseen landmarks.

    Unlike pixel-wise MSE this cannot be minimised by predicting zeros everywhere
    (the 2026-09-08 MSE run collapsed to that within three epochs). The v18 run left
    invisible keypoints unsupervised; their heatmaps came out as sharp as the visible
    ones, and the resulting phantom landmarks rejected 27 of 45 benchmark frames.
    """
    import torch

    batch, keypoints = logits.shape[:2]
    log_probabilities = torch.log_softmax(logits.reshape(batch, keypoints, -1), dim=-1)
    flat_targets = targets.reshape(batch, keypoints, -1)
    normalised = flat_targets / flat_targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    visible_term = -(normalised * log_probabilities).sum(dim=-1)
    uniform_term = -log_probabilities.mean(dim=-1)
    invisible = (1.0 - weights) * invisible_weight
    total = (visible_term * weights + uniform_term * invisible).sum()
    return total / (weights + invisible).sum().clamp(min=1.0)


def ground_truth_world_to_image(keypoints):
    """World→image homography fitted from a labelled image's visible vertices, or None.

    `geometry.calibrate` supplies the image→world fit and its guard; the inverse is
    oriented so the visible pitch has w > 0. A fit whose median residual exceeds
    `GROUND_TRUTH_MAX_RESIDUAL_M` gives no line supervision rather than wrong lines.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    visible = keypoints[:, 2] > 0
    if visible.sum() < 4:
        return None
    fit = calibrate(keypoints[visible, :2], pitch_landmarks()[visible])
    if not fit.accepted or fit.median_residual_m > GROUND_TRUTH_MAX_RESIDUAL_M:
        return None
    world_to_image = np.linalg.inv(fit.matrix)
    return pitch_template.orient_world_to_image(world_to_image, pitch_landmarks()[visible])


def primitive_samples(keypoints, width, height):
    """(M, 3) in-frame image samples of x, y, primitive index from the fitted truth; None if unfit."""
    world_to_image = ground_truth_world_to_image(keypoints)
    if world_to_image is None:
        return None
    return pitch_template.project_primitives(world_to_image, width, height)


def attach_primitive_samples(records):
    """New records carrying `primitives` samples (or None) rendered from their own vertices."""
    return [
        {**r, "primitives": primitive_samples(r["keypoints"], r["width"], r["height"])}
        for r in records
    ]


def encode_primitive_heatmaps(samples, width, height, size=HEATMAP_SIZE, sigma=PRIMITIVE_SIGMA):
    """(NUM_PRIMITIVES, size, size) targets in [0, 1]: blurred polylines through the samples.

    Consecutive samples of one primitive are joined when close in the heatmap; a gap
    (the primitive left the frame between them) is left unjoined. A primitive with
    no in-frame sample gets an all-zero map, which is correct supervision: with the
    homography known, its absence is known too.
    """
    heatmaps = np.zeros((NUM_PRIMITIVES, size, size), dtype=np.float32)
    if samples is None:
        return heatmaps
    scaled = np.column_stack([samples[:, 0] * size / width, samples[:, 1] * size / height])
    for index in range(NUM_PRIMITIVES):
        points = scaled[samples[:, 2] == index]
        if not len(points):
            continue
        canvas = np.zeros((size, size), dtype=np.float32)
        rounded = np.rint(points).astype(int)
        gaps = np.linalg.norm(np.diff(points, axis=0), axis=1) > PRIMITIVE_LINK_PX
        for start, end, gap in zip(rounded[:-1], rounded[1:], gaps):
            if not gap:
                cv2.line(canvas, tuple(start), tuple(end), 1.0, 1)
        for x, y in rounded:
            if 0 <= x < size and 0 <= y < size:
                canvas[y, x] = 1.0
        blurred = cv2.GaussianBlur(canvas, (0, 0), sigma)
        if blurred.max() > 0:
            heatmaps[index] = blurred / blurred.max()
    return heatmaps


def primitive_loss(logits, targets, mask, positive_weight=PRIMITIVE_POSITIVE_WEIGHT):
    """Per-pixel binary cross-entropy against the soft line targets, weighted toward the line.

    `mask` is (batch,) 1 for images with a fitted truth; an image without one
    contributes nothing, since its line maps are unknown rather than empty.
    """
    import torch

    per_pixel = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    weights = 1.0 + targets * (positive_weight - 1.0)
    per_image = (per_pixel * weights).mean(dim=(1, 2, 3)) / weights.mean(dim=(1, 2, 3))
    return (per_image * mask).sum() / mask.sum().clamp(min=1.0)


def _ridge_points(probability, threshold):
    """Sub-pixel ridge of one probability map: pixels above threshold that peak along x or y.

    Thinning matters: the rendered band is about four heatmap pixels wide, over a metre
    at the far touchline. A pixel that is a maximum among its horizontal neighbours is
    refined along x by a parabola through the three values, likewise along y.
    """
    padded = np.pad(probability, 1, mode="edge")
    centre = padded[1:-1, 1:-1]
    left, right = padded[1:-1, :-2], padded[1:-1, 2:]
    up, down = padded[:-2, 1:-1], padded[2:, 1:-1]
    above = centre > threshold
    peak_x = above & (centre >= left) & (centre >= right)
    peak_y = above & (centre >= up) & (centre >= down)
    ys, xs = np.nonzero(peak_x | peak_y)
    if not len(xs):
        return np.zeros((0, 2))
    c, lft, rgt, u, d = (a[ys, xs] for a in (centre, left, right, up, down))
    dx = np.where(peak_x[ys, xs], _parabolic_offset(lft, c, rgt), 0.0)
    dy = np.where(peak_y[ys, xs], _parabolic_offset(u, c, d), 0.0)
    return np.column_stack([xs + dx, ys + dy])


def _parabolic_offset(before, peak, after):
    denominator = before - 2 * peak + after
    offset = np.where(np.abs(denominator) > 1e-9, 0.5 * (before - after) / denominator, 0.0)
    return np.clip(offset, -0.5, 0.5)


def decode_primitives(logits, width, height, threshold=DEFAULT_PROTOCOL["primitive_threshold"]):
    """Primitives (lines, conics) with sub-pixel ridge points in source coordinates.

    At most `PRIMITIVE_MAX_POINTS` per primitive, chosen with a fixed seed so a frame
    decodes identically every time; fewer than `PRIMITIVE_MIN_PIXELS` is treated as
    noise and dropped. The per-channel pixel counts above threshold are returned too.
    """
    size = logits.shape[-1]
    probabilities = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    rng = np.random.default_rng(0)
    primitives, pixels = [], []
    for index in range(NUM_PRIMITIVES):
        pixels.append(int((probabilities[index] > threshold).sum()))
        ridge = _ridge_points(probabilities[index], threshold)
        if len(ridge) < PRIMITIVE_MIN_PIXELS:
            continue
        if len(ridge) > PRIMITIVE_MAX_POINTS:
            ridge = ridge[rng.choice(len(ridge), PRIMITIVE_MAX_POINTS, replace=False)]
        points = np.column_stack([ridge[:, 0] * width / size, ridge[:, 1] * height / size])
        kind = "line" if index < pitch_template.NUM_LINES else "conic"
        primitives.append(Primitive(kind, index, points.astype(float)))
    return primitives, pixels


def preprocess(image_bgr, size=INPUT_SIZE):
    """BGR uint8 → normalised CHW float32 at the square network input size."""
    rgb = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
    normalised = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalised.transpose(2, 0, 1))


def build_network(pretrained=False, num_primitives=NUM_PRIMITIVES):
    """timm HRNet-W32 stride-4 features → 32 vertex + `num_primitives` line/conic maps."""
    import timm
    from torch import nn

    class PitchKeypointNet(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone, self.head = backbone, head

        def forward(self, images):
            return self.head(self.backbone(images)[0])

    backbone = timm.create_model(
        f"{BACKBONE}.{PRETRAINED_TAG.split('.')[1]}" if pretrained else BACKBONE,
        pretrained=pretrained,
        features_only=True,
        out_indices=(1,),
    )
    channels = backbone.feature_info.channels()[0]
    if backbone.feature_info.reduction()[0] != STRIDE:
        raise ValueError("Backbone stride differs from the heatmap stride")
    head = nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, NUM_KEYPOINTS + num_primitives, 1),
    )
    return PitchKeypointNet(backbone, head)


def checkpoint_primitives(state):
    """Line/conic channel count recorded in a checkpoint; vertex-only checkpoints predate it."""
    return int(state.get("num_primitives", 0))


def fit_from_maps(maps, width, height, protocol, num_primitives):
    """Decode one image's maps and fit: (fit, keypoint confidence, valid mask, extra diagnostics).

    Shared by the adapter and the trainer's per-epoch validation so the checkpoint is
    selected on exactly the fit the workflow will run.
    """
    points, confidence = decode_heatmaps(maps[:NUM_KEYPOINTS], width, height)
    valid = confidence >= protocol["landmark_confidence"]
    if not num_primitives:
        return calibrate(points[valid], pitch_landmarks()[valid]), confidence, valid, {}
    primitives, pixels = decode_primitives(
        maps[NUM_KEYPOINTS:], width, height, protocol["primitive_threshold"]
    )
    vertices = [
        Primitive("point", k, points[k : k + 1], pitch_landmarks()[k])
        for k in np.flatnonzero(valid)
    ]
    fit = calibrate_primitives(vertices + primitives)
    extra = {"primitives_detected": [p.index for p in primitives], "primitive_pixels": pixels}
    return fit, confidence, valid, extra


class HeatmapCalibrationModel:
    """Same contract as YoloCalibrationModel: predict(image_bgr) → accepted / homography."""

    def __init__(self, checkpoint, protocol, device="cuda:0"):
        import torch

        from soccerviz.candidates.calibration_model import centered_homography, package_versions

        self.torch, self.device = torch, device
        self.protocol = {**DEFAULT_PROTOCOL, **(protocol.get("heatmap") or {})}
        self.centered_homography = centered_homography
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.num_primitives = checkpoint_primitives(state)
        self.net = build_network(pretrained=False, num_primitives=self.num_primitives)
        self.net.load_state_dict(state.get("model", state), strict=True)
        self.net.to(device).eval()
        fitter = "primitive_calibration" if self.num_primitives else "geometry.calibrate"
        self.metadata = {
            "name": f"timm-{BACKBONE}-heatmap+{fitter}",
            "num_primitives": self.num_primitives,
            "device": device,
            "checkpoints": {
                "pitch": {"path": str(Path(checkpoint).resolve()), "sha256": sha256(checkpoint)}
            },
            "protocol": self.protocol,
            "input_hw": [self.protocol["input"], self.protocol["input"]],
            "heatmap_hw": [HEATMAP_SIZE, HEATMAP_SIZE],
            "versions": package_versions(),
            "geometry_source_sha256": sha256(Path(geometry.__file__)),
            "coordinates": "Existing0..105/0..68 projection translated by[-52.5,-34]",
        }

    def heatmaps(self, image_bgr):
        """All output maps of one image: (32 + num_primitives, H, W) logits."""
        tensor = self.torch.from_numpy(preprocess(image_bgr, self.protocol["input"]))[None]
        with self.torch.inference_mode():
            return self.net(tensor.to(self.device))[0].float().cpu().numpy()

    def keypoints(self, image_bgr):
        """(32, 2) source-pixel points and (32,) peak confidences."""
        height, width = image_bgr.shape[:2]
        return decode_heatmaps(self.heatmaps(image_bgr)[:NUM_KEYPOINTS], width, height)

    def predict(self, image):
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize()
        start = time.perf_counter()
        height, width = image.shape[:2]
        fit, confidence, valid, extra = fit_from_maps(
            self.heatmaps(image), width, height, self.protocol, self.num_primitives
        )
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize()
        return {
            "accepted": fit.accepted,
            "reason": fit.reason,
            "homography": self.centered_homography(fit.matrix).tolist() if fit.accepted else None,
            "diagnostics": {
                **fit.diagnostics(),
                **extra,
                "landmarks_above_threshold": int(valid.sum()),
                "landmark_confidence": [round(float(c), 4) for c in confidence],
                "inference_and_geometry_ms": (time.perf_counter() - start) * 1000,
            },
        }
