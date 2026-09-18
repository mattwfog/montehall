"""Court landmark segmentation -> per-frame homography.

Wraps our own MIT-licensed court model (smp DeepLabV3Plus / efficientnet-b3,
11 landmark classes, 768px input, ImageNet normalization — architecture
facts recovered from the checkpoint). Landmark line classes are fitted to
lines, intersected into point correspondences with known court geometry,
and solved into an image->court homography per sampled frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from montehall_cv.court import CENTER_COURT_FT, LANDMARK_LINES_FT, line_intersection_ft
from montehall_cv.pipeline.homography import (
    FittedLine,
    correspondence_spread,
    fit_line,
    intersect,
    reprojection_residual,
    solve_homography,
)

CLASSES = [
    "background",
    "baseline_left",
    "baseline_right",
    "sideline_top",
    "sideline_bottom",
    "midline",
    "three_pt_arc_left",
    "three_pt_arc_right",
    "center_circle",
    "free_throw_left",
    "free_throw_right",
]
INPUT_SIZE = 768
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MIN_LINE_PIXELS = 250
MIN_CIRCLE_PIXELS = 150
MAX_RESIDUAL_FT = 3.0
MIN_SPREAD = 4.0


@dataclass(frozen=True)
class CourtSolve:
    frame_idx: int
    ts_ms: int
    h: np.ndarray | None  # 3x3 image->court, None if no valid solve
    residual_ft: float | None
    n_points: int


class CourtSegmenter:
    def __init__(self, weights_path: Path, device: str = "cuda") -> None:
        import segmentation_models_pytorch as smp
        import torch

        self._torch = torch
        self._device = torch.device(device)
        model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b3",
            encoder_weights=None,
            in_channels=3,
            classes=len(CLASSES),
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
        self._model = model.to(self._device).eval().half()

    def masks(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Batched segmentation: RGB uint8 frames -> class-id masks at INPUT_SIZE."""
        torch = self._torch
        batch = torch.stack(
            [
                torch.nn.functional.interpolate(
                    torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
                    size=(INPUT_SIZE, INPUT_SIZE),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                for img in images
            ]
        )
        mean = torch.from_numpy(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.from_numpy(IMAGENET_STD).view(1, 3, 1, 1)
        batch = ((batch - mean) / std).half().to(self._device)
        with torch.no_grad():
            logits = self._model(batch)
        return list(logits.argmax(1).cpu().numpy().astype(np.uint8))


def solve_frame(
    mask: np.ndarray, frame_w: int, frame_h: int, frame_idx: int, ts_ms: int
) -> CourtSolve:
    """Fit landmark lines in a mask and solve the image->court homography."""
    scale_x = frame_w / mask.shape[1]
    scale_y = frame_h / mask.shape[0]

    lines: dict[str, FittedLine] = {}
    for name in LANDMARK_LINES_FT:
        class_id = CLASSES.index(name)
        ys, xs = np.nonzero(mask == class_id)
        if len(xs) < MIN_LINE_PIXELS:
            continue
        pts = np.stack([xs * scale_x, ys * scale_y], axis=1)
        lines[name] = fit_line(pts)

    image_pts: list[np.ndarray] = []
    court_pts: list[tuple[float, float]] = []
    names = sorted(lines)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            court_pt = line_intersection_ft(LANDMARK_LINES_FT[name_a], LANDMARK_LINES_FT[name_b])
            if court_pt is None:
                continue
            image_pt = intersect(lines[name_a], lines[name_b])
            if image_pt is None:
                continue
            image_pts.append(image_pt)
            court_pts.append(court_pt)

    center_ys, center_xs = np.nonzero(mask == CLASSES.index("center_circle"))
    if len(center_xs) >= MIN_CIRCLE_PIXELS:
        image_pts.append(
            np.array([center_xs.mean() * scale_x, center_ys.mean() * scale_y])
        )
        court_pts.append(CENTER_COURT_FT)

    if len(image_pts) < 4:
        return CourtSolve(frame_idx, ts_ms, None, None, len(image_pts))

    src = np.asarray(image_pts)
    dst = np.asarray(court_pts)
    if correspondence_spread(dst) < MIN_SPREAD:
        return CourtSolve(frame_idx, ts_ms, None, None, len(image_pts))
    try:
        h = solve_homography(src, dst)
        # residual inside the guard: a degenerate H (w~0 at a correspondence)
        # raises in project_points and is a failed solve, not a fatal error
        residual = reprojection_residual(h, src, dst)
    except ValueError:
        return CourtSolve(frame_idx, ts_ms, None, None, len(image_pts))
    if residual > MAX_RESIDUAL_FT:
        return CourtSolve(frame_idx, ts_ms, None, residual, len(image_pts))
    return CourtSolve(frame_idx, ts_ms, h, residual, len(image_pts))
