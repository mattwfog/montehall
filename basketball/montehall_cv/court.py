"""Canonical court coordinate system (design ruling D4).

One canonical space; every other convention is a transform at the edge.
Canonical: NCAA court in feet, x along the 94ft length, y along the 50ft
width, origin at the bottom-left corner (SportVU convention).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

NCAA_LENGTH_FT = 94.0
NCAA_WIDTH_FT = 50.0
CENTER_COURT_FT = (47.0, 25.0)

# Court-space lines for landmark classes, as (axis, value): x-lines are
# baselines/midline (constant x), y-lines are sidelines (constant y).
# Image-top is assumed to be the far sideline (standard broadcast side view).
LANDMARK_LINES_FT: dict[str, tuple[str, float]] = {
    "baseline_left": ("x", 0.0),
    "baseline_right": ("x", NCAA_LENGTH_FT),
    "midline": ("x", NCAA_LENGTH_FT / 2),
    "sideline_top": ("y", NCAA_WIDTH_FT),
    "sideline_bottom": ("y", 0.0),
}


def line_intersection_ft(a: tuple[str, float], b: tuple[str, float]) -> tuple[float, float] | None:
    """Court-space intersection of two landmark lines; None if parallel."""
    if a[0] == b[0]:
        return None
    x = a[1] if a[0] == "x" else b[1]
    y = a[1] if a[0] == "y" else b[1]
    return (x, y)


class CourtSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    length_ft: float = NCAA_LENGTH_FT
    width_ft: float = NCAA_WIDTH_FT

    def contains(self, x: float, y: float, margin_ft: float = 3.0) -> bool:
        return (
            -margin_ft <= x <= self.length_ft + margin_ft
            and -margin_ft <= y <= self.width_ft + margin_ft
        )


def project_points(homography: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    """Apply a 3x3 image->court homography to Nx2 pixel points, returning Nx2 court feet."""
    if homography.shape != (3, 3):
        raise ValueError(f"homography must be 3x3, got {homography.shape}")
    pts = np.asarray(points_px, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be Nx2, got {pts.shape}")
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([pts, ones]) @ homography.T
    w = homogeneous[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("degenerate homography: zero w component")
    return homogeneous[:, :2] / w


def bbox_ground_point(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Pixel point that maps to a player's court position: bottom-center of the box."""
    return ((x1 + x2) / 2.0, y2)
