"""Homography estimation from court-landmark line correspondences.

Pure numpy: fit lines to landmark-class mask pixels, intersect them into
point correspondences with known court geometry, solve image->court H via
normalized DLT, and score it by reprojection residual in feet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_CORRESPONDENCES = 4


@dataclass(frozen=True)
class FittedLine:
    """Homogeneous image line l=(a,b,c) with |(a,b)|=1; l.p=0 for points on it."""

    coeffs: np.ndarray
    n_pixels: int


def fit_line(points_xy: np.ndarray, tol_px: float = 4.0, iterations: int = 32) -> FittedLine:
    """Robust line fit: mini-RANSAC consensus, then total-least-squares refit.

    Plain TLS has a zero breakdown point — one far outlier blob can flip the
    principal axis entirely — so consensus selection must come first.
    Deterministic (fixed seed) so identical inputs give identical solves.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        raise ValueError(f"need Nx2 points with N>=2, got {pts.shape}")

    n = pts.shape[0]
    best_mask = np.ones(n, dtype=bool)
    if n > 2:
        rng = np.random.default_rng(0)
        best_count = 0
        for _ in range(iterations):
            i, j = rng.choice(n, size=2, replace=False)
            direction = pts[j] - pts[i]
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            normal = np.array([-direction[1], direction[0]]) / norm
            dists = np.abs((pts - pts[i]) @ normal)
            mask = dists <= tol_px
            count = int(mask.sum())
            if count > best_count:
                best_count = count
                best_mask = mask

    consensus = pts[best_mask] if best_mask.sum() >= 2 else pts
    centroid = consensus.mean(axis=0)
    _, _, vt = np.linalg.svd(consensus - centroid, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    line = np.array([normal[0], normal[1], -normal @ centroid])
    return FittedLine(coeffs=line, n_pixels=n)


def intersect(a: FittedLine, b: FittedLine, min_angle_deg: float = 8.0) -> np.ndarray | None:
    """Intersection point of two image lines; None if near-parallel."""
    cos_angle = abs(float(a.coeffs[:2] @ b.coeffs[:2]))
    if cos_angle > np.cos(np.deg2rad(min_angle_deg)):
        return None
    p = np.cross(a.coeffs, b.coeffs)
    if abs(p[2]) < 1e-9:
        return None
    return p[:2] / p[2]


def solve_homography(image_pts: np.ndarray, court_pts: np.ndarray) -> np.ndarray:
    """Normalized DLT for image->court H from >=4 point correspondences."""
    src = np.asarray(image_pts, dtype=np.float64)
    dst = np.asarray(court_pts, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < MIN_CORRESPONDENCES or src.shape[1] != 2:
        raise ValueError(f"need matching Nx2 with N>={MIN_CORRESPONDENCES}: {src.shape}")

    t_src = _normalizer(src)
    t_dst = _normalizer(dst)
    src_n = _apply(t_src, src)
    dst_n = _apply(t_dst, dst)

    rows = []
    for (x, y), (u, v) in zip(src_n, dst_n, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, s, vt = np.linalg.svd(np.asarray(rows))
    if s[-2] < 1e-10:
        raise ValueError("degenerate correspondence configuration")
    h_normalized = vt[-1].reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h_normalized @ t_src
    return h / h[2, 2]


def reprojection_residual(h: np.ndarray, image_pts: np.ndarray, court_pts: np.ndarray) -> float:
    """Mean distance (court units, i.e. feet) between projected and true points."""
    from montehall_cv.court import project_points

    projected = project_points(h, image_pts)
    return float(np.linalg.norm(projected - np.asarray(court_pts), axis=1).mean())


def correspondence_spread(court_pts: np.ndarray) -> float:
    """Area proxy of the correspondence set — guards against collinear solves."""
    pts = np.asarray(court_pts, dtype=np.float64)
    centered = pts - pts.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    return float(s[-1])


def _normalizer(pts: np.ndarray) -> np.ndarray:
    centroid = pts.mean(axis=0)
    mean_dist = np.linalg.norm(pts - centroid, axis=1).mean()
    scale = np.sqrt(2.0) / max(mean_dist, 1e-9)
    return np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]]
    )


def _apply(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))]) @ t.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]
