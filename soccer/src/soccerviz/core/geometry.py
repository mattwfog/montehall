"""Robust pitch projection with explicit rejection and diagnostics."""

from dataclasses import dataclass

import cv2
import numpy as np


def pitch_landmarks(length=105.0, width=68.0):
    """32 semantic keypoints in Roboflow soccer model order, metres on assumed pitch.

    Index j is vertex j+1 of the roboflow/sports SoccerPitchConfiguration, and that
    is also the keypoint index order of the Roboflow Universe
    football-field-detection exports and of every model trained on them. The
    exports' keypoint NAME list ('01'..'13', '15'..'18', '20'..'32', '14', '19') is
    not the index semantics: verified 2026-09-08 by homography consistency on all
    222 v12 training images (index order 222/222 accepted, 0.27 m median residual;
    the name order 74/222) and by the 45-frame YOLO control (38 vs 2 accepted).

    Regulation penalty/goal areas replace the demo's stylized dimensions. Actual
    pitch dimensions must be supplied for metric validation on a new recording.
    """
    p, g, spot, radius = 16.5, 5.5, 11.0, 9.15
    py0, py1 = (width - 40.32) / 2, (width + 40.32) / 2
    gy0, gy1 = (width - 18.32) / 2, (width + 18.32) / 2
    return np.array(
        [
            (0, 0),
            (0, py0),
            (0, gy0),
            (0, gy1),
            (0, py1),
            (0, width),
            (g, gy0),
            (g, gy1),
            (spot, width / 2),
            (p, py0),
            (p, gy0),
            (p, gy1),
            (p, py1),
            (length / 2, 0),
            (length / 2, width / 2 - radius),
            (length / 2, width / 2 + radius),
            (length / 2, width),
            (length - p, py0),
            (length - p, gy0),
            (length - p, gy1),
            (length - p, py1),
            (length - spot, width / 2),
            (length - g, gy0),
            (length - g, gy1),
            (length, 0),
            (length, py0),
            (length, gy0),
            (length, gy1),
            (length, py1),
            (length, width),
            (length / 2 - radius, width / 2),
            (length / 2 + radius, width / 2),
        ],
        dtype=float,
    )


def project(points, matrix):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    good = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 2]) > 1e-8)
    result = np.full((len(points), 2), np.nan)
    result[good] = homogeneous[good, :2] / homogeneous[good, 2:]
    return result


@dataclass
class Calibration:
    matrix: np.ndarray | None
    accepted: bool
    reason: str
    input_points: int
    inliers: int
    median_residual_m: float | None
    median_loo_error_m: float | None

    def diagnostics(self):
        return {k: v for k, v in self.__dict__.items() if k != "matrix"}


def calibrate(image_points, world_points, threshold_m=1.5):
    image_points, world_points = np.asarray(image_points, float), np.asarray(world_points, float)
    valid = np.isfinite(image_points).all(axis=1) & np.isfinite(world_points).all(axis=1)
    a, b = image_points[valid], world_points[valid]
    if len(a) < 6:
        return Calibration(None, False, "fewer_than_six_landmarks", len(a), 0, None, None)
    if (
        np.linalg.matrix_rank(a - a.mean(axis=0)) < 2
        or np.linalg.matrix_rank(b - b.mean(axis=0)) < 2
    ):
        return Calibration(None, False, "collinear_landmarks", len(a), 0, None, None)
    matrix, mask = cv2.findHomography(
        a, b, cv2.RANSAC, threshold_m, maxIters=3000, confidence=0.995
    )
    if matrix is None or mask is None:
        return Calibration(None, False, "fit_failed", len(a), 0, None, None)
    keep = mask.ravel().astype(bool)
    residual = np.linalg.norm(project(a[keep], matrix) - b[keep], axis=1)
    errors = []
    for i in np.flatnonzero(keep):
        select = keep.copy()
        select[i] = False
        if select.sum() < 4:
            continue
        candidate, _ = cv2.findHomography(a[select], b[select], 0)
        if candidate is not None:
            errors.append(float(np.linalg.norm(project(a[i : i + 1], candidate)[0] - b[i])))
    median = float(np.median(residual)) if len(residual) else None
    loo = float(np.median(errors)) if errors and np.isfinite(errors).all() else None
    accepted = bool(keep.sum() >= 6 and keep.mean() >= 0.55 and loo is not None and loo < 3.0)
    return Calibration(
        matrix if accepted else None,
        accepted,
        "accepted" if accepted else "inconsistent_landmarks",
        len(a),
        int(keep.sum()),
        median,
        loo,
    )


def synthetic_calibration_benchmark():
    rng = np.random.default_rng(20260908)
    world = pitch_landmarks()
    truth = np.array([[10, 2, 200], [0.5, 7, 90], [0.002, 0.004, 1.0]])
    image = project(world, truth)
    errors, accepted = [], 0
    holdout = rng.uniform([0, 0], [105, 68], (100, 2))
    for _ in range(50):
        noisy = image + rng.normal(0, 1.5, image.shape)
        noisy[rng.choice(len(noisy), 6, replace=False)] += rng.normal(0, 100, (6, 2))
        result = calibrate(noisy, world)
        if result.accepted:
            accepted += 1
            errors.extend(
                np.linalg.norm(project(project(holdout, truth), result.matrix) - holdout, axis=1)
            )
    return {
        "benchmark": "synthetic camera, 1.5px noise and 6/32 corrupted landmarks",
        "trials": 50,
        "accepted": accepted,
        "median_holdout_projection_error_m": float(np.median(errors)) if errors else None,
        "p95_holdout_projection_error_m": float(np.quantile(errors, 0.95)) if errors else None,
        "real_video_accuracy": "unmeasured: manually labeled image/pitch correspondences required",
    }
