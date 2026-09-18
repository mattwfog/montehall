"""Homography fit from pitch vertices, points on pitch lines and points on pitch conics.

Generalises `geometry.calibrate` (vertices only, six-landmark guard) so a view with
few vertices but visible lines is solvable. Every residual is a distance in metres
on the pitch plane: vertex to its landmark, line point to its world line, conic
point to its world circle. Acceptance is the same idea as the vertex guard,
generalised: enough inlier primitives, an inlier majority, and a leave-one-primitive-
out error below 3 m, where a refit that is underdetermined without the dropped
primitive counts as unverifiable rather than as a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from soccerviz.core.geometry import project
from soccerviz.core.pitch_template import CONICS, NUM_LINES, line_coefficients

MIN_INLIER_PRIMITIVES = 4
MIN_INLIER_FRACTION = 0.55
MAX_LOO_ERROR_M = 3.0
LINE_INLIER_FRACTION = 0.5
# Relative singular value below which a refined Jacobian direction counts as unconstrained.
# Under the tangent-space gauge the 45 SNGS benchmark frames sit at 6e-5..2.6e-4 and a
# fully degenerate input (three collinear vertices on their own line) at 1.2e-7.
RANK_TOLERANCE = 3e-6
LOO_MAX_EVALUATIONS = 25
PRUNE_M = 5.0  # further than this from the seed: a wrong label or a noise ridge, never refined


@dataclass
class Primitive:
    """One constraint source: a vertex (kind 'point'), a line or a conic, with image points."""

    kind: str
    index: int
    image_points: np.ndarray  # (n, 2) source pixels
    world_point: np.ndarray | None = None  # for kind == 'point'


@dataclass
class PrimitiveCalibration:
    matrix: np.ndarray | None
    accepted: bool
    reason: str
    input_primitives: int
    inlier_primitives: int
    inlier_fraction: float | None
    median_residual_m: float | None
    median_loo_error_m: float | None
    loo_errors_m: dict = field(default_factory=dict)

    def diagnostics(self):
        return {k: v for k, v in self.__dict__.items() if k != "matrix"}


def _normaliser(points):
    centre = points.mean(axis=0)
    scale = np.sqrt(2.0) / max(np.mean(np.linalg.norm(points - centre, axis=1)), 1e-9)
    return np.array([[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1]])


def _dlt(primitives, lines_world):
    """Linear seed from vertices (two rows each) and line points (one row: l_w · G p = 0)."""
    image_points = np.vstack([p.image_points for p in primitives])
    world_points = np.vstack(
        [p.world_point[None] for p in primitives if p.kind == "point"] or [np.zeros((1, 2))]
    )
    t_image, t_world = _normaliser(image_points), _normaliser(world_points)
    rows = []
    for primitive in primitives:
        for point in primitive.image_points:
            p = t_image @ np.array([point[0], point[1], 1.0])
            if primitive.kind == "point":
                w = t_world @ np.array([*primitive.world_point, 1.0])
                rows.append(np.concatenate([np.zeros(3), -w[2] * p, w[1] * p]))
                rows.append(np.concatenate([w[2] * p, np.zeros(3), -w[0] * p]))
            elif primitive.kind == "line":
                line = np.linalg.inv(t_world).T @ lines_world[primitive.index]
                weight = 1.0 / np.sqrt(len(primitive.image_points))
                rows.append(weight * np.concatenate([line[0] * p, line[1] * p, line[2] * p]))
    if len(rows) < 8:
        return None
    _, _, vt = np.linalg.svd(np.asarray(rows))
    g = vt[-1].reshape(3, 3)
    matrix = np.linalg.inv(t_world) @ g @ t_image
    if not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-12:
        return None
    return matrix / matrix[2, 2]


def _residuals(matrix, primitives, lines_world, weighted):
    """Metre residuals per image point, optionally weighted by 1/sqrt(n) per primitive."""
    parts = []
    for primitive in primitives:
        world = project(primitive.image_points, matrix)
        if primitive.kind == "point":
            r = (world - primitive.world_point).ravel()
        elif primitive.kind == "line":
            line = lines_world[primitive.index]
            r = world @ line[:2] + line[2]
        else:
            _, centre, radius, _ = CONICS[primitive.index - NUM_LINES]
            r = np.linalg.norm(world - np.asarray(centre), axis=1) - radius
        r = np.where(np.isfinite(r), r, 1e3)
        parts.append(r / np.sqrt(len(primitive.image_points)) if weighted else r)
    return np.concatenate(parts)


def _gauge(seed):
    """Local 8-parameter chart around the seed: H = v + B·x with v the unit-norm seed and
    B an orthonormal basis of its complement. Only the projective scale is removed.
    Fixing h33 = 1 instead is singular whenever the image origin lies on the pitch
    horizon (h31·0 + h32·0 + h33 = 0), which broadcast views cross routinely; the
    2026-09-08 audit traced every `underdetermined` verdict on SNGS to that gauge."""
    unit = seed.ravel() / np.linalg.norm(seed)
    basis = np.linalg.svd(unit[None, :])[2][1:].T  # (9, 8)
    return lambda params: (unit + basis @ params).reshape(3, 3)


def _refine(seed, primitives, lines_world, threshold_m, max_nfev=None):
    unpack = _gauge(seed)
    result = least_squares(
        lambda x: _residuals(unpack(x), primitives, lines_world, weighted=True),
        np.zeros(8),
        loss="soft_l1",
        f_scale=threshold_m,
        x_scale="jac",
        max_nfev=max_nfev,
    )
    return unpack(result.x), _rank(result.jac)


def _refine_homography(seed, primitives, lines_world, threshold_m, max_nfev=None):
    """The default model, a free homography: (image→world matrix, fully determined).

    `calibrate_primitives` takes any callable with this signature as its `refine` step,
    so a constrained model (`pinhole_calibration`) shares the seed, prune, inlier and
    leave-one-out logic and differs only in what it fits.
    """
    matrix, rank = _refine(seed, primitives, lines_world, threshold_m, max_nfev)
    return matrix, rank >= 8


def _rank(jacobian):
    """Rank with each parameter column normalised: homography entries differ by ~1e5 in scale."""
    norms = np.linalg.norm(jacobian, axis=0)
    if not np.all(norms > 0):
        return 0
    singular = np.linalg.svd(jacobian / norms, compute_uv=False)
    return int(np.sum(singular > RANK_TOLERANCE * singular[0]))


def _primitive_errors(matrix, primitives, lines_world):
    """Median absolute metre residual per primitive."""
    errors = []
    for primitive in primitives:
        r = _residuals(matrix, [primitive], lines_world, weighted=False)
        if primitive.kind == "point":
            r = np.linalg.norm(r.reshape(-1, 2), axis=1)
        errors.append(float(np.median(np.abs(r))))
    return np.asarray(errors)


def _inlier_mask(matrix, primitives, lines_world, threshold_m):
    mask = []
    for primitive in primitives:
        r = _residuals(matrix, [primitive], lines_world, weighted=False)
        if primitive.kind == "point":
            mask.append(bool(np.linalg.norm(r) < threshold_m))
        else:
            mask.append(bool(np.mean(np.abs(r) < threshold_m) >= LINE_INLIER_FRACTION))
    return np.asarray(mask)


def _prune(seed, primitives, lines_world, limit_m=PRUNE_M):
    """Drop primitives the seed already places further than `limit_m` from their world
    line, circle or landmark: a mislabelled line or an 8-pixel noise ridge 28 m out would
    otherwise steer the robust refine. Keeps everything if too few would remain."""
    errors = _primitive_errors(seed, primitives, lines_world)
    kept = [p for p, error in zip(primitives, errors) if error < limit_m]
    return kept if len(kept) >= MIN_INLIER_PRIMITIVES else primitives


def _seed(primitives, lines_world, threshold_m):
    points = [p for p in primitives if p.kind == "point"]
    if len(points) >= 4:
        a = np.vstack([p.image_points for p in points])
        b = np.vstack([p.world_point[None] for p in points])
        matrix, _ = cv2.findHomography(
            a, b, cv2.RANSAC, threshold_m, maxIters=3000, confidence=0.995
        )
        if matrix is not None and np.isfinite(matrix).all() and abs(matrix[2, 2]) > 1e-12:
            return matrix / matrix[2, 2]
    return _dlt(primitives, lines_world)


def calibrate_primitives(primitives, threshold_m=1.5, refine=_refine_homography):
    """Fit image→world (top-left origin, metres) from vertices, line points and conic points.

    `refine(seed, primitives, lines_world, threshold_m, max_nfev)` fits the model from a
    homography seed and reports whether it was fully determined; the default is the
    free homography.
    """
    primitives = [p for p in primitives if len(p.image_points)]
    lines_world = line_coefficients()
    count = len(primitives)
    if count < MIN_INLIER_PRIMITIVES:
        return PrimitiveCalibration(None, False, "too_few_primitives", count, 0, None, None, None)
    seed = _seed(primitives, lines_world, threshold_m)
    if seed is None:
        return PrimitiveCalibration(None, False, "fit_failed", count, 0, None, None, None)
    primitives = _prune(seed, primitives, lines_world)
    matrix, determined = refine(seed, primitives, lines_world, threshold_m)
    if not determined:
        # The vertex-only RANSAC seed can fit four points exactly and strand the rest
        # (three halfway-line vertices are collinear); the DLT over every primitive
        # is the better start then.
        dlt = _dlt(primitives, lines_world)
        if dlt is not None:
            matrix, determined = refine(dlt, primitives, lines_world, threshold_m)
    if not determined:
        return PrimitiveCalibration(None, False, "underdetermined", count, 0, None, None, None)
    inliers = _inlier_mask(matrix, primitives, lines_world, threshold_m)
    kept = [p for p, keep in zip(primitives, inliers) if keep]
    fraction = float(inliers.mean())
    if len(kept) < MIN_INLIER_PRIMITIVES or fraction < MIN_INLIER_FRACTION:
        return PrimitiveCalibration(
            None, False, "inconsistent_primitives", count, len(kept), fraction, None, None
        )
    matrix, determined = refine(matrix, kept, lines_world, threshold_m)
    if not determined:
        return PrimitiveCalibration(
            None, False, "underdetermined", count, len(kept), fraction, None, None
        )
    residual = float(np.median(_primitive_errors(matrix, kept, lines_world)))
    loo = {}
    for i, dropped in enumerate(kept):
        rest = kept[:i] + kept[i + 1 :]
        if len(rest) < 3:
            continue
        candidate, candidate_determined = refine(
            matrix, rest, lines_world, threshold_m, max_nfev=LOO_MAX_EVALUATIONS
        )
        error = (
            float(_primitive_errors(candidate, [dropped], lines_world)[0])
            if candidate_determined
            else float("inf")
        )
        loo[f"{dropped.kind}:{dropped.index}"] = error
    loo_median = float(np.median(list(loo.values()))) if loo else None
    accepted = bool(
        loo_median is not None and np.isfinite(loo_median) and loo_median < MAX_LOO_ERROR_M
    )
    return PrimitiveCalibration(
        matrix if accepted else None,
        accepted,
        "accepted" if accepted else "inconsistent_primitives",
        count,
        len(kept),
        fraction,
        residual,
        loo_median,
        {k: (None if not np.isfinite(v) else round(v, 4)) for k, v in loo.items()},
    )
