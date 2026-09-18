"""Pinhole-camera fit over the same vertex, line and conic evidence as `primitive_calibration`.

The free homography has eight parameters and no notion of a camera. The 2026-09-09
census found it fails or drifts on correct detections in two views: a halfway view
with one circle extreme out of frame, and an end view without the near touchline. The
published calibrators (PnLCalib, BroadTrack) fit a pinhole camera instead: square
pixels, zero skew, principal point at the image centre, so seven unknowns (focal,
rotation, position), or four when the position is held (a tripod-mounted broadcast
camera). This module plugs that model into `calibrate_primitives` as its refine step,
so seeding, pruning, inlier and leave-one-out acceptance are shared and only the model
differs.

World frame: the library's top-left pitch metres with z = x × y, so a camera above the
pitch has negative z. That is SoccerNet's convention too: PnLCalib's SNGS-060 cameras
decompose from their own homographies to exactly their recorded focal and position
(checked 2026-09-10, six frames, to 0.01 m and 0.1 px).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from soccerviz.core import primitive_calibration as pc
from soccerviz.core.geometry import project
from soccerviz.core.pitch_template import CONICS, LENGTH, LINES, NUM_LINES, WIDTH

# Main broadcast camera when nothing better is known: on the halfway line, 21 m behind
# the near touchline, 12 m up (BroadTrack's default, moved to the top-left frame).
DEFAULT_POSITION = np.array([LENGTH / 2, WIDTH + 21.0, -12.0])
FOCAL_GRID_PX = np.geomspace(600.0, 16000.0, 17)  # initial focal search when no seed decomposes
PRIOR_REUSE_M = 10.0  # a seed camera this close to a held position keeps its rotation and focal
DOWN = np.array([0.0, 0.0, 1.0])  # world z points into the ground


@dataclass(frozen=True)
class Camera:
    """Pinhole camera: image = K [R | t] X with K = diag(f, f, 1) about the image centre."""

    focal: float
    rotation: np.ndarray  # (3, 3) world → camera
    translation: np.ndarray  # (3,) camera coordinates of the world origin

    @property
    def position(self):
        return -self.rotation.T @ self.translation

    def world_to_image(self, width, height):
        r = self.rotation
        return intrinsics(self.focal, width, height) @ np.column_stack(
            [r[:, 0], r[:, 1], self.translation]
        )

    def image_to_world(self, width, height):
        try:
            return np.linalg.inv(self.world_to_image(width, height))
        except np.linalg.LinAlgError:
            return np.full((3, 3), np.nan)

    def describe(self):
        x, y, z = self.position
        return {
            "focal_px": round(float(self.focal), 1),
            "position_m": [round(float(x), 2), round(float(y), 2), round(float(-z), 2)],
        }


def principal_point(width, height):
    return (width - 1) / 2.0, (height - 1) / 2.0


def intrinsics(focal, width, height):
    cx, cy = principal_point(width, height)
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])


def focal_from_homography(world_to_image, width, height):
    """Focal length (px) of the pinhole camera behind a world→image homography, or None.

    With the principal point removed, G = [g1 g2 g3] = λ [r1 r2 t] / f-scaled, and the
    two constraints r1 ⊥ r2, |r1| = |r2| are linear in w = 1/f²; least squares over both.
    A homography that is not a pinhole projection of the plane gives w ≤ 0 (None).
    """
    cx, cy = principal_point(width, height)
    matrix = np.asarray(world_to_image, float)
    g = np.vstack([matrix[0] - cx * matrix[2], matrix[1] - cy * matrix[2], matrix[2]])
    h1, h2 = g[:, 0], g[:, 1]
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    denominator = float(a @ a)
    if not np.isfinite(denominator) or denominator < 1e-18:
        return None
    w = -float(a @ b) / denominator
    if not np.isfinite(w) or w <= 0:
        return None
    return 1.0 / np.sqrt(w)


def _orthonormal(matrix):
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        rotation = (u * np.array([1.0, 1.0, -1.0])) @ vt
    return rotation


def camera_from_homography(world_to_image, width, height):
    """Decompose a world→image homography into a pinhole camera, or None if it is not one."""
    focal = focal_from_homography(world_to_image, width, height)
    if focal is None:
        return None
    matrix = np.asarray(world_to_image, float)
    a = np.linalg.inv(intrinsics(focal, width, height)) @ matrix
    n1, n2 = np.linalg.norm(a[:, 0]), np.linalg.norm(a[:, 1])
    if not (np.isfinite(n1) and np.isfinite(n2)) or min(n1, n2) < 1e-12:
        return None
    scale = 2.0 / (n1 + n2)
    r1, r2, t = scale * a[:, 0], scale * a[:, 1], scale * a[:, 2]
    cx, cy = principal_point(width, height)
    centre = project([[cx, cy]], np.linalg.inv(matrix))[0]
    if not np.isfinite(centre).all():
        centre = np.array([LENGTH / 2, WIDTH / 2])
    if r1[2] * centre[0] + r2[2] * centre[1] + t[2] < 0:  # the visible pitch is in front
        r1, r2, t = -r1, -r2, -t
    rotation = _orthonormal(np.column_stack([r1, r2, np.cross(r1, r2)]))
    if not np.isfinite(rotation).all():
        return None
    return Camera(float(focal), rotation, t)


def look_at(position, target):
    """World→camera rotation with the optical axis through `target`, zero roll."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    forward = target - position
    forward = forward / max(np.linalg.norm(forward), 1e-12)
    right = np.cross(DOWN, forward)
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.vstack([right, down, forward])


def evidence_target(primitives):
    """World point the evidence is centred on: one anchor per primitive."""
    anchors = []
    for primitive in primitives:
        if primitive.kind == "point":
            anchors.append(np.asarray(primitive.world_point, float))
        elif primitive.kind == "line":
            _, p0, p1 = LINES[primitive.index]
            anchors.append((np.asarray(p0, float) + np.asarray(p1, float)) / 2)
        else:
            anchors.append(np.asarray(CONICS[primitive.index - NUM_LINES][1], float))
    xy = np.mean(anchors, axis=0) if anchors else np.array([LENGTH / 2, WIDTH / 2])
    return np.array([xy[0], xy[1], 0.0])


def _robust_cost(matrix, primitives, lines_world, threshold_m):
    residuals = pc._residuals(matrix, primitives, lines_world, weighted=True) / threshold_m
    return float(np.sum(np.sqrt(1.0 + residuals**2) - 1.0))


def initial_camera(primitives, lines_world, width, height, position, threshold_m, focal=None):
    """Camera at `position` steered at the evidence, focal from `focal` or a grid search."""
    rotation = look_at(position, evidence_target(primitives))
    translation = -rotation @ np.asarray(position, float)
    if focal is not None:
        return Camera(float(focal), rotation, translation)
    candidates = [Camera(float(candidate), rotation, translation) for candidate in FOCAL_GRID_PX]
    costs = [
        _robust_cost(camera.image_to_world(width, height), primitives, lines_world, threshold_m)
        for camera in candidates
    ]
    return candidates[int(np.argmin(costs))]


def _pack(camera, fixed_position):
    rotvec = Rotation.from_matrix(camera.rotation).as_rotvec()
    head = np.concatenate([[np.log(camera.focal)], rotvec])
    return head if fixed_position else np.concatenate([head, camera.translation])


def _unpack(params, position):
    rotation = Rotation.from_rotvec(params[1:4]).as_matrix()
    translation = params[4:7] if position is None else -rotation @ position
    return Camera(float(np.exp(params[0])), rotation, translation)


def _start(seed, primitives, lines_world, width, height, position, prior, threshold_m):
    """Where the refine starts: the seed's own camera when it is one, else steered."""
    seed_camera = None
    try:
        seed_camera = camera_from_homography(np.linalg.inv(seed), width, height)
    except np.linalg.LinAlgError:
        pass
    if position is None:
        if seed_camera is not None:
            return seed_camera
        return initial_camera(primitives, lines_world, width, height, prior, threshold_m)
    if seed_camera is not None and np.linalg.norm(seed_camera.position - position) < PRIOR_REUSE_M:
        return Camera(seed_camera.focal, seed_camera.rotation, -seed_camera.rotation @ position)
    focal = seed_camera.focal if seed_camera is not None else None
    return initial_camera(primitives, lines_world, width, height, position, threshold_m, focal)


def pinhole_refiner(width, height, position=None, prior=None):
    """A `calibrate_primitives` refine step fitting a pinhole camera.

    `position` (world metres, z negative above the pitch) holds the camera there and
    fits focal and rotation only; `prior` only initialises the free fit when the seed
    homography is not a pinhole projection.
    """
    held = None if position is None else np.asarray(position, float)
    fallback = DEFAULT_POSITION if prior is None else np.asarray(prior, float)

    def refine(seed, primitives, lines_world, threshold_m, max_nfev=None):
        start = _start(seed, primitives, lines_world, width, height, held, fallback, threshold_m)
        x0 = _pack(start, held is not None)
        result = least_squares(
            lambda x: pc._residuals(
                _unpack(x, held).image_to_world(width, height), primitives, lines_world, True
            ),
            x0,
            loss="soft_l1",
            f_scale=threshold_m,
            x_scale="jac",
            max_nfev=max_nfev,
        )
        camera = _unpack(result.x, held)
        return camera.image_to_world(width, height), pc._rank(result.jac) >= len(x0)

    return refine


def calibrate_pinhole(primitives, width, height, position=None, prior=None, threshold_m=1.5):
    """`calibrate_primitives` with the pinhole model; same acceptance contract."""
    return pc.calibrate_primitives(
        primitives, threshold_m, refine=pinhole_refiner(width, height, position, prior)
    )


def camera_of(calibration, width, height):
    """The camera behind an accepted pinhole calibration (exact: the matrix is one)."""
    if calibration.matrix is None:
        return None
    return camera_from_homography(np.linalg.inv(calibration.matrix), width, height)
