"""Pitch line segments and conics in metres, and their projection into an image.

Companion to `geometry.pitch_landmarks()`: the 32 vertices are the intersections of
these primitives. A calibrator that only predicts vertices cannot solve a halfway-line
view (SNGS-021 has five vertices in frame); lines and conics give constraints the
vertices cannot. Rendering targets from a labelled image's own fitted homography
means line supervision costs no new labels.
"""

from __future__ import annotations

import numpy as np

from soccerviz.core.geometry import project

LENGTH, WIDTH = 105.0, 68.0
PENALTY, GOAL_AREA, SPOT, RADIUS = 16.5, 5.5, 11.0, 9.15
PY0, PY1 = (WIDTH - 40.32) / 2, (WIDTH + 40.32) / 2
GY0, GY1 = (WIDTH - 18.32) / 2, (WIDTH + 18.32) / 2
SAMPLE_STEP_M = 0.25

LINES = (
    ("side_top", (0, 0), (LENGTH, 0)),
    ("side_bottom", (0, WIDTH), (LENGTH, WIDTH)),
    ("goal_left", (0, 0), (0, WIDTH)),
    ("goal_right", (LENGTH, 0), (LENGTH, WIDTH)),
    ("halfway", (LENGTH / 2, 0), (LENGTH / 2, WIDTH)),
    ("penalty_left_top", (0, PY0), (PENALTY, PY0)),
    ("penalty_left_bottom", (0, PY1), (PENALTY, PY1)),
    ("penalty_left_front", (PENALTY, PY0), (PENALTY, PY1)),
    ("penalty_right_top", (LENGTH - PENALTY, PY0), (LENGTH, PY0)),
    ("penalty_right_bottom", (LENGTH - PENALTY, PY1), (LENGTH, PY1)),
    ("penalty_right_front", (LENGTH - PENALTY, PY0), (LENGTH - PENALTY, PY1)),
    ("goal_area_left_top", (0, GY0), (GOAL_AREA, GY0)),
    ("goal_area_left_bottom", (0, GY1), (GOAL_AREA, GY1)),
    ("goal_area_left_front", (GOAL_AREA, GY0), (GOAL_AREA, GY1)),
    ("goal_area_right_top", (LENGTH - GOAL_AREA, GY0), (LENGTH, GY0)),
    ("goal_area_right_bottom", (LENGTH - GOAL_AREA, GY1), (LENGTH, GY1)),
    ("goal_area_right_front", (LENGTH - GOAL_AREA, GY0), (LENGTH - GOAL_AREA, GY1)),
)
# (name, centre, radius, x-range of the drawn arc); the penalty arcs stop at the box front.
CONICS = (
    ("centre_circle", (LENGTH / 2, WIDTH / 2), RADIUS, (0.0, LENGTH)),
    ("penalty_arc_left", (SPOT, WIDTH / 2), RADIUS, (PENALTY, LENGTH)),
    ("penalty_arc_right", (LENGTH - SPOT, WIDTH / 2), RADIUS, (0.0, LENGTH - PENALTY)),
)
NUM_LINES, NUM_CONICS = len(LINES), len(CONICS)
NUM_PRIMITIVES = NUM_LINES + NUM_CONICS


def line_names():
    return [name for name, _, _ in LINES]


def conic_names():
    return [name for name, _, _, _ in CONICS]


def primitive_names():
    """Channel order of a line-aware calibrator: the 17 segments, then the 3 conics."""
    return line_names() + conic_names()


def line_coefficients():
    """(NUM_LINES, 3) homogeneous world lines a x + b y + c = 0 with unit normals."""
    rows = []
    for _, p0, p1 in LINES:
        (x0, y0), (x1, y1) = p0, p1
        a, b = y1 - y0, x0 - x1
        norm = float(np.hypot(a, b))
        rows.append((a / norm, b / norm, -(a * x0 + b * y0) / norm))
    return np.asarray(rows, dtype=float)


def sample_world_primitives(step=SAMPLE_STEP_M):
    """Dense world samples: (N, 3) of x, y, primitive index; conics follow the lines."""
    samples = []
    for index, (_, p0, p1) in enumerate(LINES):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        count = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step)) + 1)
        t = np.linspace(0.0, 1.0, count)[:, None]
        samples.append(np.column_stack([p0 + t * (p1 - p0), np.full(count, index)]))
    for index, (_, centre, radius, (x_min, x_max)) in enumerate(CONICS):
        count = max(8, int(np.ceil(2 * np.pi * radius / step)))
        angle = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
        xy = np.column_stack(
            [centre[0] + radius * np.cos(angle), centre[1] + radius * np.sin(angle)]
        )
        keep = (xy[:, 0] >= x_min - 1e-9) & (xy[:, 0] <= x_max + 1e-9)
        samples.append(np.column_stack([xy[keep], np.full(int(keep.sum()), NUM_LINES + index)]))
    return np.vstack(samples)


def mirror_primitive_pairs():
    """Primitive index pairs swapped by a horizontal image flip (mirror across halfway)."""
    pairs = []
    for i, (_, p0, p1) in enumerate(LINES):
        mirrored = {(LENGTH - p0[0], p0[1]), (LENGTH - p1[0], p1[1])}
        for j, (_, q0, q1) in enumerate(LINES):
            if i < j and {tuple(map(float, q0)), tuple(map(float, q1))} == mirrored:
                pairs.append((i, j))
    for i, (_, centre, _, _) in enumerate(CONICS):
        for j, (_, other, _, _) in enumerate(CONICS):
            if i < j and abs(LENGTH - centre[0] - other[0]) < 1e-9 and centre[1] == other[1]:
                pairs.append((NUM_LINES + i, NUM_LINES + j))
    return pairs


def project_primitives(world_to_image, width, height, step=SAMPLE_STEP_M):
    """In-frame image samples (M, 3) of x, y, primitive index, in front of the camera."""
    samples = sample_world_primitives(step)
    homogeneous = (
        np.column_stack([samples[:, :2], np.ones(len(samples))])
        @ np.asarray(world_to_image, float).T
    )
    in_front = homogeneous[:, 2] > 1e-8
    xy = project(samples[:, :2], np.asarray(world_to_image, float))
    inside = (
        in_front
        & np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    return np.column_stack([xy[inside], samples[inside, 2]]).astype(np.float32)


def orient_world_to_image(world_to_image, world_reference_points):
    """Scale a world→image homography so reference points on the visible pitch have w > 0.

    A homography's overall sign is arbitrary; only after fixing it does `w > 0`
    separate the pitch in front of the camera from its continuation behind it.
    """
    matrix = np.asarray(world_to_image, float)
    reference = np.asarray(world_reference_points, float).reshape(-1, 2)
    w = (np.column_stack([reference, np.ones(len(reference))]) @ matrix.T)[:, 2]
    if len(w) and np.median(w) < 0:
        return -matrix
    return matrix
