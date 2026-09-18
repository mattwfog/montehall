"""Vertices, lines and conics fit one homography; a halfway-line view is solvable; degeneracy is refused."""

import os
import time

import numpy as np

from soccerviz.core.geometry import pitch_landmarks, project
from soccerviz.core.pitch_template import NUM_LINES, line_names, project_primitives
from soccerviz.core.primitive_calibration import Primitive, calibrate_primitives

TRUTH = np.array([[9.0, -1.5, 120.0], [0.4, 6.5, 40.0], [0.0, 0.002, 1.0]])  # world → image
GROUND = np.linalg.inv(TRUTH)
HALFWAY = line_names().index("halfway")
CIRCLE = NUM_LINES  # centre circle channel


def _primitives(vertex_indices, primitive_indices, noise_px=1.0, seed=0, points_per=60):
    rng = np.random.default_rng(seed)
    world = pitch_landmarks()
    image = project(world, TRUTH)
    out = [
        Primitive("point", k, image[k : k + 1] + rng.normal(0, noise_px, (1, 2)), world[k])
        for k in vertex_indices
    ]
    samples = project_primitives(TRUTH, 2000, 2000)
    for index in primitive_indices:
        pts = samples[samples[:, 2] == index][:, :2]
        pts = pts[rng.choice(len(pts), min(points_per, len(pts)), replace=False)]
        kind = "line" if index < NUM_LINES else "conic"
        out.append(Primitive(kind, index, pts + rng.normal(0, noise_px, pts.shape)))
    return out


def _world_error(matrix, x_range=(0, 106)):
    grid = np.array([[x, y] for x in range(*x_range, 5) for y in range(0, 69, 17)], float)
    return np.linalg.norm(project(project(grid, TRUTH), matrix) - grid, axis=1)


def test_vertices_only_matches_the_truth():
    result = calibrate_primitives(_primitives(range(32), []))
    assert result.accepted and result.reason == "accepted"
    assert np.median(_world_error(result.matrix)) < 0.3


def test_halfway_view_with_five_vertices_is_solved_by_line_and_circle():
    """The SNGS-021 case: vertices 14, 15, 16, 31, 32 plus the halfway line and centre circle."""
    result = calibrate_primitives(_primitives([13, 14, 15, 30, 31], [HALFWAY, CIRCLE]))
    assert result.accepted, result.diagnostics()
    assert result.inlier_primitives == 7
    # Judged where such a camera sees players: the middle third of the pitch.
    assert np.median(_world_error(result.matrix, (35, 71))) < 0.5


def test_points_on_a_single_line_are_underdetermined():
    result = calibrate_primitives(_primitives([13, 14, 15], [HALFWAY]))
    assert not result.accepted and result.reason in {"underdetermined", "too_few_primitives"}


def test_outlier_vertex_is_excluded_not_absorbed():
    primitives = _primitives(range(32), [CIRCLE])
    primitives[4].image_points = primitives[4].image_points + 300.0
    result = calibrate_primitives(primitives)
    assert result.accepted and result.inlier_primitives == 32
    assert np.median(_world_error(result.matrix)) < 0.3


def test_wrong_line_label_is_refused_when_it_dominates():
    """Three vertices and the goal line relabelled as the halfway line cannot agree."""
    primitives = _primitives([0, 1, 5], [line_names().index("goal_left")])
    primitives[-1].index = HALFWAY
    result = calibrate_primitives(primitives)
    assert not result.accepted


# A wall-clock budget only means something on known hardware. The 3 s budget is
# the development-machine figure (the fit measures about 1.4 s there); a shared CI
# runner took 4.3 s for the same fit, so CI keeps a looser bound that still
# catches an order-of-magnitude regression.
FIT_BUDGET_S = 15.0 if os.environ.get("CI") else 3.0


def test_fit_is_fast_enough_for_frame_rate():
    primitives = _primitives(range(32), list(range(NUM_LINES + 3)))
    start = time.perf_counter()
    result = calibrate_primitives(primitives)
    assert result.accepted
    assert time.perf_counter() - start < FIT_BUDGET_S


# --- 2026-09-08 audit cases: the h33 gauge, noise ridges, the stranded fifth vertex.


def _primitives_under(truth, vertex_indices, primitive_indices, noise_px=1.0, seed=0, bounds=4000):
    rng = np.random.default_rng(seed)
    world = pitch_landmarks()
    image = project(world, truth)
    out = [
        Primitive("point", k, image[k : k + 1] + rng.normal(0, noise_px, (1, 2)), world[k])
        for k in vertex_indices
    ]
    samples = project_primitives(truth, bounds, bounds)
    for index in primitive_indices:
        pts = samples[samples[:, 2] == index][:, :2]
        pts = pts[rng.choice(len(pts), min(60, len(pts)), replace=False)]
        out.append(
            Primitive(
                "line" if index < NUM_LINES else "conic",
                index,
                pts + rng.normal(0, noise_px, pts.shape),
            )
        )
    return out


def test_image_origin_on_the_horizon_is_still_solvable():
    """SNGS-089 f41: the pitch horizon passes through image (0, 0), so h33 of the
    image-to-world homography is 0 and the h33 = 1 gauge cannot represent the answer."""
    ground = np.linalg.inv(TRUTH)
    y_horizon = -ground[2, 2] / ground[2, 1]  # horizon's image y at x = 0
    flip = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, y_horizon], [0.0, 0.0, 1.0]])
    truth = flip @ TRUTH
    assert abs(np.linalg.inv(truth)[2, 2]) < 1e-9 * np.abs(np.linalg.inv(truth)).max()
    result = calibrate_primitives(_primitives_under(truth, range(32), [HALFWAY, CIRCLE]))
    assert result.accepted, result.diagnostics()
    grid = np.array([[x, y] for x in range(0, 106, 5) for y in range(0, 69, 17)], float)
    assert (
        np.median(np.linalg.norm(project(project(grid, truth), result.matrix) - grid, axis=1)) < 0.3
    )


def test_noise_ridge_far_from_the_seed_is_pruned_not_refined():
    """SNGS-021 f81: an 8-point 'side_bottom' blob near the circle, 28 m from the real line."""
    primitives = _primitives(
        [13, 14, 15, 30, 31], [HALFWAY, CIRCLE, line_names().index("side_top")]
    )
    circle = next(p for p in primitives if p.kind == "conic")
    blob = circle.image_points[:8] + np.array([-40.0, 60.0])
    primitives.append(Primitive("line", line_names().index("side_bottom"), blob))
    result = calibrate_primitives(primitives)
    assert result.accepted, result.diagnostics()
    assert result.inlier_primitives == 8  # five vertices, three real primitives; the blob is gone
    assert np.median(_world_error(result.matrix, (35, 71))) < 0.5


def test_stranded_fifth_vertex_falls_back_to_the_dlt_seed():
    """SNGS-021 f31: RANSAC fits the four circle vertices exactly and leaves the sideline
    vertex 2.5 m out; the fit must still come from all five plus the lines."""
    primitives = _primitives(
        [13, 14, 15, 30, 31], [HALFWAY, CIRCLE, line_names().index("side_top")]
    )
    sideline = next(p for p in primitives if p.kind == "point" and p.index == 13)
    sideline.image_points = sideline.image_points + np.array([[0.0, 6.0]])
    result = calibrate_primitives(primitives)
    assert result.accepted, result.diagnostics()
    assert np.median(_world_error(result.matrix, (35, 71))) < 0.6
