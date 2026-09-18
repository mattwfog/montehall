"""A pinhole camera fits the same evidence as the free homography and solves the census views."""

import numpy as np

from soccerviz.core.geometry import pitch_landmarks, project
from soccerviz.core.pinhole_calibration import (
    Camera,
    calibrate_pinhole,
    camera_from_homography,
    camera_of,
    focal_from_homography,
    look_at,
)
from soccerviz.core.pitch_template import NUM_LINES, line_names, project_primitives
from soccerviz.core.primitive_calibration import Primitive, calibrate_primitives

WIDTH, HEIGHT = 1920, 1080
POSITION = np.array([52.5, 95.0, -14.0])  # main camera, 27 m behind the near touchline, 14 m up
HALFWAY = line_names().index("halfway")
SIDE_TOP = line_names().index("side_top")
CIRCLE = NUM_LINES


def _camera(target, focal):
    rotation = look_at(POSITION, target)
    return Camera(focal, rotation, -rotation @ POSITION)


def _primitives(truth, vertex_indices, primitive_indices, noise_px=1.0, seed=0, points_per=60):
    """In-frame evidence under a world→image camera matrix: vertices and sampled primitives."""
    rng = np.random.default_rng(seed)
    world = pitch_landmarks()
    image = project(world, truth)
    out = [
        Primitive("point", k, image[k : k + 1] + rng.normal(0, noise_px, (1, 2)), world[k])
        for k in vertex_indices
    ]
    samples = project_primitives(truth, WIDTH, HEIGHT)
    for index in primitive_indices:
        pts = samples[samples[:, 2] == index][:, :2]
        pts = pts[rng.choice(len(pts), min(points_per, len(pts)), replace=False)]
        kind = "line" if index < NUM_LINES else "conic"
        out.append(Primitive(kind, index, pts + rng.normal(0, noise_px, pts.shape)))
    return out


def _in_frame(truth, indices):
    image = project(pitch_landmarks(), truth)
    return [k for k in indices if 0 <= image[k, 0] < WIDTH and 0 <= image[k, 1] < HEIGHT]


def _world_error(matrix, truth, x_range):
    x0, x1 = x_range
    grid = np.array([[x, y] for x in range(x0, x1, 5) for y in range(0, 69, 17)], float)
    return np.linalg.norm(project(project(grid, truth), matrix) - grid, axis=1)


def test_camera_decomposition_round_trips():
    camera = _camera((60.0, 30.0, 0.0), 3000.0)
    matrix = camera.world_to_image(WIDTH, HEIGHT)
    focal = focal_from_homography(matrix, WIDTH, HEIGHT)
    assert focal is not None and abs(focal - 3000.0) < 1e-6
    recovered = camera_from_homography(matrix, WIDTH, HEIGHT)
    assert recovered is not None
    assert np.allclose(recovered.position, POSITION, atol=1e-6)
    assert np.allclose(recovered.rotation, camera.rotation, atol=1e-9)


def test_an_affine_map_is_not_a_pinhole_projection():
    assert focal_from_homography(np.diag([10.0, 10.0, 1.0]), WIDTH, HEIGHT) is None


def test_full_view_pinhole_matches_the_truth():
    truth = _camera((52.5, 34.0, 0.0), 1200.0).world_to_image(WIDTH, HEIGHT)
    vertices = _in_frame(truth, range(32))
    assert len(vertices) >= 12
    result = calibrate_pinhole(
        _primitives(truth, vertices, [SIDE_TOP, HALFWAY, CIRCLE]), WIDTH, HEIGHT
    )
    assert result.accepted, result.diagnostics()
    assert np.median(_world_error(result.matrix, truth, (0, 106))) < 0.3
    camera = camera_of(result, WIDTH, HEIGHT)
    assert camera is not None
    assert abs(camera.focal - 1200.0) / 1200.0 < 0.05
    assert np.linalg.norm(camera.position - POSITION) < 2.0


def test_halfway_view_with_one_circle_extreme_out_of_frame_is_solved_with_the_position_held():
    """The census's first geometry view: halfway line, far touchline and a partial centre
    circle, the left circle extreme (v30) outside the frame. The free homography has no
    conditioned scale along the pitch there; a camera held at its position has four
    unknowns and the same evidence over-determines them."""
    truth = _camera((60.0, 34.0, 0.0), 6500.0).world_to_image(WIDTH, HEIGHT)
    vertices = _in_frame(truth, [13, 14, 15, 30, 31])
    assert 30 not in vertices and 31 in vertices, vertices
    evidence = _primitives(truth, vertices, [SIDE_TOP, HALFWAY, CIRCLE])
    held = calibrate_pinhole(evidence, WIDTH, HEIGHT, position=POSITION)
    assert held.accepted, held.diagnostics()
    assert np.median(_world_error(held.matrix, truth, (35, 71))) < 0.5


def test_held_position_fit_matches_the_free_fit_on_a_full_view():
    """Holding the true position costs nothing where the free homography already solves."""
    truth = _camera((52.5, 34.0, 0.0), 1200.0).world_to_image(WIDTH, HEIGHT)
    evidence = _primitives(truth, _in_frame(truth, range(32)), [SIDE_TOP, HALFWAY, CIRCLE])
    held = calibrate_pinhole(evidence, WIDTH, HEIGHT, position=POSITION)
    free = calibrate_primitives(evidence)
    assert held.accepted and free.accepted
    assert np.median(_world_error(held.matrix, truth, (0, 106))) < 0.3
