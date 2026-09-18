"""Pitch primitives agree with the vertex table, mirror cleanly and project only in front of the camera."""

import numpy as np

from soccerviz.core.geometry import pitch_landmarks, project
from soccerviz.core.pitch_template import (
    NUM_CONICS,
    NUM_LINES,
    NUM_PRIMITIVES,
    line_coefficients,
    mirror_primitive_pairs,
    orient_world_to_image,
    project_primitives,
    sample_world_primitives,
)


def test_every_vertex_lies_on_at_least_one_primitive():
    lines = line_coefficients()
    samples = sample_world_primitives(0.05)
    for k, vertex in enumerate(pitch_landmarks()):
        if k in (8, 21):  # the penalty spots lie on no line or conic
            continue
        on_line = np.min(np.abs(lines[:, :2] @ vertex + lines[:, 2])) < 1e-9
        near_sample = np.min(np.linalg.norm(samples[:, :2] - vertex, axis=1)) < 0.06
        assert on_line or near_sample, vertex


def test_counts_and_mirror_pairs():
    assert (NUM_LINES, NUM_CONICS, NUM_PRIMITIVES) == (17, 3, 20)
    pairs = mirror_primitive_pairs()
    swapped = {i for pair in pairs for i in pair}
    assert len(pairs) == 8 and len(swapped) == 16
    fixed = set(range(NUM_PRIMITIVES)) - swapped
    assert fixed == {0, 1, 4, NUM_LINES}  # touchlines, halfway line, centre circle


def test_projection_keeps_only_in_frame_points_in_front_of_the_camera():
    truth = np.array([[9.0, -1.5, 120.0], [0.4, 6.5, 40.0], [0.0, 0.002, 1.0]])
    oriented = orient_world_to_image(-truth, pitch_landmarks())
    assert np.allclose(oriented, truth)
    samples = project_primitives(oriented, 640, 480)
    assert samples.shape[1] == 3 and len(samples) > 100
    assert (samples[:, 0] >= 0).all() and (samples[:, 0] < 640).all()
    assert (samples[:, 1] >= 0).all() and (samples[:, 1] < 480).all()
    world = project(samples[:, :2], np.linalg.inv(truth))
    assert (world[:, 0] > -1).all() and (world[:, 0] < 106).all()
