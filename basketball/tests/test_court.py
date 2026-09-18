import numpy as np
import pytest

from montehall_cv.court import CourtSpec, bbox_ground_point, project_points


def test_identity_homography_passes_points_through():
    h = np.eye(3)
    pts = np.array([[10.0, 20.0], [47.0, 25.0]])
    out = project_points(h, pts)
    np.testing.assert_allclose(out, pts)


def test_scale_homography():
    h = np.diag([2.0, 3.0, 1.0])
    out = project_points(h, np.array([[5.0, 5.0]]))
    np.testing.assert_allclose(out, [[10.0, 15.0]])


def test_perspective_division():
    h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    out = project_points(h, np.array([[8.0, 4.0]]))
    np.testing.assert_allclose(out, [[4.0, 2.0]])


def test_bad_shapes_rejected():
    with pytest.raises(ValueError):
        project_points(np.eye(2), np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError):
        project_points(np.eye(3), np.array([1.0, 2.0]))


def test_court_contains_margin():
    court = CourtSpec()
    assert court.contains(47.0, 25.0)
    assert court.contains(-1.0, 25.0)
    assert not court.contains(120.0, 25.0)


def test_bbox_ground_point_is_bottom_center():
    assert bbox_ground_point(10.0, 20.0, 30.0, 80.0) == (20.0, 80.0)
