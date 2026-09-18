import numpy as np
import pytest

from montehall_cv.court import LANDMARK_LINES_FT, line_intersection_ft
from montehall_cv.pipeline.homography import (
    correspondence_spread,
    fit_line,
    intersect,
    reprojection_residual,
    solve_homography,
)


def test_fit_line_recovers_horizontal():
    xs = np.linspace(0, 100, 50)
    pts = np.stack([xs, np.full_like(xs, 7.0)], axis=1)
    line = fit_line(pts)
    a, b, c = line.coeffs
    assert abs(a) < 1e-6
    assert abs(b * 7.0 + c) < 1e-6


def test_fit_line_trims_outliers():
    xs = np.linspace(0, 100, 50)
    pts = np.stack([xs, np.full_like(xs, 7.0)], axis=1)
    pts = np.vstack([pts, [[50.0, 300.0]]])
    line = fit_line(pts)
    residual = abs(pts[0] @ line.coeffs[:2] + line.coeffs[2])
    assert residual < 0.5


def test_intersect_perpendicular():
    horizontal = fit_line(np.array([[0.0, 5.0], [10.0, 5.0], [20.0, 5.0]]))
    vertical = fit_line(np.array([[3.0, 0.0], [3.0, 10.0], [3.0, 20.0]]))
    pt = intersect(horizontal, vertical)
    np.testing.assert_allclose(pt, [3.0, 5.0], atol=1e-9)


def test_intersect_rejects_near_parallel():
    a = fit_line(np.array([[0.0, 0.0], [100.0, 1.0]]))
    b = fit_line(np.array([[0.0, 5.0], [100.0, 6.0]]))
    assert intersect(a, b) is None


def test_solve_homography_roundtrip():
    h_true = np.array([[1.2, 0.1, 5.0], [-0.05, 0.9, 12.0], [1e-4, 2e-4, 1.0]])
    court = np.array([[0.0, 0.0], [94.0, 0.0], [94.0, 50.0], [0.0, 50.0], [47.0, 25.0]])
    image = _project(np.linalg.inv(h_true), court)
    h = solve_homography(image, court)
    assert reprojection_residual(h, image, court) < 1e-6


def test_solve_rejects_too_few():
    with pytest.raises(ValueError):
        solve_homography(np.zeros((3, 2)), np.zeros((3, 2)))


def test_spread_flags_collinear():
    collinear = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])
    spread_pts = np.array([[0.0, 0.0], [94.0, 0.0], [94.0, 50.0], [0.0, 50.0]])
    assert correspondence_spread(collinear) < 1.0
    assert correspondence_spread(spread_pts) > 10.0


def test_landmark_intersections():
    assert line_intersection_ft(
        LANDMARK_LINES_FT["baseline_left"], LANDMARK_LINES_FT["sideline_top"]
    ) == (0.0, 50.0)
    assert (
        line_intersection_ft(
            LANDMARK_LINES_FT["baseline_left"], LANDMARK_LINES_FT["baseline_right"]
        )
        is None
    )


def _project(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))]) @ h.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]
