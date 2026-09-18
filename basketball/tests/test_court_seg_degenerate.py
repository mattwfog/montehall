"""A degenerate homography on one sampled frame must yield an invalid
CourtSolve, never a raise — one bad frame previously killed whole games
(rung-30: 9 of 19 games died fatally at court.py project_points)."""

import numpy as np

import montehall_cv.pipeline.court_seg as court_seg
from montehall_cv.pipeline.court_seg import CLASSES, solve_frame


def _mask_with_four_lines() -> np.ndarray:
    mask = np.zeros((768, 768), dtype=np.uint8)
    mask[100:668, 100:103] = CLASSES.index("baseline_left")
    mask[100:668, 500:503] = CLASSES.index("midline")
    mask[100:103, 50:700] = CLASSES.index("sideline_top")
    mask[600:603, 50:700] = CLASSES.index("sideline_bottom")
    return mask


def test_mask_reaches_solver_and_solves():
    # guards the degenerate-case test against vacuity: this mask must make
    # it all the way through the solver with a valid H
    solve = solve_frame(_mask_with_four_lines(), 768, 768, frame_idx=0, ts_ms=0)
    assert solve.h is not None
    assert solve.n_points == 4


def test_degenerate_homography_yields_invalid_solve_not_raise(monkeypatch):
    def boom(h, src, dst):
        raise ValueError("degenerate homography: zero w component")

    monkeypatch.setattr(court_seg, "reprojection_residual", boom)
    solve = solve_frame(_mask_with_four_lines(), 768, 768, frame_idx=7, ts_ms=140)
    assert solve.h is None
    assert solve.residual_ft is None
    assert solve.n_points == 4
