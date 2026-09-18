"""The two attribution joins (2026-07-09): shot windows from detector rim
boxes, and ball-gap interpolation feeding control samples."""

from __future__ import annotations

from montehall_cv.pipeline.controls import interpolate_ball
from montehall_cv.pipeline.run_shots import (
    _detector_hits,
    _merge_hits,
    _region_around,
)


def _ball(frame: int, ts: int, x: float, y: float) -> dict:
    return {"frame_idx": frame, "ts_ms": ts, "x1": x - 4, "y1": y - 4,
            "x2": x + 4, "y2": y + 4}


def _rim(frame: int, cx: float, cy: float, court_x: float | None = 5.0) -> dict:
    return {"frame_idx": frame, "x1": cx - 20, "y1": cy - 10,
            "x2": cx + 20, "y2": cy + 10, "court_x": court_x}


class TestDetectorHits:
    def test_ball_in_scaled_rim_region_hits(self) -> None:
        rims = {10: [_rim(10, 500, 200)]}
        hits = _detector_hits([_ball(10, 333, 500, 230)], rims)
        assert len(hits) == 1
        assert hits[0][2] == "left"  # court_x 5.0 < midline

    def test_right_end_from_court_projection(self) -> None:
        rims = {10: [_rim(10, 500, 200, court_x=88.0)]}
        hits = _detector_hits([_ball(10, 333, 500, 230)], rims)
        assert hits[0][2] == "right"

    def test_nearby_frame_rim_matches(self) -> None:
        # rim detected at frame 8, ball at frame 10 — within +-5 frames
        rims = {8: [_rim(8, 500, 200)]}
        assert len(_detector_hits([_ball(10, 333, 500, 230)], rims)) == 1

    def test_distant_frame_rim_does_not_match(self) -> None:
        rims = {2: [_rim(2, 500, 200)]}
        assert _detector_hits([_ball(10, 333, 500, 230)], rims) == []

    def test_ball_far_from_rim_never_hits(self) -> None:
        rims = {10: [_rim(10, 500, 200)]}
        assert _detector_hits([_ball(10, 333, 100, 600)], rims) == []

    def test_unlocalized_rim_still_windows_with_none_end(self) -> None:
        # A rim without a court projection must still trigger a window
        # (2026-07-12: the old skip dropped 5 of 8 real attempts on the
        # 07df788c benchmark); only the end label degrades to None.
        rims = {10: [_rim(10, 500, 200, court_x=None)]}
        hits = _detector_hits([_ball(10, 333, 500, 230)], rims)
        assert len(hits) == 1
        assert hits[0][2] is None

    def test_unlocalized_rim_inherits_end_from_nearby_projected_rim(self) -> None:
        # Same hoop seen 100 frames later with a valid projection.
        rims = {
            10: [_rim(10, 500, 200, court_x=None)],
            110: [_rim(110, 505, 202, court_x=88.0)],
        }
        hits = _detector_hits([_ball(10, 333, 500, 230)], rims)
        assert len(hits) == 1
        assert hits[0][2] == "right"

    def test_inheritance_ignores_far_away_projected_rim(self) -> None:
        # Projected rim at a different pixel spot (the other hoop) — no inherit.
        rims = {
            10: [_rim(10, 500, 200, court_x=None)],
            60: [_rim(60, 1400, 210, court_x=88.0)],
        }
        hits = _detector_hits([_ball(10, 333, 500, 230)], rims)
        assert hits[0][2] is None

    def test_inheritance_respects_frame_horizon(self) -> None:
        # Same spot but 400 frames away (> END_INHERIT_FRAMES) — no inherit.
        rims = {
            10: [_rim(10, 500, 200, court_x=None)],
            410: [_rim(410, 500, 200, court_x=88.0)],
        }
        hits = _detector_hits([_ball(10, 333, 500, 230)], rims)
        assert hits[0][2] is None

    def test_nearest_in_time_projection_wins(self) -> None:
        rims = {
            10: [_rim(10, 500, 200, court_x=None)],
            40: [_rim(40, 500, 200, court_x=5.0)],
            200: [_rim(200, 500, 200, court_x=88.0)],
        }
        hits = _detector_hits([_ball(10, 333, 500, 230)], rims)
        assert hits[0][2] == "left"


class TestMergeHits:
    def test_consecutive_hits_merge_into_one_window(self) -> None:
        region = _region_around(_rim(10, 500, 200), "left")
        hits = [(10, 333, "left", region), (15, 500, "left", region)]
        windows = _merge_hits(hits)
        assert len(windows) == 1
        assert windows[0]["n_ball_obs"] == 2

    def test_gap_or_end_change_starts_new_window(self) -> None:
        region = _region_around(_rim(10, 500, 200), "left")
        hits = [(10, 333, "left", region), (100, 3333, "left", region),
                (105, 3500, "right", region)]
        assert len(_merge_hits(hits)) == 3

    def test_none_projection_does_not_split_a_burst(self) -> None:
        # moving-camera homography dropout: left -> None -> left is ONE event
        region = _region_around(_rim(10, 500, 200), "left")
        hits = [(10, 333, "left", region), (40, 1333, None, region),
                (70, 2333, "left", region)]
        windows = _merge_hits(hits)
        assert len(windows) == 1
        assert windows[0]["court_end"] == "left"
        assert windows[0]["n_ball_obs"] == 3

    def test_merge_is_time_based_not_frame_based(self) -> None:
        # 120 frames apart at 60fps = 2s: one event regardless of frame gap
        region = _region_around(_rim(10, 500, 200), "left")
        hits = [(10, 333, "left", region), (130, 2333, "left", region)]
        assert len(_merge_hits(hits)) == 1

    def test_window_length_cap_starts_a_new_window(self) -> None:
        # a chain of 2s gaps must not snowball past MAX_WINDOW_MS
        region = _region_around(_rim(10, 500, 200), "left")
        hits = [(10 + 60 * i, 333 + 2000 * i, "left", region) for i in range(5)]
        windows = _merge_hits(hits)
        assert len(windows) == 2
        assert all(w["ts_end_ms"] - w["ts_start_ms"] <= 6000 for w in windows)

    def test_merged_region_is_the_union(self) -> None:
        r1 = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        r2 = {"x1": 150, "y1": 80, "x2": 260, "y2": 190}
        hits = [(10, 333, "left", r1), (20, 666, "left", r2)]
        windows = _merge_hits(hits)
        assert len(windows) == 1
        assert windows[0]["region"] == {"x1": 100, "y1": 80, "x2": 260, "y2": 200}


class TestBallInterpolation:
    def test_gap_filled_linearly(self) -> None:
        balls = [
            {"frame_idx": 0, "ts_ms": 0, "court_x": 10.0, "court_y": 20.0,
             "is_detected": True},
            {"frame_idx": 27, "ts_ms": 900, "court_x": 19.0, "court_y": 20.0,
             "is_detected": True},
        ]
        out = interpolate_ball(balls)
        synth = [b for b in out if not b["is_detected"]]
        assert len(synth) == 5  # 150ms steps inside a 900ms gap
        mid = synth[2]
        assert 13.0 < mid["court_x"] < 16.0
        assert mid["court_y"] == 20.0
        assert 0 < mid["frame_idx"] < 27

    def test_long_gap_stays_a_gap(self) -> None:
        balls = [
            {"frame_idx": 0, "ts_ms": 0, "court_x": 10.0, "court_y": 20.0,
             "is_detected": True},
            {"frame_idx": 90, "ts_ms": 3000, "court_x": 80.0, "court_y": 20.0,
             "is_detected": True},
        ]
        assert all(b["is_detected"] for b in interpolate_ball(balls))

    def test_output_stays_ts_ordered(self) -> None:
        balls = [
            {"frame_idx": 0, "ts_ms": 0, "court_x": 10.0, "court_y": 20.0,
             "is_detected": True},
            {"frame_idx": 12, "ts_ms": 400, "court_x": 14.0, "court_y": 20.0,
             "is_detected": True},
            {"frame_idx": 24, "ts_ms": 800, "court_x": 18.0, "court_y": 20.0,
             "is_detected": True},
        ]
        out = interpolate_ball(balls)
        assert [b["ts_ms"] for b in out] == sorted(b["ts_ms"] for b in out)


class TestFramePlanScaling:
    def test_short_window_keeps_floor(self) -> None:
        from montehall_cv.pipeline.run_shots import _frames_for_span
        assert _frames_for_span(200) == 5

    def test_long_window_samples_denser_capped(self) -> None:
        from montehall_cv.pipeline.run_shots import _frames_for_span
        assert _frames_for_span(3000) == 6
        assert _frames_for_span(6000) == 11
        assert _frames_for_span(60000) == 12
