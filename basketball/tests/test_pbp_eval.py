"""Pure-function coverage for the PBP eval layer (clock parse, matching, box)."""

from montehall_cv.eval.clock_ocr import parse_clock, scaled_clock_box
from montehall_cv.eval.espn_pbp import matchup_from_title
from montehall_cv.eval.pbp_align import build_timeline, video_time


class _Read:
    def __init__(self, video_t, clock_s, conf=0.95):
        self.video_t = video_t
        self.clock_s = clock_s
        self.conf = conf
        self.text = "x"


def test_parse_clock():
    assert parse_clock("19:45") == 19 * 60 + 45
    assert parse_clock("0:03") == 3
    assert parse_clock("45.7") == 45.7
    assert parse_clock("95.0") is None  # >60 as tenths display is nonsense
    assert parse_clock("12:75") is None
    assert parse_clock("FOULS") is None


def test_scaled_clock_box_identity_at_720p():
    assert scaled_clock_box(1280, 720) == (596, 592, 680, 634)
    x1, y1, x2, y2 = scaled_clock_box(1920, 1080)
    assert (x1, y1, x2, y2) == (894, 888, 1020, 951)


def test_build_timeline_period_break():
    reads = [
        _Read(10.0, 100.0),
        _Read(12.0, 98.0),
        _Read(14.0, 1200.0),  # clock jumped up >4min -> period 2
        _Read(16.0, 1198.0),
        _Read(18.0, 30.0, conf=0.5),  # below floor, dropped
    ]
    timeline = build_timeline(reads)
    assert [(p, ck) for p, ck, _ in timeline] == [
        (1, 100.0),
        (1, 98.0),
        (2, 1200.0),
        (2, 1198.0),
    ]


def test_video_time_tolerance():
    by_period = {1: [(100.0, 50.0), (90.0, 62.0)]}
    assert video_time(by_period, 1, 99.0) == (50.0, 1.0)
    assert video_time(by_period, 1, 60.0) is None  # gap 30s > tolerance
    assert video_time(by_period, 2, 99.0) is None


def test_matchup_from_title():
    name = "Cal vs. Florida State Full Game Replay ｜ 2026 ACC [id].mp4"
    assert matchup_from_title(name) == ("California", "Florida State")
    assert matchup_from_title("garbage.mp4") is None


# The three-level pbp_score scorer was retired 2026-08-26 — its box level
# lives on as score_e2e_vs_anchors' box block fed by pbp_truth's box_truth
# (tests in test_pbp_truth.py); event/attribution levels are covered by
# the e2e scorer itself.
