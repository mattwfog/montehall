"""The PBP→anchors-truth bridge: aligned broadcast plays become the same
truth format the hand-labeled HS-clip files feed score_e2e_vs_anchors.py."""

import pytest

from montehall_cv.eval.pbp_truth import MATCH_PAD_MS, truth_from_plays


def _play(video_t, type_text="JumpShot", jersey: str | None = "3",
          scoring=True,
          score_value=2, shooting=True, play_id="p1", text="made Jumper."):
    return {
        "play_id": play_id,
        "video_t": video_t,
        "type_text": type_text,
        "text": text,
        "shooting_play": shooting,
        "scoring_play": scoring,
        "score_value": score_value,
        "shooter_jersey": jersey,
    }


def test_fg_rows_become_timed_events():
    doc = truth_from_plays(
        [_play(30.03, scoring=True), _play(55.5, scoring=False, jersey="21",
                                           score_value=3, play_id="p2")],
        "eid_test")
    assert doc["match_pad_ms"] == MATCH_PAD_MS
    assert [t["ts_ms"] for t in doc["timed_events"]] == [30030, 55500]
    assert doc["timed_events"][0] == {
        "ts_ms": 30030, "jersey": 3, "made": True, "points": 2,
        "play_id": "p1"}
    assert doc["timed_events"][1]["made"] is False
    assert doc["timed_events"][1]["jersey"] == 21
    # ESPN score_value on a miss is the attempt's worth; truth scores 0
    assert doc["timed_events"][1]["points"] == 0


def test_ft_rows_go_untimed_and_type_name_is_not_outcome():
    # ESPN type is 'MadeFreeThrow' for makes AND misses; scoring_play rules
    doc = truth_from_plays(
        [_play(80.0, type_text="MadeFreeThrow", scoring=False, score_value=0,
               text="missed Free Throw.")],
        "eid_test")
    assert doc["timed_events"] == []
    assert len(doc["untimed_truth"]) == 1
    assert doc["untimed_truth"][0]["what"].startswith("FT miss by #3")
    assert "video_t=80.0s" in doc["untimed_truth"][0]["what"]
    assert doc["untimed_truth"][0]["points"] == 0


def test_non_shooting_plays_and_bad_jerseys():
    doc = truth_from_plays(
        [_play(10.0, shooting=False),          # rebound/foul class: dropped
         _play(20.0, jersey=None),             # unrostered shooter
         _play(40.0, jersey="")],
        "eid_test")
    assert len(doc["timed_events"]) == 2
    assert all(t["jersey"] is None for t in doc["timed_events"])


def test_timed_events_sorted_by_video_time():
    doc = truth_from_plays([_play(90.0), _play(15.0, play_id="p0")],
                           "eid_test")
    assert [t["play_id"] for t in doc["timed_events"]] == ["p0", "p1"]


def test_team_is_omitted_pending_cluster_mapping():
    doc = truth_from_plays([_play(30.0)], "eid_test")
    assert "team" not in doc["timed_events"][0]


def test_box_truth_totals():
    # box level truth: FG vs FT split by type (the name lies about
    # outcome), assists counted over ALL plays' text
    doc = truth_from_plays(
        [_play(10.0, scoring=True),
         _play(20.0, scoring=False, score_value=3, play_id="p2"),
         _play(30.0, type_text="MadeFreeThrow", scoring=True, score_value=1,
               play_id="p3"),
         _play(40.0, type_text="MadeFreeThrow", scoring=False, score_value=0,
               play_id="p4"),
         _play(50.0, shooting=False, scoring=False, play_id="p5",
               text="Steal, assist by X.")],
        "eid_test")
    assert doc["box_truth"] == {
        "fga": 2, "fgm": 1, "fta": 2, "ftm": 1, "ast": 1}


def test_segment_mode_filters_and_shifts():
    doc = truth_from_plays(
        [_play(100.0, play_id="before"),
         _play(850.0, play_id="in1", scoring=True),
         _play(1400.0, type_text="MadeFreeThrow", scoring=True,
               score_value=1, play_id="ft_in"),
         _play(1439.0, shooting=False, play_id="ast_in",
               text="Layup, assist by X."),
         _play(1500.0, play_id="after")],
        "eid_test", start_s=830.0, end_s=1440.0)
    assert [t["play_id"] for t in doc["timed_events"]] == ["in1"]
    assert doc["timed_events"][0]["ts_ms"] == 20000  # 850 - 830
    assert "video_t=570.0s" in doc["untimed_truth"][0]["what"]
    # box over the window only: 1 FGA/FGM, 1 FTA/FTM, 1 ast
    assert doc["box_truth"] == {
        "fga": 1, "fgm": 1, "fta": 1, "ftm": 1, "ast": 1}
    assert "segment [830s, 1440s)" in doc["comment"]


def test_live_spans_from_reads():
    from montehall_cv.eval.pbp_truth import live_spans_from_reads
    reads = ([{"video_t": float(t), "conf": 0.9, "clock_s": 100.0}
              for t in (10, 12, 14, 16)]           # live run
             + [{"video_t": 20.0, "conf": 0.5, "clock_s": 100.0}]  # low conf
             + [{"video_t": 40.0, "conf": 0.9, "clock_s": None}]   # no clock
             + [{"video_t": float(t), "conf": 0.9, "clock_s": 50.0}
                for t in (60, 62)])                # second live run
    assert live_spans_from_reads(reads) == [[10.0, 16.0], [60.0, 62.0]]


def test_segment_clips_and_shifts_live_spans():
    doc = truth_from_plays([_play(850.0)], "eid_test", start_s=830.0,
                           end_s=1440.0,
                           live_spans_s=[[100.0, 200.0],     # before cut
                                         [820.0, 900.0],     # straddles start
                                         [1430.0, 1500.0]])  # straddles end
    assert doc["live_spans_ms"] == [[0, 70000], [600000, 610000]]


def test_empty_alignment_is_an_error():
    from pathlib import Path

    from montehall_cv.eval.pbp_truth import truth_from_alignment
    with pytest.raises(RuntimeError, match="no aligned plays"):
        truth_from_alignment(Path("/nonexistent/align_dir"))
