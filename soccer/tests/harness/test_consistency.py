import copy
import math

import pytest

from soccerviz.harness.consistency import check_state, run_consistency


def player(track=1, identity=None, xy=None, evidence="p1", team=0):
    return {
        "tracklet_id": track,
        "identity": identity,
        "xy_m": xy,
        "evidence_id": evidence,
        "team": team,
    }


def frame(fid=1, time=0, shot=0, players=None):
    return {
        "frame_id": fid,
        "timestamp_s": time,
        "shot_id": shot,
        "calibration_usable": True,
        "players": players or [],
        "ball_hypotheses": [],
        "orientation": {},
        "possession": {"team": None, "status": "unknown", "hypotheses": []},
    }


def state(*frames):
    return {"schema": "match-state/v1", "frames": list(frames)}


def codes(report):
    return {v["code"] for v in report["violations"]}


def test_missing_channels_are_unknown_never_a_clean_bill_of_health():
    report = check_state(state({"frame_id": 1, "timestamp_s": 0}))
    assert report["status"] == "partial"
    for rule in [
        "simultaneous_tracks",
        "geometry",
        "ball_possession",
        "orientation",
        "source_alignment",
    ]:
        assert report["rules"][rule]["status"] == "skipped"
        assert report["rules"][rule]["skip_reasons"]
    assert report["blocking_frame_ids"] == []
    empty = check_state(state())
    assert all(r["skipped"] for r in empty["rules"].values())


def test_simultaneous_duplicate_tracks_and_identities_name_evidence_owner():
    a = player(3, "Alex", [5, 6], "p1")
    b = player(3, "Alex", [7, 8], "p2")
    report = check_state(state(frame(players=[a, b])))
    assert {"duplicate_simultaneous_tracklet", "duplicate_simultaneous_identity"} <= codes(report)
    assert report["blocking_frame_ids"] == [1]
    issue = next(v for v in report["violations"] if v["rule"] == "simultaneous_tracks")
    assert issue["owner"] == "players" and issue["key"] == "shot:0:tracklet:3"
    assert set(issue["evidence_ids"]) == {"p1", "p2"}


def test_numeric_tracks_are_namespaced_by_shot_and_motion_never_bridges_cuts():
    a = frame(1, 0, 0, [player(3, "Alex", [0, 0])])
    b = frame(2, 0.2, 1, [player(3, "Alex", [1000, 1000])])
    report = check_state(state(a, b), config={"max_speed_m_s": 1})
    assert not report["violations"]
    assert report["break_before_frame_ids"] == [2]
    assert report["boundaries"][0]["reason"] == "camera_cut"


@pytest.mark.parametrize("xy", [[math.nan, 2], [1, math.inf], [1], ["1", 2]])
def test_invalid_geometry_blocks_its_frame(xy):
    report = check_state(state(frame(players=[player(xy=xy)])))
    assert "invalid_position" in codes(report) and report["blocking_frame_ids"] == [1]


def test_rejected_or_nonunique_source_calibration_cannot_emit_positions():
    f = frame(players=[player(xy=[2, 3])])
    f["calibration_usable"] = False
    report = check_state(state(f))
    assert "position_with_rejected_calibration" in codes(report)
    f["calibration_usable"] = True
    report = check_state(state(f), {"calibration_hypotheses": []})
    assert "position_without_unique_accepted_source_calibration" in codes(report)


def test_pitch_bounds_must_be_explicit_and_coordinate_consistent():
    s = state(frame(players=[player(xy=[-1, 20])]))
    report = check_state(s)
    assert report["rules"]["pitch_bounds"]["status"] == "skipped"
    report = check_state(
        s, config={"pitch_bounds_m": {"x_min": 0, "x_max": 104, "y_min": 0, "y_max": 68}}
    )
    assert "outside_explicit_pitch_bounds" in codes(report)
    centered = check_state(
        s, config={"pitch_bounds_m": {"x_min": -52, "x_max": 52, "y_min": -34, "y_max": 34}}
    )
    assert "outside_explicit_pitch_bounds" not in codes(centered)


def test_provider_positions_skip_camera_homography_requirement():
    s = state(frame(players=[player(xy=[2, 3])]))
    s.update({"source_kind": "provider", "position_origin": "provider_tracking"})
    report = check_state(s, {"schema": "provider-evidence/v1", "calibration_hypotheses": []})
    assert "position_without_unique_accepted_source_calibration" not in codes(report)
    assert report["rules"]["calibration_gate"]["status"] == "skipped"


def test_source_frame_mapping_cannot_be_silently_changed():
    s = state(frame(1, 0.2, 2))
    report = check_state(s, {"frames": [{"frame_id": 1, "timestamp_s": 0.1, "shot_id": 1}]})
    assert "source_clock_or_camera_mismatch" in codes(report)


def test_time_gaps_break_candidates_and_time_regression_blocks():
    s = state(frame(1, 0), frame(2, 0.2), frame(3, 2), frame(4, 1))
    report = run_consistency({"sample_period_s": 0.2}, {"state": s}, [])
    assert report["break_before_frame_ids"] == [3, 4]
    assert report["blocking_frame_ids"] == [4]
    assert "nonincreasing_source_time" in codes(report)


def test_ambiguous_ball_cannot_support_unreviewed_possession_but_review_can():
    f = frame(players=[player(xy=[1, 1])])
    f["ball_hypotheses"] = [
        {"evidence_id": "b1", "xy_m": [1, 2]},
        {"evidence_id": "b2", "xy_m": [8, 9]},
    ]
    f["possession"] = {
        "team": 0,
        "status": "hypothesis",
        "hypotheses": [
            {"value": 0, "source": "nearest_player_proxy", "evidence_ids": ["p1", "b1"]}
        ],
    }
    assert "unsupported_or_ambiguous_possession" in codes(check_state(state(f)))
    f["possession"]["status"] = "reviewed"
    f["possession"]["hypotheses"].append({"value": 0, "source": "review", "evidence_ids": ["p1"]})
    assert "unsupported_or_ambiguous_possession" not in codes(check_state(state(f)))


def test_single_ball_without_matching_local_player_evidence_is_unsupported():
    f = frame(players=[player(xy=[1, 1])])
    f["ball_hypotheses"] = [{"evidence_id": "b1", "xy_m": [1, 2]}]
    f["possession"] = {
        "team": 0,
        "status": "hypothesis",
        "hypotheses": [
            {"value": 0, "source": "nearest_player_proxy", "evidence_ids": ["wrong-player", "b1"]}
        ],
    }
    assert "unsupported_or_ambiguous_possession" in codes(check_state(state(f)))


def test_opposite_orientation_checked_only_with_support():
    f = frame()
    f["orientation"] = {
        str(t): {"selected": 1, "hypotheses": [{"value": 1, "source": "unresolved"}]}
        for t in (0, 1)
    }
    assert check_state(state(f))["rules"]["orientation"]["status"] == "skipped"
    for item in f["orientation"].values():
        item["hypotheses"] = [{"value": 1, "source": "review", "evidence_ids": ["p1"]}]
    assert "opponents_share_attacking_direction" in codes(check_state(state(f)))


def test_optional_motion_threshold_is_diagnostic_and_inputs_stay_unchanged():
    s = state(frame(1, 0, players=[player(xy=[0, 0])]), frame(2, 0.2, players=[player(xy=[10, 0])]))
    original = copy.deepcopy(s)
    report = check_state(s, config={"max_speed_m_s": 20})
    assert s == original
    assert "configured_speed_diagnostic_exceeded" in codes(report)
    assert report["blocking_frame_ids"] == []
    issue = report["violations"][0]
    assert issue["details"]["speed_m_s"] == 50 and not issue["blocking"]


@pytest.mark.parametrize("position_status", ["estimated", "unknown"])
def test_estimated_provider_ball_cannot_masquerade_as_observed_possession_support(position_status):
    f = frame(players=[player(xy=[1, 1])])
    f.update({"source_kind": "provider", "position_origin": "provider_tracking"})
    f["ball_hypotheses"] = [
        {"evidence_id": "b1", "xy_m": [1, 2], "position_status": position_status}
    ]
    f["possession"] = {
        "team": 0,
        "status": "hypothesis",
        "hypotheses": [
            {"value": 0, "source": "nearest_player_proxy", "evidence_ids": ["p1", "b1"]}
        ],
    }
    assert "unsupported_or_ambiguous_possession" in codes(check_state(state(f)))
