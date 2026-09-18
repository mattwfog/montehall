import numpy as np
import pytest

from soccerviz.vision.video_safety import BallTrack, CameraGuard, jersey_consensus


def test_camera_rejects_catastrophic_finite_projection_and_never_reuses_it():
    guard = CameraGuard()
    feet = np.array([[100, 100], [200, 100], [100, 200], [200, 200]])
    good = np.diag([0.1, 0.1, 1])
    assert guard.check({"accepted": True, "homography": good}, feet, 0)["accepted"]
    bad = good.copy()
    bad[0, 2] = 500
    rejected = guard.check({"accepted": True, "homography": bad}, feet, 0.2)
    assert rejected["homography"] is None
    assert rejected["reason"] == "implausible_person_projection"
    assert not guard.check({"accepted": False, "reason": "missing"}, feet, 0.4)["accepted"]


def test_camera_needs_observed_support_and_cut_reset():
    guard = CameraGuard()
    result = guard.check({"accepted": True, "homography": np.eye(3)}, [], 0)
    assert result["reason"] == "insufficient_person_support"
    guard.previous = (np.eye(3), 0)
    guard.reset()
    assert guard.previous is None


def candidate(x, y=200, score=0.8):
    return {"bbox_xyxy": [x - 5, y - 5, x + 5, y + 5], "score": score}


def test_ball_confirms_rejects_distant_false_positive_then_expires():
    track = BallTrack()
    assert track.update([candidate(200)], 0, (1080, 1920))["status"] == "tentative"
    assert track.update([candidate(210)], 0.1, (1080, 1920))["status"] == "detected"
    gap = track.update([candidate(1800, score=0.99)], 0.2, (1080, 1920))
    assert gap["status"] == "estimated" and not gap["eligible_for_state"]
    assert gap["selected_index"] is None
    assert track.update([], 0.6, (1080, 1920))["status"] == "unknown"


def test_ball_camera_motion_and_cuts_do_not_inherit_confirmation():
    track = BallTrack()
    track.update([candidate(200)], 0, (1080, 1920))
    output = track.update([candidate(710)], 0.1, (1080, 1920), (500, 0))
    assert output["status"] == "detected"
    assert track.update([candidate(700)], 0.2, (1080, 1920), cut=True)["status"] == "tentative"
    with pytest.raises(ValueError):
        track.update([], 0.1, (1080, 1920))


def test_jersey_needs_distinct_frames_is_causal_and_abstains_on_conflict():
    base = {"tracklet_id": 1, "shot_id": 0, "accepted": True, "number": 7}
    rows = [{**base, "frame_id": i, "timestamp_s": i, "detection_id": str(i)} for i in range(4)]
    rows[2]["number"] = 8
    result = jersey_consensus(rows)
    assert [r["jersey_hypothesis"] for r in result] == [None, 7, None, None]
    assert (
        jersey_consensus([rows[0], {**rows[0], "detection_id": "copy"}])[-1]["jersey_hypothesis"]
        is None
    )
    rows[1]["shot_id"] = 1
    assert jersey_consensus(rows[:2])[-1]["jersey_hypothesis"] is None
