import copy

import pytest

from soccerviz.harness.engine import Harness
from soccerviz.vision.observation_channels import (
    ball_stage,
    compose_state,
    players_stage,
    video_channels,
)
from soccerviz.vision.specialists import evidence_stage, specialists, state_stage


def source_fixture():
    tables = {
        name: [] for name in ("frames", "detections", "state", "calibration", "ball_candidates")
    }
    for fid, time in enumerate((1.0, 1.2, 1.4)):
        tables["frames"].append(
            {
                "frame_id": fid,
                "timestamp_s": time,
                "shot_id": int(fid == 2),
                "source_frame": 25 + fid * 5,
                "pts": 12500 + fid * 2500,
                "time_base": "1/12500",
            }
        )
        tables["calibration"].append(
            {
                "frame_id": fid,
                "timestamp_s": time,
                "accepted": True,
                "homography": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            }
        )
        for track, team, x in ((1, 0, 20.0), (2, 1, 60.0)):
            eid = f"f{fid}-d{track}"
            tables["detections"].append(
                {
                    "frame_id": fid,
                    "timestamp_s": time,
                    "detection_id": eid,
                    "tracklet_id": track,
                    "role_hypothesis": "player",
                    "bbox_x0": x - 1,
                    "bbox_y0": 28.0,
                    "bbox_x1": x + 1,
                    "bbox_y1": 30.0,
                    "confidence": 0.9,
                    "source": "synthetic_test",
                }
            )
            tables["state"].append(
                {
                    "frame_id": fid,
                    "timestamp_s": time,
                    "evidence_id": eid,
                    "tracklet_id": track,
                    "entity": "player",
                    "team_cluster": team,
                    "x_m": x,
                    "y_m": 30.0,
                    "calibration_accepted": True,
                }
            )
        tables["ball_candidates"].append(
            {
                "frame_id": fid,
                "timestamp_s": time,
                "rank": 0,
                "confidence": 0.99,
                "x0": 19.0,
                "y0": 29.0,
                "x1": 21.0,
                "y1": 31.0,
            }
        )
    return {
        "schema": "video-input/v1",
        "source_sha256": "synthetic-test-video",
        "model_sha256": {"player": "synthetic-test-model"},
        "artifact_sha256": {"frames": "fixture"},
        "window": {"start_s": 1.0, "end_s": 1.6},
        "tables": tables,
    }


def review(field="identity", value="Player A"):
    return {
        "revision": 1,
        "created_at": "2026-09-07T12:00:00Z",
        "review": {
            "field": field,
            "value": value,
            "tracklet_id": 1,
            "start_s": 1.0,
            "end_s": 1.6,
            "reviewer": "synthetic-test-reviewer",
            "reason": "fixture review",
            "evidence_ids": ["f0-d1"],
        },
    }


def test_explicit_graph_and_legacy_composition_preserve_clock_and_review_alternatives():
    source = source_fixture()
    evidence = evidence_stage(source, {}, [])
    channels = video_channels(source, evidence, [review()])
    state = compose_state(source, channels)
    assert state == state_stage(source, {"evidence": evidence}, [review()])
    assert state["schema"] == "match-state/v1"
    assert list(state["channels"]) == ["camera", "calibration", "players", "ball", "context"]
    assert state["frames"][0]["camera"]["pts"] == 12500
    assert state["frames"][2]["camera"]["cut_before"]
    assert state["frames"][0]["players"][0]["identity"] == "Player A"
    assert state["frames"][0]["players"][0]["identity_hypotheses"][0]["value"] is None
    # Reused integer track 1 after the cut is a different anonymous entity.
    assert state["frames"][2]["players"][0]["identity"] is None
    assert (
        state["frames"][0]["players"][0]["entity_id"]
        != state["frames"][2]["players"][0]["entity_id"]
    )
    assert state["frames"][0]["possession"]["team"] == 0
    assert state["frames"][0]["possession"]["probability"] is None
    assert state["frames"][0]["players"][0]["position_sigma_m"] is None
    assert channels["players"]["frames"][0]["entities"][0]["identity"] is None
    assert channels["context"]["review_history"][0]["review"]["reason"] == "fixture review"


def test_unavailable_empty_and_rejected_channels_keep_clock_and_unknowns():
    source = source_fixture()
    source["tables"]["ball_candidates"] = []
    source["tables"]["calibration"][1]["homography"] = [0.0] * 9
    evidence = evidence_stage(source, {}, [])
    channels = video_channels(source, evidence)
    assert channels["ball"]["census"]["status"] == "empty"
    assert channels["ball"]["census"]["frames_without_observations"] == [0, 1, 2]
    assert channels["calibration"]["frames"][1]["status"] == "invalid_homography"
    assert channels["players"]["frames"][1]["entities"][0]["xy_m"] is None
    missing = copy.deepcopy(evidence)
    del missing["ball_candidates"]
    channels["ball"] = ball_stage(source, {**channels, "evidence": missing}, [])
    assert channels["ball"]["census"]["status"] == "missing"
    state = compose_state(source, channels)
    assert len(state["frames"]) == 3
    assert all(frame["possession"]["status"] == "unknown" for frame in state["frames"])


def test_person_detection_without_projection_is_retained_and_auxiliary_roles_are_explicit():
    source = source_fixture()
    source["tables"]["state"] = source["tables"]["state"][1:]
    source["tables"]["detections"][1]["role_hypothesis"] = "goalkeeper"
    source["tables"]["state"][0]["entity"] = "goalkeeper"
    evidence = evidence_stage(source, {}, [])
    channels = video_channels(source, evidence)
    frame = compose_state(source, channels)["frames"][0]
    assert len(frame["entities"]) == 2
    assert len(frame["players"]) == 1
    assert frame["players"][0]["xy_m"] is None
    assert frame["players"][0]["team"] is None
    assert frame["entities"][1]["role"] == "goalkeeper"
    assert channels["players"]["census"]["observation_count"] == 6


def test_composition_rejects_cross_source_or_duplicate_frame_channels():
    source = source_fixture()
    channels = video_channels(source, evidence_stage(source, {}, []))
    broken = copy.deepcopy(channels)
    broken["ball"]["source_sha256"] = "another-video"
    with pytest.raises(ValueError, match="another source"):
        compose_state(source, broken)
    broken = copy.deepcopy(channels)
    broken["players"]["frames"].append(broken["players"]["frames"][0])
    with pytest.raises(ValueError, match="frame IDs"):
        compose_state(source, broken)


def test_review_invalidates_context_and_descendants_without_rerunning_observation_organs(tmp_path):
    source = source_fixture()
    harness = Harness(tmp_path)
    run = harness.create(source)
    harness.execute(run, specialists())
    harness.review(run, review()["review"])
    result = harness.execute(run, specialists())
    actions = {row["stage"]: row["action"] for row in result["stages"]}
    assert {name for name, action in actions.items() if action == "reused"} == {
        "evidence",
        "camera",
        "calibration",
        "players",
        "ball",
    }
    assert {name for name, action in actions.items() if action == "executed"} == {
        "context",
        "state",
        "tactics",
        "brief",
    }
    state_record = next(row for row in harness.status(run)["results"] if row["stage"] == "state")
    payload = harness.get(state_record["output_id"])["payload"]
    assert payload["frames"][0]["players"][0]["identity"] == "Player A"


def test_projection_organ_does_not_mutate_input_hypotheses():
    source = source_fixture()
    evidence = evidence_stage(source, {}, [])
    channels = video_channels(source, evidence)
    original = copy.deepcopy(evidence)
    players_stage(source, channels, [review()])
    assert evidence == original
