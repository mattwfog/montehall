import json
from fractions import Fraction

import av
import numpy as np
import pytest

from soccerviz.core.geometry import calibrate, pitch_landmarks, project
from soccerviz.core.identity import TrackletAssociator
from soccerviz.datasets.evaluation import detection_metrics, evaluate_annotations
from soccerviz.modeling.manager import lineup_assignment
from soccerviz.modeling.tactics import lane_simulation, pass_geometry, shape_assignment
from soccerviz.modeling.uncertainty import LastSeenState
from soccerviz.vision.pipeline import video_frames


def test_robust_calibration_and_collinear_rejection():
    rng = np.random.default_rng(0)
    world = pitch_landmarks()
    matrix = np.array([[8, 1, 100], [1, 5, 40], [0.002, 0.001, 1]])
    image = project(world, matrix) + rng.normal(0, 0.2, world.shape)
    image[5] += [500, 200]
    result = calibrate(image, world)
    assert result.accepted
    test_points = np.array([[50, 30], [20, 20], [70, 50]])
    np.testing.assert_allclose(
        project(project(test_points, matrix), result.matrix), test_points, atol=0.1
    )
    line = np.column_stack([np.arange(8), np.zeros(8)])
    assert not calibrate(line, line).accepted
    assert not calibrate(world[:4], world[:4]).accepted


def test_tracker_appearance_prevents_cross_team_swap_and_cut_reuses_no_ids():
    tracker = TrackletAssociator()
    boxes = np.array([[10, 10, 30, 60], [35, 10, 55, 60]], float)
    colors = [np.array([220, 120, 130]), np.array([60, 140, 140])]
    initial = tracker.update(boxes, [2, 2], 0, colors)
    swapped = tracker.update(boxes, [2, 2], 0.2, colors[::-1])
    assert swapped.tolist() == initial[::-1].tolist()
    tracker.reset()
    after_cut = tracker.update(boxes, [2, 2], 0.4, colors)
    assert set(after_cut).isdisjoint(initial)


def test_missing_state_expires_and_never_uses_future_or_unseen_players():
    memory = LastSeenState(2, ttl_s=1)
    memory.update(np.array([[10, 10], [np.nan, np.nan]]), 0)
    memory.update(np.array([[11, 10], [np.nan, np.nan]]), 0.2)
    held, moving, _age, eligible = memory.update(np.full((2, 2), np.nan), 0.4)
    np.testing.assert_allclose(held[0], [11, 10])
    np.testing.assert_allclose(moving[0], [12, 10])
    assert np.isnan(moving[1]).all()
    assert eligible.tolist() == [True, False]
    assert np.isnan(memory.update(np.full((2, 2), np.nan), 1.4)[1]).all()


def test_detection_evaluation_excludes_unreviewed_frames_and_matches_classes():
    predictions = [
        {"frame_id": 0, "bbox": [0, 0, 10, 10], "class": "player"},
        {"frame_id": 0, "bbox": [20, 20, 30, 30], "class": "referee"},
        {"frame_id": 1, "bbox": [0, 0, 10, 10], "class": "player"},
    ]
    truth = {
        "reviewed_frames": [0],
        "objects": [
            {"frame_id": 0, "bbox": [0, 0, 10, 10], "class": "player"},
            {"frame_id": 0, "bbox": [20, 20, 30, 30], "class": "player"},
        ],
    }
    result = detection_metrics(predictions, truth)
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (
        1,
        1,
        1,
    )
    assert result["precision_at_iou_0_5"] == 0.5


def test_model_suggestions_are_never_ground_truth(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"source_sha256": "abc"}))
    annotations = tmp_path / "annotation-task.json"
    annotations.write_text(
        json.dumps(
            {
                "source_sha256": "abc",
                "frames": [
                    {
                        "frame_id": 0,
                        "status": "unreviewed",
                        "model_suggestions": [{"class": "player"}],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="No reviewed"):
        evaluate_annotations(tmp_path, annotations)


def test_lineup_requires_context_and_respects_availability():
    assert lineup_assignment([{"player_id": "A", "available": None}], ["GK"])["status"] == "abstain"
    profiles = [
        {"player_id": "A", "available": True, "role_scores": {"GK": 0.9, "CF": 0.2}},
        {"player_id": "B", "available": True, "role_scores": {"GK": 0.1, "CF": 0.8}},
        {"player_id": "C", "available": False, "role_scores": {"GK": 1, "CF": 1}},
    ]
    result = lineup_assignment(profiles, ["GK", "CF"])
    assert {(p["player_id"], p["role"]) for p in result["lineup"]} == {("A", "GK"), ("B", "CF")}
    assert result["total_supplied_score"] == pytest.approx(1.7)
    assert lineup_assignment(profiles[:1], ["GK", "CF"])["status"] == "abstain"


def test_shape_and_pass_geometry_abstain_when_incomplete():
    assert shape_assignment(np.zeros((10, 2)))["shape"] == "unknown"
    assert pass_geometry(np.array([0, 0]), np.array([0, 0]), np.array([[1, 1]])) is None
    ball, receiver = np.array([10, 34]), np.array([40, 34])
    near = lane_simulation(ball, receiver, np.array([[25, 34]]))
    far = lane_simulation(ball, receiver, np.array([[25, 0]]))
    assert near < far


def test_video_sampling_uses_pts_including_nonzero_start(tmp_path):
    source = tmp_path / "vfr.mkv"
    with av.open(str(source), mode="w") as output:
        stream = output.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for pts in [1000, 1040, 1200, 1240, 1500]:
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    frames = list(video_frames(source, hz=5, seconds=1))
    assert [row[0] for row in frames] == [0, 2, 4]
    assert frames[0][1] >= 1
    assert np.diff([row[1] for row in frames]).min() >= 0.19
