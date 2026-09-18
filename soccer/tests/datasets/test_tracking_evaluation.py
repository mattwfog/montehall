import copy

import numpy as np
import pytest

from soccerviz.datasets.tracking_evaluation import evaluate_tracking, pixel_iou, sequence_data


def fixture():
    boxes = [
        {
            "frame_id": f,
            "track_id": "human-1",
            "label": "player",
            "bbox": [0, 0, 10, 10],
            "reviewed": True,
        }
        for f in range(4)
    ]
    reviewed = {
        "schema": "reviewed-tracking/v1",
        "source_sha256": "synthetic",
        "review": {
            "reviewer": "test",
            "reason": "synthetic only",
            "annotations_sha256": "synthetic-xml",
            "reviewed_frames": [
                {"frame_id": f, "labels": ["player"], "exhaustive": True} for f in range(4)
            ],
        },
        "ground_truth": boxes,
    }
    return reviewed, copy.deepcopy(boxes)


def test_iou_empty_and_invalid():
    assert pixel_iou([], [[0, 0, 1, 1]]).shape == (0, 1)
    assert pixel_iou([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        pixel_iou([[0, 0, float("nan"), 1]], [])


def test_sequence_rejects_duplicate_identity():
    _, pred = fixture()
    with pytest.raises(ValueError, match="Duplicate identity"):
        sequence_data([], [pred[0], pred[0]], [0])


def test_official_metrics_distinguish_identity_switch_from_detection_error():
    pytest.importorskip("trackeval")
    reviewed, pred = fixture()
    perfect = evaluate_tracking(reviewed, pred, source_sha256="synthetic")["classes"]["player"]
    assert perfect["HOTA"] == perfect["IDF1"] == 1
    for row in pred[2:]:
        row["track_id"] = "switched"
    switched = evaluate_tracking(reviewed, pred, source_sha256="synthetic")["classes"]["player"]
    assert switched["DetA"] == 1 and switched["AssA"] == 0.5
    assert switched["HOTA"] == pytest.approx(np.sqrt(0.5))
    assert switched["IDF1"] == 0.5
    dropped = evaluate_tracking(reviewed, pred[:2], source_sha256="synthetic")["classes"]["player"]
    assert dropped["DetA"] == 0.5 and dropped["IDFN"] == 2


def test_exhaustive_empty_frames_count_false_positives_and_unreviewed_frames_excluded():
    pytest.importorskip("trackeval")
    reviewed, pred = fixture()
    reviewed["ground_truth"] = []
    pred.append({"frame_id": 100, "track_id": "outside", "label": "player", "bbox": [0, 0, 1, 1]})
    metrics = evaluate_tracking(reviewed, pred, source_sha256="synthetic")["classes"]["player"]
    assert metrics["IDFP"] == 4 and metrics["IDF1"] == 0


def test_no_truth_and_source_mismatch():
    reviewed, pred = fixture()
    with pytest.raises(ValueError, match="hashes differ"):
        evaluate_tracking(reviewed, pred, source_sha256="wrong")
    reviewed["ground_truth"] = []
    reviewed["review"]["reviewed_frames"] = []
    assert (
        evaluate_tracking(reviewed, pred, source_sha256="synthetic")["status"]
        == "awaiting_human_review"
    )


def test_upstream_numpy_globals_are_restored():
    pytest.importorskip("trackeval")
    import trackeval.metrics.hota as hota_module
    import trackeval.metrics.identity as identity_module

    reviewed, pred = fixture()
    originals = (hota_module.np, identity_module.np)
    evaluate_tracking(reviewed, pred, source_sha256="synthetic")
    assert (hota_module.np, identity_module.np) == originals
    assert "float" not in np.__dict__
    assert "int" not in np.__dict__
