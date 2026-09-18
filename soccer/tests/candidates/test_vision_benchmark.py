import json

import pytest

from soccerviz.candidates.vision_benchmark import canonical_label, evaluate, size_bucket
from soccerviz.datasets.soccernet_adapter import sha256, write_json


def detection(label, box=(0, 0, 10, 10), score=0.9):
    return {"label": label, "bbox_xyxy": list(box), "score": score}


@pytest.fixture
def corpus(tmp_path):
    protocol = tmp_path / "protocol.json"
    write_json(
        protocol, {"score_threshold": 0.25, "iou_threshold": 0.5, "ball_center_threshold_px": 10.0}
    )
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fixture source bytes")
    frames = [
        {
            "sequence": "SNGS-021",
            "frame_id": index,
            "image_path": str(image),
            "sha256": sha256(image),
            "width": 1920,
            "height": 1080,
        }
        for index in (1, 2, 3)
    ]
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "schema": "vision-benchmark/v1",
            "frames": frames,
            "protocol_path": str(protocol),
            "protocol_sha256": sha256(protocol),
        },
    )
    targets = [
        {
            "sequence": "SNGS-021",
            "frame_id": 1,
            "detections": [
                detection("person", (0, 0, 30, 60)),
                detection("ball", (50, 50, 60, 60)),
            ],
        },
        {
            "sequence": "SNGS-021",
            "frame_id": 2,
            "detections": [
                detection("person", (0, 0, 30, 60)),
                detection("ball", (50, 50, 60, 60)),
            ],
        },
        {"sequence": "SNGS-021", "frame_id": 3, "detections": []},
    ]
    truth = tmp_path / "ground-truth.json"
    write_json(
        truth,
        {
            "schema": "vision-ground-truth/v1",
            "manifest_sha256": sha256(manifest),
            "sources": [],
            "frames": targets,
        },
    )
    predictions = tmp_path / "predictions.json"

    def score(frames, **overrides):
        write_json(
            predictions,
            {
                "schema": "vision-predictions/v1",
                "manifest_sha256": sha256(manifest),
                "backend": "unit-test",
                "checkpoint_hashes": [],
                "frames": frames,
                **overrides,
            },
        )
        return evaluate(manifest, predictions)

    return manifest, truth, predictions, score


def test_missing_and_empty_frames_stay_in_denominator(corpus):
    _, _, _, score = corpus
    result = score(
        [
            {
                "sequence": "SNGS-021",
                "frame_id": 1,
                "detections": [
                    detection("player", (0, 0, 30, 60)),
                    detection("sports ball", (50, 50, 60, 60)),
                ],
            }
        ]
    )
    assert result["overall"]["frames"] == 3
    assert result["missing_prediction_frames"] == 2
    for label in ("person", "ball"):
        assert result["overall"]["classes"][label]["recall"] == 0.5
        assert result["overall"]["classes"][label]["precision"] == 1.0
    empty = score([])
    assert empty["overall"]["classes"]["ball"]["recall"] == 0
    assert empty["overall"]["classes"]["person"]["precision"] is None


def test_empty_ground_truth_frame_false_positives_count(corpus):
    _, _, _, score = corpus
    result = score([{"sequence": "SNGS-021", "frame_id": 3, "detections": [detection("ball")]}])
    assert result["overall"]["ball_center"]["fp"] == 1
    assert result["overall"]["ball_center"]["false_positives_per_frame"] == pytest.approx(1 / 3)


def test_role_collapse_and_strict_numeric_class_rejection():
    assert all(
        canonical_label(role) == "person" for role in ("person", "Player", "goalkeeper", "referee")
    )
    assert canonical_label("sports_ball") == "ball"
    with pytest.raises(ValueError, match="Unsupported prediction label"):
        canonical_label(0)


def test_wrong_manifest_rejected(corpus):
    _, _, _, score = corpus
    with pytest.raises(ValueError, match="provenance mismatch"):
        score([], manifest_sha256="a" * 64)


def test_changed_protocol_rejected(corpus):
    manifest, _, _, score = corpus
    protocol = json.loads(manifest.read_text())["protocol_path"]
    write_json(protocol, {"score_threshold": 0.1})
    with pytest.raises(ValueError, match="protocol SHA256"):
        score([])


@pytest.mark.parametrize(
    "frames",
    [
        [{"sequence": "SNGS-999", "frame_id": 1, "detections": []}],
        [{"sequence": "SNGS-021", "frame_id": 1, "detections": []}] * 2,
        [{"sequence": "SNGS-021", "frame_id": 0, "detections": []}],
    ],
)
def test_unexpected_duplicate_invalid_frames_rejected(corpus, frames):
    with pytest.raises(ValueError):
        corpus[3](frames)


def test_class_confusion_is_both_false_positive_and_miss(corpus):
    result = corpus[3](
        [{"sequence": "SNGS-021", "frame_id": 1, "detections": [detection("ball", (0, 0, 30, 60))]}]
    )
    assert result["overall"]["classes"]["person"]["fn"] == 2
    assert result["overall"]["classes"]["ball"]["fp"] == 1
    assert result["overall"]["classes"]["ball"]["tp"] == 0


def test_duplicate_detection_and_score_boundary(corpus):
    result = corpus[3](
        [
            {
                "sequence": "SNGS-021",
                "frame_id": 1,
                "detections": [
                    detection("ball", (50, 50, 60, 60), 0.25),
                    detection("ball", (50, 50, 60, 60), 0.8),
                    detection("ball", (50, 50, 60, 60), 0.2499),
                ],
            }
        ]
    )
    assert result["overall"]["classes"]["ball"]["tp"] == 1
    assert result["overall"]["classes"]["ball"]["fp"] == 1
    assert result["overall"]["classes"]["ball"]["predictions"] == 2


def test_ball_center_boundary_does_not_claim_box_match(corpus):
    result = corpus[3](
        [
            {
                "sequence": "SNGS-021",
                "frame_id": 1,
                "detections": [detection("ball", (60, 50, 70, 60))],
            }
        ]
    )
    assert result["overall"]["classes"]["ball"]["tp"] == 0
    assert result["overall"]["ball_center"]["tp"] == 1
    assert result["overall"]["ball_center"]["matched_center_error_px"]["mean"] == 10


def test_ignored_predictions_reported_separately(corpus):
    _, truth_path, _, score = corpus
    truth = json.loads(truth_path.read_text())
    truth["frames"][2]["ignored_boxes"] = [[0, 0, 10, 10]]
    write_json(truth_path, truth)
    result = score([{"sequence": "SNGS-021", "frame_id": 3, "detections": [detection("person")]}])
    metrics = result["overall"]["classes"]["person"]
    assert metrics["ignored"] == 1
    assert metrics["fp"] == 0


def test_size_buckets_are_ground_truth_recall(corpus):
    assert size_bucket([0, 0, 31, 31]) == "small"
    assert size_bucket([0, 0, 32, 32]) == "medium"
    assert size_bucket([0, 0, 96, 96]) == "large"
    result = corpus[3]([])
    assert result["overall"]["classes"]["ball"]["size_buckets"]["small"]["fn"] == 2
    assert result["overall"]["classes"]["person"]["size_buckets"]["medium"]["fn"] == 2


def test_source_image_integrity(corpus):
    manifest, _, _, score = corpus
    from pathlib import Path

    image_path = json.loads(manifest.read_text())["frames"][0]["image_path"]
    Path(image_path).write_bytes(b"modified image")
    with pytest.raises(ValueError, match="Image SHA256"):
        score([])


@pytest.mark.parametrize(
    "row", [detection("ball", score=1.01), detection("ball", box=(1, 1, 0, 0))]
)
def test_malformed_detection_rejected(corpus, row):
    with pytest.raises(ValueError):
        corpus[3]([{"sequence": "SNGS-021", "frame_id": 1, "detections": [row]}])


def test_missing_ground_truth_frame_rejected(corpus):
    _, truth_path, _, score = corpus
    truth = json.loads(truth_path.read_text())
    truth["frames"].pop()
    write_json(truth_path, truth)
    with pytest.raises(ValueError, match="Ground-truth frame set"):
        score([])


def test_center_only_predictions_do_not_manufacture_boxes(corpus):
    result = corpus[3](
        [
            {
                "sequence": "SNGS-021",
                "frame_id": 1,
                "detections": [
                    {"label": "ball", "center_xy": [65, 55], "score": 0.25},
                    {"label": "ball", "center_xy": [55, 55], "score": 0.2499},
                ],
            }
        ],
        schema="vision-center-predictions/v1",
    )
    assert result["prediction_geometry"] == "center-only"
    assert result["overall"]["classes"] is None
    assert result["overall"]["ball_center"]["tp"] == 1
    assert result["overall"]["ball_center"]["recall"] == 0.5
    assert result["overall"]["ball_center"]["matched_center_error_px"]["mean"] == 10
    assert result["coco_bbox"]["map"] is None
    assert "undefined" in result["coco_bbox"]["reason"]
    assert (
        "unmatched_predicted_boxes" not in result["overall"]["ball_center"]["size_buckets"]["small"]
    )


def test_center_only_boundaries_and_false_positives(corpus):
    result = corpus[3](
        [
            {"sequence": "SNGS-021", "frame_id": 1, "detections": []},
            {
                "sequence": "SNGS-021",
                "frame_id": 2,
                "detections": [{"label": "ball", "center_xy": [65.001, 55], "score": 0.9}],
            },
            {
                "sequence": "SNGS-021",
                "frame_id": 3,
                "detections": [{"label": "ball", "center_xy": [55, 55], "score": 0.9}],
            },
        ],
        schema="vision-center-predictions/v1",
    )
    metrics = result["overall"]["ball_center"]
    assert metrics["fn"] == 2
    assert metrics["fp"] == 2
    assert metrics["tp"] == 0
    assert result["missing_prediction_frames"] == 0


@pytest.mark.parametrize(
    "row",
    [
        {"label": "person", "center_xy": [55, 55], "score": 0.9},
        {"label": "ball", "center_xy": [float("inf"), 55], "score": 0.9},
        {"label": "ball", "center_xy": [55], "score": 0.9},
    ],
)
def test_center_only_malformed_geometry_rejected(corpus, row):
    with pytest.raises(ValueError):
        corpus[3](
            [{"sequence": "SNGS-021", "frame_id": 1, "detections": [row]}],
            schema="vision-center-predictions/v1",
        )


def test_coco_ap_oracle_and_empty(corpus):
    pytest.importorskip("pycocotools")
    _, truth, _, score = corpus
    frames = json.loads(truth.read_text())["frames"]
    result = score(frames, backend={"name": "oracle sanity", "confidence_threshold": 0.01})
    assert result["coco_bbox"]["available"]
    assert result["coco_bbox"]["map"] == pytest.approx(1)
    assert result["coco_bbox"]["by_class"]["ball"]["map"] == pytest.approx(1)
    assert result["coco_bbox"]["candidate_score_floor"] == 0.01
    assert result["coco_bbox"]["score_truncated_input"]
    assert score([])["coco_bbox"]["map"] == 0
