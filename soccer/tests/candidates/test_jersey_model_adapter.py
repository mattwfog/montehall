import numpy as np
import pytest

from soccerviz.candidates.jersey_model import (
    crop_bounds,
    decision,
    jersey_label,
    select_crops,
    validate_number_output,
)


def test_evaluation_excludes_missing_jersey_label_from_precision(tmp_path):
    import json

    from soccerviz.candidates.jersey_model import evaluate, file_sha

    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(
            {
                "images": [{"image_id": "image1", "file_name": "000001.jpg"}],
                "annotations": [
                    {
                        "id": "a1",
                        "image_id": "image1",
                        "track_id": 1,
                        "supercategory": "object",
                        "attributes": {"role": "player", "jersey": "7"},
                        "bbox_image": {"x": 0, "y": 0, "w": 10, "h": 30},
                    },
                    {
                        "id": "a2",
                        "image_id": "image1",
                        "track_id": 2,
                        "supercategory": "object",
                        "attributes": {"role": "player", "jersey": None},
                        "bbox_image": {"x": 20, "y": 0, "w": 10, "h": 30},
                    },
                ],
            }
        )
    )
    crop_path = tmp_path / "crops.json"
    crop_path.write_text(
        json.dumps(
            {
                "manifest_sha256": "frozen",
                "crops": [
                    {
                        "crop_id": "c1",
                        "sequence": "seq",
                        "frame_id": 1,
                        "detector_score": 0.9,
                        "predicted_bbox_xyxy": [0, 0, 10, 30],
                    },
                    {
                        "crop_id": "c2",
                        "sequence": "seq",
                        "frame_id": 1,
                        "detector_score": 0.8,
                        "predicted_bbox_xyxy": [20, 0, 30, 30],
                    },
                ],
            }
        )
    )
    pred_path = tmp_path / "predictions.json"
    pred_path.write_text(
        json.dumps(
            {
                "crop_manifest_sha256": file_sha(crop_path),
                "predictions": [
                    {"crop_id": "c1", "pred_number": 7, "jersey_number": 7},
                    {"crop_id": "c2", "pred_number": 8, "jersey_number": 8},
                ],
            }
        )
    )
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "manifest_sha256": "frozen",
                "sources": [
                    {"sequence": "seq", "path": str(source_path), "sha256": file_sha(source_path)}
                ],
            }
        )
    )
    report = evaluate(crop_path, pred_path, truth_path, tmp_path / "evaluation.json")
    assert report["summary"]["accepted_crops"] == 2
    assert report["summary"]["matched_number_labeled_crops"] == 1
    assert report["summary"]["accepted_number_precision"] == 1.0
    assert report["summary"]["accepted_unmatched_or_number_unlabeled"] == 1
    partial_path = tmp_path / "partial.json"
    partial_path.write_text(
        json.dumps(
            {
                "crop_manifest_sha256": file_sha(crop_path),
                "predictions": [{"crop_id": "c2", "pred_number": 8, "jersey_number": 8}],
            }
        )
    )
    partial = evaluate(crop_path, partial_path, truth_path, tmp_path / "partial-evaluation.json")
    assert partial["summary"]["expected_crops"] == 2
    assert partial["summary"]["submitted_prediction_crops"] == 1
    assert partial["summary"]["missing_prediction_crops"] == 1
    assert partial["summary"]["evaluated_crops"] == report["summary"]["evaluated_crops"]
    assert partial["summary"]["matched_gt_crops"] == report["summary"]["matched_gt_crops"]
    assert (
        partial["summary"]["matched_number_labeled_crops"]
        == report["summary"]["matched_number_labeled_crops"]
    )
    assert partial["summary"]["acceptance_coverage"] == 0.5
    assert partial["records"][0]["reason"] == "missing_prediction"


def test_crop_clips_original_pixel_edges_without_resizing():
    assert crop_bounds([-1.1, 2.3, 20.2, 40.1], 20, 40) == [0, 2, 20, 40]


@pytest.mark.parametrize("box", [[1, 2, 1, 40], [50, 10, 60, 40], [0, 0, np.nan, 40], [0, 0, 2, 3]])
def test_unreadable_or_empty_crops_fail(box):
    with pytest.raises(ValueError):
        crop_bounds(box, 20, 40)


@pytest.mark.parametrize("value", [None, "", "-1", "unknown", "100", True, "7.5"])
def test_missing_or_invalid_gt_number_stays_unknown(value):
    assert jersey_label(value) is None


def test_zero_is_valid_number_and_unknown_is_distinct():
    assert jersey_label("0") == 0
    assert jersey_label(None) is None


def test_exported_upstream_archive_requires_exact_source_hashes(tmp_path, monkeypatch):
    import soccerviz.candidates.jersey_model as adapter

    source = tmp_path / "config.py"
    source.write_text("fixed model configuration\n")
    monkeypatch.setattr(adapter, "UPSTREAM_FILE_HASHES", {"config.py": adapter.file_sha(source)})
    assert not (tmp_path / ".git").exists()
    assert adapter.verify_upstream(tmp_path) == adapter.UPSTREAM_FILE_HASHES
    source.write_text("changed model configuration\n")
    with pytest.raises(ValueError, match="source/config hashes"):
        adapter.verify_upstream(tmp_path)


def test_high_uncertainty_and_low_probability_both_abstain():
    assert decision(8, 0.95, 0.8)["jersey_number"] is None
    assert decision(8, 0.2, 0.05)["jersey_number"] is None
    assert decision(8, 0.95, 0.05)["jersey_number"] == 8


def test_full_distribution_must_agree_with_number_score_and_uncertainty():
    alpha = np.ones(100)
    alpha[8] = 1000
    probabilities = alpha / alpha.sum()
    uncertainty = 100 / alpha.sum()
    validate_number_output(8, probabilities[8], uncertainty, probabilities, alpha)
    for number, score, u in [
        (9, probabilities[8], uncertainty),
        (8, 0.1, uncertainty),
        (8, probabilities[8], 0.5),
    ]:
        with pytest.raises(ValueError):
            validate_number_output(number, score, u, probabilities, alpha)


def test_sampling_uses_predictions_only_and_rejects_changed_source_hash():
    frames, predictions = [], []
    for seq in ("A", "B"):
        for frame in (1, 2, 3):
            frames.append(
                {
                    "sequence": seq,
                    "frame_id": frame,
                    "sha256": f"{seq}{frame}",
                    "timestamp_s": frame / 25,
                    "image_path": "/not-needed",
                    "width": 100,
                    "height": 100,
                }
            )
            predictions.append(
                {
                    "sequence": seq,
                    "frame_id": frame,
                    "image_sha256": f"{seq}{frame}",
                    "detections": [
                        {
                            "label": "person",
                            "role": "player",
                            "score": 0.9,
                            "bbox_xyxy": [10, 10, 30, 60],
                        }
                    ],
                }
            )
    rows, counts = select_crops({"frames": frames}, {"frames": predictions}, limit=4)
    assert counts == {"A": 3, "B": 3}
    assert [(r["sequence"], r["frame_id"]) for r in rows] == [
        ("A", 1),
        ("A", 3),
        ("B", 1),
        ("B", 3),
    ]
    predictions[0]["image_sha256"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        select_crops({"frames": frames}, {"frames": predictions}, limit=4)
