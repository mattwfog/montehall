import pytest

from soccerviz.ui.datasets_ui import experiment_summary


def summary(report):
    return experiment_summary({"name": "Candidate experiment", "split": "valid", "report": report})


def test_detection_summary_shows_scope_thresholds_and_ap_candidate_floor():
    result = summary(
        {
            "schema": "vision-benchmark-results/v1",
            "backend": {"name": "RF-DETR", "training_domain": "COCO; not soccer fine-tuned"},
            "submitted_frames": 450,
            "missing_prediction_frames": 0,
            "thresholds": {
                "score_threshold": 0.25,
                "iou_threshold": 0.5,
                "ball_center_threshold_px": 10,
            },
            "overall": {
                "frames": 450,
                "classes": {"person": {"precision": 0.9143, "recall": 0.9369}},
                "ball_center": {"precision": 0.4435, "recall": 0.2546, "tp": 110, "gt": 432},
            },
            "by_sequence": {"one": {}, "two": {}, "three": {}},
            "coco_bbox": {"available": True, "map": 0.2894, "candidate_score_floor": 0.01},
        }
    )
    assert "450 frozen frames across 3 sequences" in result
    assert "person precision 91.4%, recall 93.7%" in result
    assert "score ≥0.25" in result
    assert "Ball centers within 10.0 px" in result
    assert "score floor 0.01" in result
    assert "matching candidate score floors" in result
    assert "not soccer fine-tuned" in result
    assert "Experimental comparison" in result


def test_center_only_ball_result_does_not_report_box_or_person_accuracy():
    result = summary(
        {
            "schema": "vision-benchmark-results/v1",
            "prediction_geometry": "center-only",
            "backend": {"name": "WASB"},
            "submitted_frames": 450,
            "missing_prediction_frames": 0,
            "overall": {
                "frames": 450,
                "classes": None,
                "ball_center": {"precision": 0.394, "recall": 0.0602, "tp": 26, "gt": 432},
            },
            "thresholds": {"ball_center_threshold_px": 10},
            "coco_bbox": {"available": False},
        }
    )
    assert "recall 6.0% (26/432 annotated balls)" in result
    assert "box AP is undefined" in result
    assert "no person detection score is measured" in result
    assert "Box mAP" not in result


def test_tracking_summary_keeps_anonymous_short_sequence_scope():
    result = summary(
        {
            "schema": "detector-tracking-comparison/v1",
            "tracker": "TrackletAssociator; no appearance",
            "overall": {"HOTA": 0.71144, "IDF1": 0.8654, "DetA": 0.7005, "AssA": 0.7286},
            "by_sequence": {
                "one": {"frames": 150},
                "two": {"frames": 150},
                "three": {"frames": 150},
            },
        }
    )
    assert "450 frames across 3 sequences" in result
    assert "HOTA **0.711**" in result
    assert "IDF1 **0.865**" in result
    assert "no appearance" in result
    assert "does not establish team or jersey identity" in result
    assert "camera cuts" in result


def test_calibration_summary_cannot_hide_large_errors_behind_good_median():
    result = summary(
        {
            "schema": "calibration-benchmark-results/v1",
            "oracle_image_footpoints": True,
            "ground_truth_used_for_fitting": False,
            "missing_prediction_frames": 0,
            "overall": {
                "frames": 45,
                "accepted_frames": 45,
                "frame_coverage": 1.0,
                "valid_target_points": 876,
                "projected_points": 876,
                "all_target_fraction_within_2m": 0.83904,
                "accepted_point_error": {"median_m": 0.4762, "p95_m": 18.2275, "mean_m": 37.8864},
            },
        }
    )
    for expected in (
        "45/45 cameras accepted",
        "876/876 target points",
        "83.9%",
        "0.476",
        "18.23",
        "37.89",
    ):
        assert expected in result
    assert "Large projection errors remain" in result
    assert "Ground-truth image footpoints" in result
    assert "not end-to-end" in result


def test_jersey_summary_binds_precision_to_coverage_and_labels():
    result = summary(
        {
            "schema": "jersey-evaluation/v1",
            "summary": {
                "expected_crops": 200,
                "submitted_prediction_crops": 200,
                "missing_prediction_crops": 0,
                "accepted_crops": 19,
                "abstained_crops": 181,
                "acceptance_coverage": 0.095,
                "matched_number_labeled_crops": 114,
                "accepted_labeled_crops": 19,
                "accepted_number_precision": 17 / 19,
                "correct_accepted_number_coverage_of_labeled_crops": 17 / 114,
            },
            "records": [{"correct_number": True}] * 17 + [{"correct_number": False}] * 2,
        }
    )
    for expected in (
        "200 sampled predicted crops",
        "19 accepted",
        "9.5%",
        "181 abstained",
        "114 crops",
        "89.5%",
        "14.9%",
        "17 correct accepted readings",
    ):
        assert expected in result
    assert "do not establish current-frame legibility" in result
    assert "Confidence is uncalibrated" in result


def test_missing_jersey_outputs_and_undefined_precision_are_not_success():
    result = summary(
        {
            "schema": "jersey-evaluation/v1",
            "summary": {
                "expected_crops": 200,
                "submitted_prediction_crops": 8,
                "missing_prediction_crops": 192,
                "accepted_crops": 0,
                "abstained_crops": 200,
                "acceptance_coverage": 0,
                "accepted_labeled_crops": 0,
                "accepted_number_precision": None,
            },
        }
    )
    assert "8 outputs, 192 missing" in result
    assert "0.0%" in result
    assert "Accepted labeled precision **unavailable**" in result
    assert "missing outputs count as abstentions" in result


def test_efpi_summary_reports_exclusions_and_stability_not_accuracy():
    result = summary(
        {
            "schema_version": 1,
            "method": "UnravelSports EFPI public API",
            "game": 1,
            "sampled_frames": 300,
            "eligible_frames": 126,
            "omitted_frames": 174,
            "duration_s_requested": 300,
            "sample_frequency_hz": 1,
            "frame_efpi": {
                "home": {
                    "dominant_formation": "442",
                    "dominant_share": 0.2857,
                    "same_phase_adjacent_stability": 0.6762,
                }
            },
            "baseline_three_template": {"home": {"same_phase_adjacent_stability": 0.8667}},
        }
    )
    for expected in ("126/300", "42.0%", "174 omitted", "28.6%", "67.6%", "86.7%"):
        assert expected in result
    assert "Stability is not accuracy" in result
    assert "geometry proxies" in result
    assert "without interpolation" in result


@pytest.mark.parametrize(
    "schema",
    [
        "vision-benchmark-results/v1",
        "detector-tracking-comparison/v1",
        "calibration-benchmark-results/v1",
        "jersey-evaluation/v1",
    ],
)
def test_unavailable_experiment_gracefully_reports_failure(schema):
    result = summary({"schema": schema, "status": "failed", "error": "Checkpoint missing"})
    assert "failed" in result
    assert "No completed result is available" in result
    assert "Checkpoint missing" in result
    assert "0.0%" not in result


def test_missing_report_is_unavailable_and_legacy_training_branch_is_preserved():
    assert "unavailable or incomplete" in summary(None)
    result = summary(
        {
            "test_metrics": {
                "advance": {"model": {"brier": 0.12}, "train_prevalence_baseline": {"brier": 0.21}}
            }
        }
    )
    assert "| advance | 0.120000 | 0.210000 |" in result
    assert "Match-disjoint evaluation does not establish coaching value" in result
