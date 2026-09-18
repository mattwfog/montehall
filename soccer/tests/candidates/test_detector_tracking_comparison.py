import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).parents[2] / "scripts/benchmarks/compare_detector_tracking.py"
    spec = importlib.util.spec_from_file_location("detector_tracking_comparison", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_matched_person_is_retained_inside_an_ignore_region():
    target = {
        "detections": [{"label": "player", "bbox_xyxy": [0, 0, 10, 10]}],
        "ignored_boxes": [[0, 0, 100, 100]],
        "ignore_polygons": [[[0, 0], [100, 0], [100, 100], [0, 100]]],
    }
    rows = [
        {"label": "person", "bbox_xyxy": [0, 0, 10, 10], "score": 0.9},
        {"label": "person", "bbox_xyxy": [40, 40, 50, 50], "score": 0.8},
    ]
    assert module().ignored_prediction_indices(rows, target) == {1}


def test_empty_truth_keeps_false_positive_outside_ignore_region():
    target = {"detections": [], "ignored_boxes": [], "ignore_polygons": []}
    rows = [{"label": "person", "bbox_xyxy": [0, 0, 10, 10], "score": 0.9}]
    assert module().ignored_prediction_indices(rows, target) == set()
