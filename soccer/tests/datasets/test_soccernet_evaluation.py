import copy
import importlib.util
import json

import pytest

from soccerviz.datasets.soccernet_adapter import write_json
from soccerviz.datasets.soccernet_evaluation import evaluate_gsr, evaluate_mot

try:
    has_gsr = importlib.util.find_spec("trackeval.datasets.soccernet_gs") is not None
except ModuleNotFoundError:
    has_gsr = False


def fixture(tmp_path):
    box = {"x": 20, "y": 10, "w": 30, "h": 40}
    pitch = {
        "x_bottom_left": 0,
        "y_bottom_left": 0,
        "x_bottom_middle": 0.5,
        "y_bottom_middle": 0,
        "x_bottom_right": 1,
        "y_bottom_right": 0,
    }
    ann = {
        "image_id": "2001000001",
        "track_id": 1,
        "supercategory": "object",
        "category_id": 1,
        "bbox_image": box,
        "bbox_pitch": pitch,
        "attributes": {"role": "player", "team": "left", "jersey": "3"},
    }
    gt = {
        "info": {"version": "1.3"},
        "categories": [{"id": 1, "name": "person"}],
        "images": [
            {
                "image_id": "2001000001",
                "file_name": "000001.jpg",
                "has_labeled_pitch": True,
                "has_labeled_camera": True,
                "has_labeled_person": True,
            }
        ],
        "annotations": [ann],
    }
    labels = tmp_path / "labels.json"
    predictions = tmp_path / "predictions.json"
    write_json(labels, gt)
    write_json(predictions, {"predictions": [copy.deepcopy(ann)]})
    return labels, predictions


@pytest.mark.skipif(not has_gsr, reason="official SoccerNet evaluator environment required")
def test_official_gs_hota_attributes_gate_but_pixel_metrics_do_not(tmp_path):
    labels, predictions = fixture(tmp_path)
    result = evaluate_gsr(labels, predictions, "valid", "SNGS-001")
    assert result["gs_hota"]["GS-HOTA"] == pytest.approx(1)
    p = json.loads(predictions.read_text())
    p["predictions"][0]["attributes"]["team"] = "right"
    write_json(predictions, p)
    result = evaluate_gsr(labels, predictions, "valid", "SNGS-001")
    assert result["image"]["HOTA"] == pytest.approx(1)
    assert result["gs_hota"]["GS-HOTA"] == pytest.approx(0)


@pytest.mark.skipif(not has_gsr, reason="official SoccerNet evaluator environment required")
def test_missing_projection_is_explicit_not_silently_filtered(tmp_path):
    labels, predictions = fixture(tmp_path)
    p = json.loads(predictions.read_text())
    p["predictions"][0]["bbox_pitch"] = None
    write_json(predictions, p)
    result = evaluate_gsr(labels, predictions, "valid", "SNGS-001")
    assert result["image"]["IDF1"] == pytest.approx(1)
    assert result["gs_hota"] is None
    assert result["missing_prediction_pitch"] == 1


@pytest.mark.skipif(not has_gsr, reason="official SoccerNet evaluator environment required")
def test_source_mapping_cannot_be_guessed(tmp_path):
    labels, predictions = fixture(tmp_path)
    p = json.loads(predictions.read_text())
    p["predictions"][0]["image_id"] = "wrong"
    write_json(predictions, p)
    with pytest.raises(ValueError, match="source"):
        evaluate_gsr(labels, predictions, "valid", "SNGS-001")


def test_mot_prediction_split_is_checked_before_metrics(tmp_path):
    (tmp_path / "gt").mkdir()
    (tmp_path / "gt/gt.txt").write_text("1,1,0,0,20,40,1,-1,-1,-1\n")
    (tmp_path / "seqinfo.ini").write_text(
        "[Sequence]\nframeRate=25\nseqLength=1\nimWidth=1920\nimHeight=1080\n"
    )
    pred = tmp_path / "pred.json"
    write_json(pred, {"sequence": tmp_path.name, "split": "test", "predictions": []})
    with pytest.raises(ValueError, match="provenance"):
        evaluate_mot(tmp_path, "train", pred)


@pytest.mark.skipif(not has_gsr, reason="official SoccerNet evaluator environment required")
def test_unlabeled_frames_are_not_reported_as_scored(tmp_path):
    labels, predictions = fixture(tmp_path)
    source = json.loads(labels.read_text())
    extra = copy.deepcopy(source["images"][0])
    extra.update(image_id="2001000002", file_name="000002.jpg", has_labeled_pitch=False)
    source["images"].append(extra)
    write_json(labels, source)
    result = evaluate_gsr(labels, predictions, "valid", "SNGS-001", frame_ids=[1, 2])
    assert result["frame_ids"] == [1, 2]
    assert result["scored_frame_ids"] == [1]
    assert result["image"]["frames"] == 1


def test_evaluation_worker_request_schema_version(monkeypatch):
    from soccerviz.datasets import soccernet_evaluation

    observed = []

    def evaluator(**kwargs):
        observed.append(kwargs)
        return {"status": "evaluated"}

    monkeypatch.setattr(soccernet_evaluation, "evaluate_mot", evaluator)
    request = {
        "schema_version": 1,
        "operation": "tracking",
        "sequence_dir": "sequence",
        "split": "train",
        "predictions": "predictions.json",
    }
    assert soccernet_evaluation.run(request) == {"status": "evaluated"}
    assert observed == [
        {"sequence_dir": "sequence", "split": "train", "predictions": "predictions.json"}
    ]
    with pytest.raises(ValueError, match="schema_version"):
        soccernet_evaluation.run({**request, "schema_version": 2})
