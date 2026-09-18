import json

import pytest

from soccerviz.datasets.soccernet_adapter import (
    RangeReader,
    import_gsr,
    import_tracking,
    read_mot,
    validate_version,
)


def gsr_fixture(path):
    value = {
        "info": {"version": "1.3", "frame_rate": 25},
        "images": [
            {
                "image_id": "2001000001",
                "file_name": "000001.jpg",
                "width": 1920,
                "height": 1080,
                "has_labeled_pitch": True,
                "has_labeled_camera": True,
                "has_labeled_person": True,
            },
            {
                "image_id": "2001000006",
                "file_name": "000006.jpg",
                "width": 1920,
                "height": 1080,
                "has_labeled_pitch": False,
                "has_labeled_camera": True,
                "has_labeled_person": True,
            },
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"}],
    }
    path.write_text(json.dumps(value))
    return value


def test_gsr_preserves_frame_clock_and_label_coverage(tmp_path):
    path = tmp_path / "Labels-GameState.json"
    gsr_fixture(path)
    result = import_gsr(path, "valid", "SNGS-001")
    assert [f["timestamp_s"] for f in result["frames"]] == [0, 0.2]
    assert [f["scored"] for f in result["frames"]] == [True, False]
    assert result["labels_are_model_input"] is False
    assert result["split"] == "valid"


def test_gsr_rejects_old_annotations(tmp_path):
    with pytest.raises(ValueError, match="1.3"):
        validate_version({"info": {"version": "1.2"}})


def test_range_reader_rejects_unbounded_download_before_network():
    reader = RangeReader("https://example.invalid/data.zip", 1000, budget=50)
    with pytest.raises(ValueError, match="budget"):
        reader.read(100)


def test_mot_rejects_duplicate_ids_and_invalid_box(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("1,1,0,0,20,40,1,-1,-1,-1\n1,1,1,0,20,40,1,-1,-1,-1\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_mot(path, True)
    path.write_text("1,1,0,0,-20,40,1,-1,-1,-1\n")
    with pytest.raises(ValueError, match="Invalid"):
        read_mot(path, True)


def test_mot_import_preserves_empty_frames(tmp_path):
    seq = tmp_path / "SNMOT-001"
    seq.mkdir()
    (seq / "gt").mkdir()
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nframeRate=25\nseqLength=3\nimWidth=1920\nimHeight=1080\n"
    )
    (seq / "gt/gt.txt").write_text("2,1,10,20,30,40,1,-1,-1,-1\n")
    result = import_tracking(seq, "train")
    assert len(result["frames"]) == 3
    assert result["ground_truth"][0]["bbox"] == [10, 20, 40, 60]
    assert result["frames"][1]["timestamp_s"] == 0.04


def test_gsr_source_rate_cannot_be_overridden(tmp_path):
    path = tmp_path / "Labels-GameState.json"
    gsr_fixture(path)
    with pytest.raises(ValueError, match="Frame rate"):
        import_gsr(path, "valid", fps=5)


def test_worker_request_version_is_checked_and_not_forwarded(tmp_path):
    from soccerviz.datasets.soccernet_adapter import run

    path = tmp_path / "Labels-GameState.json"
    gsr_fixture(path)
    request = {
        "schema_version": 1,
        "operation": "import-gsr",
        "labels": str(path),
        "split": "valid",
    }
    assert run(request)["schema"] == "soccernet-benchmark/v1"
    with pytest.raises(ValueError, match="schema_version"):
        run({**request, "schema_version": 2})
