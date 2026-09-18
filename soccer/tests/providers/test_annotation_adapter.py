import json
import xml.etree.ElementTree as ET

import pytest

from soccerviz.core.assets import sha256
from soccerviz.harness.engine import Harness, digest
from soccerviz.providers.annotation import apply_harness_reviews, export_cvat, import_cvat


@pytest.fixture
def evidence():
    frames = [
        {
            "frame_id": i,
            "source_frame": i * 5,
            "timestamp_s": i * 0.2,
            "pts": i * 2560,
            "time_base": "1/12800",
            "width": 1920,
            "height": 1080,
        }
        for i in range(3)
    ]
    detections = [
        {
            "frame_id": i,
            "timestamp_s": i * 0.2,
            "detection_id": f"f{i}-d0",
            "tracklet_id": 8,
            "role_hypothesis": "player",
            "bbox_x0": 10.0,
            "bbox_y0": 20.0,
            "bbox_x1": 30.0,
            "bbox_y1": 60.0,
        }
        for i in range(3)
    ]
    return {
        "schema": "video-input/v1",
        "window": {"start_s": 0, "end_s": 0.6},
        "source_sha256": "synthetic-video",
        "model_sha256": {},
        "tables": {
            "frames": frames,
            "detections": detections,
            "state": [],
            "calibration": [],
            "ball_candidates": [],
        },
    }


def bundle(tmp_path, evidence, approve=True):
    export_cvat(evidence, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    xml = tmp_path / "annotations.xml"
    tree = ET.parse(xml)
    for box in tree.findall(".//box"):
        if box.get("outside") == "0":
            box.find("attribute[@name='reviewed']").text = str(approve).lower()
            box.find("attribute[@name='identity']").text = "test-player"
    tree.write(xml)
    review = {
        "schema": "cvat-review/v1",
        "manifest_id": digest(manifest),
        "annotations_sha256": sha256(xml),
        "reviewer": "synthetic test reviewer",
        "reason": "Synthetic fixture only",
        "reviewed_frames": [
            {"frame_id": i, "labels": ["player"], "exhaustive": True} for i in range(3)
        ],
    }
    return xml, manifest, review


def test_export_retains_source_clock_and_no_interpolation(tmp_path, evidence):
    report = export_cvat(evidence, tmp_path)
    assert report["reviewed_boxes"] == 0 and report["hota"] is None
    boxes = ET.parse(tmp_path / "annotations.xml").findall(".//box")
    assert [(b.get("frame"), b.get("outside")) for b in boxes] == [
        ("0", "0"),
        ("1", "1"),
        ("5", "0"),
        ("6", "1"),
        ("10", "0"),
        ("11", "1"),
    ]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["frames"][1]["pts"] == 2560
    assert manifest["frames"][1]["source_frame"] == 5
    with pytest.raises(ValueError, match="new export directory"):
        export_cvat(evidence, tmp_path)


def test_roundtrip_applies_only_reviewed_timestamps_idempotently(tmp_path, evidence):
    xml, manifest, review = bundle(tmp_path / "export", evidence)
    imported = import_cvat(xml, manifest, review, evidence)
    assert len(imported["ground_truth"]) == len(imported["harness_reviews"]) == 3
    harness = Harness(tmp_path / "harness")
    run = harness.create(evidence)
    result = apply_harness_reviews(harness, run, imported)
    assert result["count"] == 3
    assert apply_harness_reviews(harness, run, imported)["reused"] is True
    assert harness.context(run, 0)["reviews"] == []
    for correction in imported["harness_reviews"]:
        assert correction["end_s"] - correction["start_s"] < 1e-12


def test_unreviewed_model_shapes_are_never_ground_truth(tmp_path, evidence):
    xml, manifest, review = bundle(tmp_path, evidence, approve=False)
    with pytest.raises(ValueError, match="unreviewed"):
        import_cvat(xml, manifest, review, evidence)
    review["reviewed_frames"] = []
    assert import_cvat(xml, manifest, review, evidence)["ground_truth"] == []


def test_hash_mismatch_and_non_exhaustive_frames_rejected(tmp_path, evidence):
    xml, manifest, review = bundle(tmp_path, evidence)
    review["annotations_sha256"] = "wrong"
    with pytest.raises(ValueError, match="exact edited XML hash"):
        import_cvat(xml, manifest, review, evidence)
    review["annotations_sha256"] = sha256(xml)
    review["reviewed_frames"][0]["exhaustive"] = False
    with pytest.raises(ValueError, match="exhaustive"):
        import_cvat(xml, manifest, review, evidence)


def test_implicit_interpolated_box_rejected(tmp_path, evidence):
    xml, manifest, review = bundle(tmp_path, evidence)
    tree = ET.parse(xml)
    track = tree.find("track")
    for box in list(track):
        if box.get("frame") in ("1", "5"):
            track.remove(box)
    tree.write(xml)
    review["annotations_sha256"] = sha256(xml)
    with pytest.raises(ValueError, match="interpolated"):
        import_cvat(xml, manifest, review, evidence)


def test_reviewed_empty_frame_is_explicit_negative(tmp_path, evidence):
    xml, manifest, review = bundle(tmp_path, evidence)
    tree = ET.parse(xml)
    tree.getroot().remove(tree.find("track"))
    tree.write(xml)
    review["annotations_sha256"] = sha256(xml)
    imported = import_cvat(xml, manifest, review, evidence)
    assert imported["ground_truth"] == []
    assert len(imported["review"]["reviewed_frames"]) == 3
