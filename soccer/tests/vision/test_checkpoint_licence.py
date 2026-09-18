import pytest

from soccerviz.vision import checkpoint_licence as cl
from soccerviz.vision.video_workflow import PRESETS, model_hashes

RF_SOCCER = "4e5cb479a57e48fc13bc69361bf7321d5f09d6097dcf97c95433b3fe1b176f18"
RF_MEDIUM = "749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8"
RF_E50 = "e5a740c733863ef1ca98372314d16322d0ca9110e1c85eed6de808e8f432fd9e"
PNL_POINTS = "7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113"
PNL_LINES = "d72f4ed71734a2e3df9fa084f666e9b8adaef21bf69bac8952d6d3f970ff7455"


class Adapter:
    def __init__(self, **checkpoints):
        self.metadata = {"checkpoints": {k: {"sha256": v} for k, v in checkpoints.items()}}


def test_every_workflow_preset_checkpoint_is_registered():
    registered = {record["file"].split(" ")[0] for record in cl.CHECKPOINTS.values()}
    missing = {filename for _, filename, _, _ in PRESETS.values()} - registered
    assert missing == set()
    assert {"pnl-keypoints.pt", "pnl-lines.pt", "jersey-vitb.pt"} <= registered


def test_default_preset_and_calibrator_are_evaluation_only():
    report = cl.licence_report(
        model_hashes(Adapter(detector=RF_SOCCER), Adapter(keypoints=PNL_POINTS, lines=PNL_LINES))
    )
    assert report["class"] == cl.EVALUATION_ONLY
    assert report["blocking"] == ["camera_keypoints", "camera_lines", "detector_detector"]
    assert "SoccerNet" in report["checkpoints"]["detector_detector"]["licence"]
    with pytest.raises(ValueError, match="rf-soccer.pth"):
        cl.require_shipping(report)


def test_promoted_default_detector_ships_but_camera_still_blocks():
    report = cl.licence_report(
        model_hashes(Adapter(detector=RF_E50), Adapter(keypoints=PNL_POINTS, lines=PNL_LINES))
    )
    assert report["checkpoints"]["detector_detector"]["class"] == cl.SHIPPING
    assert report["class"] == cl.EVALUATION_ONLY
    assert report["blocking"] == ["camera_keypoints", "camera_lines"]


def test_shipping_requires_every_checkpoint_cleared():
    report = cl.licence_report({"detector_detector": RF_MEDIUM})
    assert report["class"] == cl.SHIPPING and report["blocking"] == []
    cl.require_shipping(report)
    mixed = cl.licence_report({"detector_detector": RF_MEDIUM, "camera_keypoints": PNL_POINTS})
    assert mixed["class"] == cl.EVALUATION_ONLY and mixed["blocking"] == ["camera_keypoints"]


def test_unregistered_checkpoint_cannot_run():
    with pytest.raises(ValueError, match="not registered"):
        cl.licence_report({"detector_detector": "0" * 64})


def test_synthetic_adapters_report_no_checkpoints():
    report = cl.licence_report({})
    assert report["class"] == cl.NO_CHECKPOINTS
    with pytest.raises(ValueError, match="no checkpoints loaded"):
        cl.require_shipping(report)


def test_heatmap_calibrator_is_a_selectable_camera_backend():
    from soccerviz.vision.video_workflow import CALIBRATION_PROTOCOL, CALIBRATIONS

    assert CALIBRATIONS == ("pnl", "yolo", "heatmap")
    assert CALIBRATION_PROTOCOL["heatmap"] == {"input": 960, "landmark_confidence": 0.5}
