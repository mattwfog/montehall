import json

import numpy as np
import pytest

from soccerviz.candidates.calibration_model import (
    centered_homography,
    checked_homography,
    evaluate_calibrations,
    ground_homography_from_projection,
    load_sample,
    make_protocol,
)
from soccerviz.core.geometry import project
from soccerviz.datasets.soccernet_adapter import sha256, write_json


def test_centered_pitch_transform_preserves_corners_and_orientation():
    matrix = centered_homography(np.eye(3))
    points = project([[0, 0], [105, 68], [52.5, 34], [105, 0]], matrix)
    np.testing.assert_allclose(points, [[-52.5, -34], [52.5, 34], [0, 0], [52.5, -34]])


def test_upstream_projection_ground_plane_inversion():
    world_to_image = np.array([[10, 2, 960], [0.5, 7, 540], [0.002, 0.004, 1]])
    projection = np.column_stack(
        [world_to_image[:, 0], world_to_image[:, 1], [0.5, 0.1, 0.3], world_to_image[:, 2]]
    )
    centered_world = [[-52.5, -34], [52.5, 34], [0, 0], [18, -12]]
    image_points = project(centered_world, world_to_image)
    inverse = ground_homography_from_projection(projection)
    np.testing.assert_allclose(project(image_points, inverse), centered_world, atol=1e-10)


@pytest.mark.parametrize(
    "matrix", [np.zeros((3, 3)), np.eye(4), [[1, 0, 0], [0, 1, 0], [0, 0, np.nan]]]
)
def test_bad_homographies_rejected(matrix):
    with pytest.raises(ValueError):
        checked_homography(matrix)


@pytest.fixture
def calibration_corpus(tmp_path):
    image = tmp_path / "source.jpg"
    image.write_bytes(b"source image integrity fixture")
    frames = [
        {
            "sequence": "SNGS-021",
            "frame_id": i,
            "image_path": str(image),
            "sha256": sha256(image),
            "width": 1920,
            "height": 1080,
        }
        for i in range(1, 22)
    ]
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"purpose": "development", "split": "valid", "frames": frames})
    protocol = tmp_path / "protocol.json"
    make_protocol(manifest, protocol)
    source = tmp_path / "Labels-GameState.json"
    write_json(
        source,
        {
            "images": [{"image_id": str(i), "file_name": f"{i:06d}.jpg"} for i in range(1, 22)],
            "annotations": [
                {
                    "image_id": str(i),
                    "category_id": 1,
                    "bbox_image": {"x": 95, "y": 180, "w": 10, "h": 20},
                    "bbox_pitch": {"x_bottom_middle": -42.5, "y_bottom_middle": -14.0},
                }
                for i in (1, 11, 21)
            ],
        },
    )
    write_json(
        tmp_path / "ground-truth.json",
        {
            "manifest_sha256": sha256(manifest),
            "sources": [{"sequence": "SNGS-021", "path": str(source), "sha256": sha256(source)}],
        },
    )
    prediction_path = tmp_path / "predictions.json"

    def score(predictions, **overrides):
        write_json(
            prediction_path,
            {
                "schema": "calibration-predictions/v1",
                "backend": "test-only",
                "manifest_sha256": sha256(manifest),
                "protocol_sha256": sha256(protocol),
                "frames": predictions,
                **overrides,
            },
        )
        return evaluate_calibrations(manifest, protocol, prediction_path)

    return manifest, protocol, source, score


def camera_prediction(frame_id, dx=0.0):
    return {
        "sequence": "SNGS-021",
        "frame_id": frame_id,
        "accepted": True,
        "reason": "accepted",
        "homography": [[0.1, 0, -52.5 + dx], [0, 0.1, -34], [0, 0, 1]],
    }


def test_freeze_selection_and_reject_overwrite(calibration_corpus):
    manifest, protocol, _, _ = calibration_corpus
    _, frames = load_sample(manifest, protocol)
    assert [frame["frame_id"] for frame in frames] == [1, 11, 21]
    with pytest.raises(FileExistsError):
        make_protocol(manifest, protocol)


def test_oracle_footpoint_calibration_with_missing_frames(calibration_corpus):
    score = calibration_corpus[3]
    result = score([camera_prediction(1), camera_prediction(11, dx=1)])
    metrics = result["overall"]
    assert metrics["frame_coverage"] == pytest.approx(2 / 3)
    assert metrics["projection_coverage"] == pytest.approx(2 / 3)
    assert metrics["accepted_point_error"]["median_m"] == pytest.approx(0.5)
    assert metrics["all_target_fraction_within_2m"] == pytest.approx(2 / 3)
    assert result["missing_prediction_frames"] == 1
    assert result["oracle_image_footpoints"]
    assert not result["ground_truth_used_for_fitting"]


def test_large_errors_remain_scored(calibration_corpus):
    result = calibration_corpus[3]([camera_prediction(i, dx=1000) for i in (1, 11, 21)])
    assert result["overall"]["accepted_point_error"]["median_m"] == pytest.approx(1000)
    assert result["overall"]["frame_coverage"] == 1
    assert result["overall"]["all_target_fraction_within_2m"] == 0


def test_rejected_frames_are_explicit(calibration_corpus):
    result = calibration_corpus[3](
        [
            {
                "sequence": "SNGS-021",
                "frame_id": 1,
                "accepted": False,
                "reason": "upstream_no_camera",
                "homography": None,
            }
        ]
    )
    assert result["overall"]["accepted_point_error"]["median_m"] is None
    assert result["overall"]["failure_reasons"] == {
        "upstream_no_camera": 1,
        "missing_prediction_frame": 2,
    }
    assert result["overall"]["valid_target_points"] == 3
    assert result["overall"]["unavailable_points"] == 3


def test_provenance_mismatch_rejected(calibration_corpus):
    with pytest.raises(ValueError, match="provenance mismatch"):
        calibration_corpus[3]([], manifest_sha256="wrong")


def test_source_annotation_mutation_rejected(calibration_corpus):
    _, _, source, score = calibration_corpus
    data = json.loads(source.read_text())
    data["annotations"][0]["bbox_pitch"]["x_bottom_middle"] = 999
    write_json(source, data)
    with pytest.raises(ValueError, match="annotation SHA256"):
        score([])


def test_unexpected_frame_rejected(calibration_corpus):
    with pytest.raises(ValueError, match="Unexpected"):
        calibration_corpus[3]([camera_prediction(2)])
