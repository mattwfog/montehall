import json

import numpy as np
import pandas as pd
import pytest

from soccerviz.datasets.soccernet_predictions import export_predictions


def fixture_run(tmp_path):
    video = tmp_path / "run"
    video.mkdir()
    (video / "report.json").write_text(
        json.dumps({"source_sha256": "synthetic-video", "model_sha256": {"detector": "synthetic"}})
    )
    frame_map = tmp_path / "frame-map.json"
    frame_map.write_text(
        json.dumps(
            {
                "sequence": "SNGS-fixture",
                "source_sha256": "synthetic-video",
                "frames": [
                    {
                        "video_frame": 0,
                        "source_frame": 6,
                        "image_id": "2100006",
                        "timestamp_s": 0,
                        "width": 100,
                        "height": 100,
                    }
                ],
            }
        )
    )
    pd.DataFrame(
        [{"frame_id": 0, "source_frame": 0, "timestamp_s": 0, "width": 100, "height": 100}]
    ).to_parquet(video / "frames.parquet")
    pd.DataFrame(
        [
            {
                "detection_id": "f0-d0",
                "frame_id": 0,
                "role_hypothesis": "player",
                "shot_id": 0,
                "tracklet_id": 17,
                "team_cluster": 1,
                "bbox_x0": 10,
                "bbox_y0": 20,
                "bbox_x1": 30,
                "bbox_y1": 50,
            }
        ]
    ).to_parquet(video / "detections.parquet")
    pd.DataFrame(
        [{"evidence_id": "f0-d0", "x_m": 20.0, "y_m": 50.0, "calibration_accepted": True}]
    ).to_parquet(video / "state.parquet")
    pd.DataFrame([{"frame_id": 0, "homography": np.eye(3).ravel().tolist()}]).to_parquet(
        video / "calibration.parquet"
    )
    return video, frame_map


def test_prediction_export_maps_frames_and_geometry_without_guessing_attributes(tmp_path):
    video, frame_map = fixture_run(tmp_path)
    output = tmp_path / "predictions.json"
    export_predictions(video, frame_map, output)
    record = json.loads(output.read_text())
    row = record["predictions"][0]
    assert row["image_id"] == "2100006"
    assert row["attributes"] == {"role": "player", "team": None, "jersey": None}
    assert row["bbox_pitch"]["x_bottom_middle"] == -32.5
    assert row["bbox_pitch"]["y_bottom_middle"] == 16
    assert row["bbox_pitch"]["x_bottom_left"] == -42.5
    assert record["provenance"]["frame_ids"] == [6]
    assert len(record["provenance"]["frames_sha256"]) == 64


def test_duplicate_mapping_rejected(tmp_path):
    video, frame_map = fixture_run(tmp_path)
    mapping = json.loads(frame_map.read_text())
    mapping["frames"].append(mapping["frames"][0])
    frame_map.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="Duplicate frame-map"):
        export_predictions(video, frame_map, tmp_path / "bad.json")


def test_export_rejects_wrong_source_and_preserves_unavailable_projection(tmp_path):
    video, frame_map = fixture_run(tmp_path)
    (video / "report.json").write_text(json.dumps({"source_sha256": "other"}))
    with pytest.raises(ValueError, match="source"):
        export_predictions(video, frame_map, tmp_path / "bad.json")
    (video / "report.json").write_text(json.dumps({"source_sha256": "synthetic-video"}))
    pd.DataFrame(
        [{"evidence_id": "f0-d0", "x_m": None, "y_m": None, "calibration_accepted": False}]
    ).to_parquet(video / "state.parquet")
    export_predictions(video, frame_map, tmp_path / "unknown.json")
    assert (
        json.loads((tmp_path / "unknown.json").read_text())["predictions"][0]["bbox_pitch"] is None
    )
