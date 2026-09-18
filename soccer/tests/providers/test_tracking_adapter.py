import numpy as np
import pandas as pd
import pytest

from soccerviz.providers.tracking import BOX, diagnostics, track_frozen, validate_inputs


def fixture():
    frames = pd.DataFrame(
        {
            "frame_id": [0, 1, 2, 3, 4],
            "timestamp_s": [7.0, 7.2, 7.4, 7.6, 7.8],
            "shot_id": [0, 0, 0, 0, 1],
        }
    )
    rows = []
    for i in [0, 1, 3, 4]:
        for role, x in [("player", 10), ("referee", 100)]:
            rows.append(
                {
                    "detection_id": f"{i}-{role}",
                    "frame_id": i,
                    "timestamp_s": 7 + i * 0.2,
                    "shot_id": int(i == 4),
                    "bbox_x0": x + i,
                    "bbox_y0": 10.0,
                    "bbox_x1": x + i + 20,
                    "bbox_y1": 60.0,
                    "confidence": 0.9,
                    "role_hypothesis": role,
                    "tracklet_id": 77,
                    "team_cluster": 0,
                }
            )
    return frames, pd.DataFrame(rows)


@pytest.mark.parametrize("backend", ["soccerviz", "tracklab-ocsort"])
def test_frozen_tracking_preserves_evidence_and_resets_shots(backend):
    if backend == "tracklab-ocsort":
        pytest.importorskip("oc_sort")
        pytest.importorskip("torch")
    frames, detections = fixture()
    result = track_frozen(frames, detections, backend)
    pd.testing.assert_frame_equal(
        result[BOX + ["detection_id", "timestamp_s"]],
        detections[BOX + ["detection_id", "timestamp_s"]],
    )
    assert result.team_cluster.isna().all()
    assert result[result.shot_id == 0].tracklet_id.nunique() == 2
    assert set(result[result.shot_id == 0].tracklet_id).isdisjoint(
        result[result.shot_id == 1].tracklet_id
    )
    assert diagnostics(result)["idf1"] is None
    assert detections.team_cluster.eq(0).all()


def test_bad_evidence_fails_closed():
    frames, detections = fixture()
    detections.loc[0, "timestamp_s"] = 0
    with pytest.raises(ValueError, match="clock"):
        validate_inputs(frames, detections)
    detections.loc[0, "timestamp_s"] = 7
    detections.loc[0, "bbox_x1"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_inputs(frames, detections)


def test_ocsort_rejects_irregular_sampling():
    pytest.importorskip("oc_sort")
    pytest.importorskip("torch")
    frames, detections = fixture()
    frames.loc[2, "timestamp_s"] = 7.45
    with pytest.raises(ValueError, match="regular sampling"):
        track_frozen(frames, detections)


def test_comparison_exports_evaluation_envelopes_with_identical_frozen_boxes(tmp_path):
    import json

    from soccerviz.providers.tracking import compare_trackers

    pytest.importorskip("oc_sort")
    pytest.importorskip("torch")
    frames, detections = fixture()
    source = tmp_path / "source"
    source.mkdir()
    frames.to_parquet(source / "frames.parquet", index=False)
    detections.to_parquet(source / "detections.parquet", index=False)
    state = detections.rename(columns={"detection_id": "evidence_id"})[
        ["evidence_id", "frame_id", "tracklet_id", "team_cluster"]
    ]
    state.to_parquet(source / "state.parquet", index=False)
    (source / "report.json").write_text(json.dumps({"source_sha256": "a" * 64}))
    result = compare_trackers(source, tmp_path / "comparison")
    expected = [
        (int(row.frame_id), row.role_hypothesis, tuple(getattr(row, k) for k in BOX))
        for row in detections.itertuples()
    ]
    for backend in ["soccerviz", "tracklab-ocsort"]:
        envelope = json.loads((tmp_path / "comparison" / backend / "predictions.json").read_text())
        assert envelope["source_sha256"] == "a" * 64
        assert envelope["frozen_detection_sha256"] == result["frozen_detection_sha256"]
        assert [
            (r["frame_id"], r["label"], tuple(r["bbox"])) for r in envelope["predictions"]
        ] == expected
        assert len({(r["frame_id"], r["track_id"]) for r in envelope["predictions"]}) == len(
            expected
        )
        assert result["comparison"][backend]["predictions_path"].endswith(
            f"{backend}/predictions.json"
        )
        assert result["comparison"][backend]["hota"] is None
