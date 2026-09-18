import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from soccerviz.candidates.temporal_ball import (
    frame_histogram,
    histogram_cut_distance,
    native_candidates,
    plan_windows,
    source_fingerprint,
)
from soccerviz.core.assets import sha256


def frame(identifier, sequence="s", **overrides):
    return {
        "sequence": sequence,
        "frame_id": identifier,
        "fps": 25.0,
        "timestamp_s": (identifier - 1) / 25,
        "width": 30,
        "height": 20,
        **overrides,
    }


def test_windows_preserve_all_three_output_slots_and_leave_segment_tails():
    frames = [frame(i, s) for s in ("a", "b") for i in range(1, 6)]
    windows, statuses = plan_windows(frames)
    assert [w["indices"] for w in windows] == [[0, 1, 2], [5, 6, 7]]
    assert windows[0]["segment"] != windows[1]["segment"]
    assert [statuses[i] for i in (3, 4, 8, 9)] == ["incomplete_context"] * 4


def test_windows_do_not_compress_missing_source_frames_or_timestamps():
    frames = [frame(i) for i in [1, 2, 3, 5, 6, 7]]
    windows, _ = plan_windows(frames)
    assert [w["indices"] for w in windows] == [[0, 1, 2], [3, 4, 5]]
    assert windows[0]["segment"] != windows[1]["segment"]
    # Consecutive identifiers alone are insufficient when time has a discontinuity.
    frames = [frame(1), frame(2, timestamp_s=0.20), frame(3, timestamp_s=0.24)]
    assert plan_windows(frames)[0] == []


def test_sampled_frames_are_not_treated_as_consecutive_video():
    frames = [frame(i) for i in range(1, 40, 5)]
    assert plan_windows(frames)[0] == []


def test_explicit_cuts_and_missing_images_reset_context():
    frames = [frame(i) for i in range(1, 8)]
    windows, statuses = plan_windows(frames, cut_before={("s", 3)})
    assert [w["indices"] for w in windows] == [[2, 3, 4]]
    assert statuses[0] == statuses[1] == statuses[5] == statuses[6] == "incomplete_context"
    windows, statuses = plan_windows(frames, unavailable={("s", 2)})
    assert [w["indices"] for w in windows] == [[2, 3, 4]]
    assert statuses[1] == "image_unavailable"


def test_shot_id_cut_marker_and_dimensions_are_boundaries():
    for boundary in ({"shot_id": "next"}, {"cut_before": True}, {"width": 60}):
        frames = [frame(1), frame(2), frame(3, **boundary)]
        assert plan_windows(frames)[0] == []


def test_duplicate_frame_ids_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        plan_windows([frame(1), frame(1), frame(2)])


def test_native_centers_are_not_boxes_or_blob_mass_confidences():
    hm = np.zeros((5, 6), dtype=np.float32)
    hm[1, 1:3] = [0.8, 0.9]
    result = {"hm": hm, "xys": [np.array([14.0, 10.0])], "scores": [1.7]}
    candidates = native_candidates(result, 30, 20)
    assert candidates[0]["center_xy"] == [14.0, 10.0]
    assert candidates[0]["native_blob_score"] == 1.7
    assert candidates[0]["score"] == pytest.approx(0.9)
    assert "bbox_xyxy" not in candidates[0]
    assert native_candidates({**result, "xys": [np.array([-1, 10])]}, 30, 20) == []


def test_native_component_output_must_align_and_be_finite():
    hm = np.array([[0.8, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="misaligned"):
        native_candidates({"hm": hm, "xys": [], "scores": []}, 30, 20)
    with pytest.raises(ValueError, match="invalid sigmoid"):
        native_candidates({"hm": hm * np.nan, "xys": [], "scores": []}, 30, 20)


def test_cut_guard_distinguishes_changed_frame_colors():
    green = np.full((20, 30, 3), [0, 255, 0], dtype=np.uint8)
    blue = np.full((20, 30, 3), [255, 0, 0], dtype=np.uint8)
    assert histogram_cut_distance(frame_histogram(green), frame_histogram(green)) == 0
    assert histogram_cut_distance(frame_histogram(green), frame_histogram(blue)) > 0.65


def test_source_fingerprint_catches_changed_code(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    file = src / "a.py"
    file.write_text("initial")
    before = source_fingerprint(tmp_path)
    file.write_text("changed")
    assert source_fingerprint(tmp_path) != before


def test_source_fingerprint_ignores_only_validated_appledouble_metadata(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "real.py").write_text("real source")
    expected = source_fingerprint(tmp_path)
    sidecar = source / "._real.py"
    sidecar.write_bytes(b"\x00\x05\x16\x07\x00\x02\x00\x00resource metadata")
    assert source_fingerprint(tmp_path) == expected
    sidecar.write_text("ordinary extra code is not metadata")
    assert source_fingerprint(tmp_path) != expected
    sidecar.write_bytes(b"\x00\x05\x16\x07metadata")
    (source / "real.py").write_text("modified source")
    assert source_fingerprint(tmp_path) != expected


def test_runner_assigns_heatmaps_to_correct_source_frames(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "temporal_runner",
        Path(__file__).resolve().parents[2] / "scripts/benchmarks/run_temporal_ball_benchmark.py",
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    frames = []
    for identifier in range(1, 8):
        path = tmp_path / f"{identifier}.png"
        cv2.imwrite(str(path), np.full((20, 30, 3), identifier, dtype=np.uint8))
        frames.append(frame(identifier, image_path=str(path), sha256=sha256(path)))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "vision-benchmark/v1", "frames": frames}))

    class FakeWASB:
        def __init__(self, *args, **kwargs):
            self.metadata = {"name": "fixture"}
            self.last_timing_ms = {"forward": 1.0, "total": 2.0}

        def reset_tracker(self):
            pass

        def predict(self, images):
            return [
                [{"label": "ball", "center_xy": [int(im[0, 0, 0]), 10], "score": 0.9}]
                for im in images
            ], {}

        def select(self, candidates):
            return candidates

    monkeypatch.setattr(runner, "WASBDetector", FakeWASB)
    args = runner.parse_args(
        [
            "--manifest",
            str(manifest),
            "--checkpoint",
            "unused.pt",
            "--out",
            str(tmp_path / "result.json"),
            "--device",
            "cpu",
        ]
    )
    result = runner.run(args)
    assert result["manifest_sha256"] == sha256(manifest)
    assert [f["detections"][0]["center_xy"][0] for f in result["frames"][:6]] == [1, 2, 3, 4, 5, 6]
    assert [f["lookahead_frames"] for f in result["frames"][:6]] == [2, 1, 0, 2, 1, 0]
    assert result["frames"][3]["context_frame_ids"] == [4, 5, 6]
    assert result["frames"][6]["detections"] == []
    assert result["frames"][6]["context_status"] == "incomplete_context"
    assert result["coverage"]["s"]["statuses"] == {"inferred": 6, "incomplete_context": 1}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        runner.run(args)
