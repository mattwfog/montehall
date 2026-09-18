"""Exercise the real command surface using synthetic annotations, never demo truth."""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from soccerviz.core.assets import sha256
from soccerviz.harness.engine import Harness


def run_cli(store, *args, ok=True):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "soccerviz.cli.integrations",
            "--store",
            str(store),
            *map(str, args),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if ok:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
    return result


def source():
    return {
        "schema": "video-input/v1",
        "window": {"start_s": 0.0, "end_s": 0.2},
        "source_sha256": "synthetic-test-video-only",
        "model_sha256": {},
        "tables": {
            "frames": [
                {
                    "frame_id": 0,
                    "source_frame": 0,
                    "timestamp_s": 0.0,
                    "pts": 0,
                    "time_base": "1/25",
                    "width": 100,
                    "height": 100,
                    "shot_id": 0,
                }
            ],
            "detections": [
                {
                    "frame_id": 0,
                    "timestamp_s": 0.0,
                    "detection_id": "f0-d0",
                    "tracklet_id": 1,
                    "role_hypothesis": "player",
                    "bbox_x0": 10,
                    "bbox_y0": 10,
                    "bbox_x1": 20,
                    "bbox_y1": 30,
                    "confidence": 0.9,
                    "source": "synthetic_fixture",
                }
            ],
            "state": [],
            "calibration": [],
            "ball_candidates": [],
        },
    }


def prepared(tmp_path, *, reviewed):
    store = tmp_path / "store"
    harness = Harness(store)
    run = harness.create(source())
    out = tmp_path / "exchange"
    run_cli(store, "cvat-export", run, "--out", out)
    xml = out / "annotations.xml"
    tree = ET.parse(xml)
    box = tree.find(".//box")
    if reviewed:
        box.find("attribute[@name='reviewed']").text = "true"
        box.find("attribute[@name='identity']").text = "synthetic-player"
    tree.write(xml)
    review = json.loads((out / "review-template.json").read_text())
    review.update(
        annotations_sha256=sha256(xml),
        reviewer="synthetic reviewer",
        reason="Synthetic end-to-end fixture",
        reviewed_frames=[{"frame_id": 0, "labels": ["player"], "exhaustive": True}],
    )
    review_path = out / "review.json"
    review_path.write_text(json.dumps(review))
    imported = out / "reviewed-tracking.json"
    args = (
        "cvat-import",
        run,
        "--xml",
        xml,
        "--manifest",
        out / "manifest.json",
        "--review",
        review_path,
        "--out",
        imported,
    )
    return harness, run, imported, args


def test_cli_import_preview_then_idempotent_apply(tmp_path):
    harness, run, imported, args = prepared(tmp_path, reviewed=True)
    preview = run_cli(harness.root, *args, "--preview")
    assert json.loads(preview.stdout)["preview"] is True
    assert harness.context(run)["revision"] == 0
    first = json.loads(run_cli(harness.root, *args).stdout)
    second = json.loads(run_cli(harness.root, *args).stdout)
    assert first["receipt"]["count"] == 1
    assert second["receipt"]["reused"] is True
    assert harness.context(run)["revision"] == 1
    assert len(harness.status(run)["attachments"]) == 1
    payload = json.loads(imported.read_text())
    assert payload["ground_truth"][0]["reviewed"] is True
    assert harness.context(run, 0)["reviews"] == []


def test_cli_rejects_unreviewed_boxes_without_creating_review(tmp_path):
    harness, run, imported, args = prepared(tmp_path, reviewed=False)
    result = run_cli(harness.root, *args, ok=False)
    assert "unreviewed" in result.stderr
    assert harness.context(run)["revision"] == 0
    assert not imported.exists()
    assert harness.status(run)["attachments"] == []


def test_cli_official_evaluation_attaches_measured_result(tmp_path):
    pytest.importorskip("trackeval")
    harness, run, imported, args = prepared(tmp_path, reviewed=True)
    run_cli(harness.root, *args)
    report = tmp_path / "tracking-report.json"
    run_cli(harness.root, "evaluate", run, "--reviewed", imported, "--out", report)
    measured = json.loads(report.read_text())
    assert measured["classes"]["player"]["IDF1"] == 1.0
    assert measured["classes"]["player"]["HOTA"] == 1.0
    assert len(harness.status(run)["attachments"]) == 2


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "another video"),
        ({"source_sha256": "another-video", "predictions": []}, "another video"),
    ],
)
def test_cli_rejects_alternative_predictions_without_matching_provenance(
    tmp_path, payload, message
):
    harness, run, imported, args = prepared(tmp_path, reviewed=True)
    run_cli(harness.root, *args, "--preview")
    predictions = tmp_path / "alternative-predictions.json"
    predictions.write_text(json.dumps(payload))
    report = tmp_path / "tracking-report.json"
    result = run_cli(
        harness.root,
        "evaluate",
        run,
        "--reviewed",
        imported,
        "--predictions",
        predictions,
        "--out",
        report,
        ok=False,
    )
    assert message in result.stderr
    assert not report.exists()
    assert harness.status(run)["attachments"] == []


def test_isolated_evaluation_preserves_python_symlink(tmp_path, monkeypatch):
    import argparse
    from types import SimpleNamespace

    from soccerviz.cli import integrations as integrations_cli

    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    commands = []

    def capture(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="{}\n")

    monkeypatch.setattr(integrations_cli.subprocess, "run", capture)
    args = integrations_cli.configure(argparse.ArgumentParser()).parse_args(
        [
            "--store",
            str(tmp_path / "store"),
            "evaluate",
            "synthetic-run",
            "--reviewed",
            str(tmp_path / "reviewed.json"),
            "--out",
            str(tmp_path / "report.json"),
            "--python",
            str(interpreter),
        ]
    )
    integrations_cli.dispatch(args)
    assert commands[0][0] == str(interpreter.absolute())
    assert commands[0][0] != str(interpreter.resolve())


def test_core_cli_dispatches_real_persistent_evaluation_runtime(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    core = root / ".venv/bin/python"
    evaluation = root / ".venvs/evaluation/bin/python"
    if not core.exists() or not evaluation.exists():
        pytest.skip("Persistent core and evaluation environments are required")
    harness, run, imported, args = prepared(tmp_path, reviewed=True)
    run_cli(harness.root, *args, "--preview")
    report = tmp_path / "worker-report.json"
    completed = subprocess.run(
        [
            str(core),
            "-m",
            "soccerviz.cli.integrations",
            "--store",
            str(harness.root),
            "evaluate",
            run,
            "--reviewed",
            str(imported),
            "--out",
            str(report),
            "--python",
            str(evaluation),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ | {"PYTHONPATH": str(root / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    measured = json.loads(report.read_text())
    assert measured["classes"]["player"]["HOTA"] == 1
    assert measured["classes"]["player"]["IDF1"] == 1
    assert "12c8791b303e0a0b50f753af204249e622d0281a" in measured["trackeval_direct_url"]
    assert len(harness.status(run)["attachments"]) == 1
