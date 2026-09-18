import argparse
import json

import pytest

from soccerviz.cli.datasets import configure, dispatch
from soccerviz.datasets.dataset_catalog import DatasetCatalog


def test_report_snapshot_idempotence_and_mutated_file_detection(tmp_path):
    catalog = DatasetCatalog(tmp_path / "store")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"split": "test", "score": 0.3, "source_sha256": "fixture"}))
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("frame,track\n1,4\n")
    first = catalog.register("fixture", "tracking", "test", report, [predictions])
    second = catalog.register("fixture", "tracking", "test", report, [predictions])
    assert first["id"] == second["id"]
    assert len(catalog.list()) == 1
    assert catalog.verify(first["id"])["valid"]
    predictions.write_text("frame,track\n1,9\n")
    assert not catalog.verify(first["id"])["valid"]
    report.write_text(json.dumps({"score": 0.9}))
    assert catalog.get(first["id"])["report"]["score"] == 0.3
    new = catalog.register("fixture", "tracking", "test", report, [predictions])
    assert new["id"] != first["id"]
    assert len(catalog.list()) == 2


def test_no_partial_registration_when_artifact_missing(tmp_path):
    catalog = DatasetCatalog(tmp_path / "store")
    report = tmp_path / "report.json"
    report.write_text("{}")
    with pytest.raises(ValueError, match="not a file"):
        catalog.register("fixture", "tracking", "test", report, [tmp_path / "missing"])
    assert catalog.list() == []


def test_cli_verification_exits_nonzero_on_changed_file(tmp_path, capsys):
    report = tmp_path / "report.json"
    report.write_text("{}")
    catalog = DatasetCatalog(tmp_path / "store")
    key = catalog.register("fixture", "tracking", "test", report)["id"]
    report.unlink()
    args = configure(argparse.ArgumentParser()).parse_args(
        ["--store", str(tmp_path / "store"), "verify", key]
    )
    with pytest.raises(SystemExit) as error:
        dispatch(args)
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False
