import json
import zipfile

import pytest

from soccerviz.datasets import roboflow_universe as rf

EXPORT = {
    "project": {
        "id": "roboflow-jvuqo/football-players-detection-3zvbc",
        "type": "object-detection",
        "images": 372,
        "public": True,
        "license": "CC BY 4.0",
        "classes": {"ball": 396, "player": 12063},
    },
    "version": {
        "id": "roboflow-jvuqo/football-players-detection-3zvbc/20",
        "name": "rf-detr-m",
        "created": 1754081150.722,
        "images": 372,
        "splits": {"train": 298, "valid": 49, "test": 25},
        "preprocessing": {"resize": {"enabled": True, "format": "Stretch to", "width": "576"}},
        "augmentation": {},
    },
    "export": {"format": "coco", "link": "https://app.roboflow.com/ds/abc?key=SECRET", "size": "19.5"},
}


def fake_zip(path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("train/a.jpg", b"jpg")
        zf.writestr("train/_annotations.coco.json", "{}")
        zf.writestr("valid/b.jpg", b"jpg")
        zf.writestr("valid/_annotations.coco.json", "{}")


def test_download_records_source_and_redacts_key(tmp_path):
    calls = []

    def fetch(url, key):
        calls.append((url, key))
        return EXPORT

    record = rf.download_version(
        "roboflow-jvuqo/football-players-detection-3zvbc",
        20,
        "coco",
        tmp_path,
        "k",
        fetch=fetch,
        opener=lambda link, dest: fake_zip(dest),
    )
    assert calls == [(f"{rf.API}/roboflow-jvuqo/football-players-detection-3zvbc/20/coco", "k")]
    assert record["export_link"] == "https://app.roboflow.com/ds/abc?key=REDACTED"
    assert "SECRET" not in json.dumps(record)
    assert record["extracted"] == {
        "images": 2,
        "label_files": 2,
        "images_by_split": {"train": 1, "valid": 1},
    }
    assert record["preprocessing"]["resize"]["format"] == "Stretch to"
    assert record["license"] == "CC BY 4.0"
    folder = tmp_path / "football-players-detection-3zvbc-v20-coco"
    assert (folder / "source.json").exists()
    assert record["archive_sha256"] == rf.sha256(folder.with_suffix(".zip"))


def test_rerun_reuses_archive(tmp_path):
    downloads = []
    opener = lambda link, dest: (downloads.append(link), fake_zip(dest))
    args = ("roboflow-jvuqo/football-field-detection-f07vi", 12, "coco", tmp_path, "k")
    first = rf.download_version(*args, fetch=lambda u, k: EXPORT, opener=opener)
    second = rf.download_version(*args, fetch=lambda u, k: EXPORT, opener=opener)
    assert len(downloads) == 1
    assert first["archive_sha256"] == second["archive_sha256"]


@pytest.mark.parametrize(
    "project, version, fmt",
    [("bad project", 1, "coco"), ("ws/proj", 0, "coco"), ("ws/proj", 1, "tfrecord")],
)
def test_rejects_invalid_requests(tmp_path, project, version, fmt):
    with pytest.raises(ValueError):
        rf.download_version(project, version, fmt, tmp_path, "k", fetch=lambda u, k: EXPORT)


def test_key_comes_from_environment_only(monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        rf.run({"datasets": [{"project": "ws/proj", "version": 1}]})
