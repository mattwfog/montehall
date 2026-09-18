"""Pinned Roboflow Universe dataset downloads with recorded source metadata.

Requests name a public project, an exact version and an export format. The API
key is read from the environment only (never from the request), the export
link is redacted from the response, and every archive is hashed before it is
extracted. Version preprocessing (for example "Stretch to 576x576") is recorded
because it changes the pixel geometry of the images relative to the source.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
import zipfile
from pathlib import Path

from soccerviz.datasets.soccernet_adapter import sha256, write_json

API = "https://api.roboflow.com"
DEFAULT_KEY_ENV = "ROBOFLOW_API_KEY"
PROJECT_ID = re.compile(r"^[a-z0-9-]+/[a-z0-9-]+$")
FORMATS = {"coco", "yolov8", "yolov11", "voc", "folder"}


def api_key(env_name: str = DEFAULT_KEY_ENV) -> str:
    value = os.environ.get(env_name, "")
    if not value:
        raise RuntimeError(f"Set {env_name} in the environment; keys are never stored in requests")
    return value


def fetch_json(url: str, key: str) -> dict:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def redact(value: str) -> str:
    return re.sub(r"([?&]key=)[^&]+", r"\1REDACTED", value)


def describe_version(export: dict) -> dict:
    project = export["project"]
    version = export["version"]
    return {
        "project_id": project["id"],
        "project_type": project["type"],
        "license": project.get("license"),
        "public": project.get("public"),
        "project_images": project.get("images"),
        "classes": project.get("classes"),
        "version_id": version["id"],
        "version_name": version.get("name"),
        "version_created": version.get("created"),
        "version_images": version.get("images"),
        "splits": version.get("splits"),
        "preprocessing": version.get("preprocessing"),
        "augmentation": version.get("augmentation"),
    }


def count_extracted(folder: Path) -> dict:
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    labels = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".json", ".txt", ".xml"})
    by_split = {}
    for image in images:
        split = image.relative_to(folder).parts[0]
        by_split[split] = by_split.get(split, 0) + 1
    return {"images": len(images), "label_files": len(labels), "images_by_split": by_split}


def download_version(
    project: str, version: int, fmt: str, out: Path, key: str, fetch=fetch_json, opener=None
) -> dict:
    if not PROJECT_ID.match(project):
        raise ValueError(f"Project must be workspace/project, got {project!r}")
    if fmt not in FORMATS:
        raise ValueError(f"Unsupported export format {fmt!r}; choose from {sorted(FORMATS)}")
    if not isinstance(version, int) or version < 1:
        raise ValueError("Version must be a positive integer")
    export = fetch(f"{API}/{project}/{version}/{fmt}", key)
    if "export" not in export or "link" not in export["export"]:
        raise RuntimeError(f"Roboflow returned no export link for {project}/{version}/{fmt}")
    folder = out / f"{project.split('/')[1]}-v{version}-{fmt}"
    archive = folder.with_suffix(".zip")
    folder.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix(".zip.download")
        (opener or urllib.request.urlretrieve)(export["export"]["link"], temporary)
        temporary.replace(archive)
    digest = sha256(archive)
    if not folder.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(folder)
    record = describe_version(export)
    record.update(
        {
            "format": fmt,
            "export_link": redact(export["export"]["link"]),
            "export_size_mb": export["export"].get("size"),
            "archive": str(archive),
            "archive_sha256": digest,
            "archive_bytes": archive.stat().st_size,
            "folder": str(folder),
            "extracted": count_extracted(folder),
        }
    )
    write_json(folder / "source.json", record)
    return record


def run(request: dict) -> dict:
    if request.get("schema_version", 1) != 1:
        raise ValueError("Unsupported request schema_version")
    key = api_key(request.get("api_key_env", DEFAULT_KEY_ENV))
    out = Path(request.get("out", "artifacts/public-data/roboflow"))
    datasets = request["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Request needs a non-empty datasets list")
    records = [
        download_version(d["project"], d["version"], d.get("format", "coco"), out, key)
        for d in datasets
    ]
    return {"schema_version": 1, "out": str(out), "datasets": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args()
    write_json(args.response, run(json.loads(args.request.read_text())))


if __name__ == "__main__":
    main()
