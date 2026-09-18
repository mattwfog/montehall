"""Immutable public-dataset experiment records in the harness evidence store."""

from __future__ import annotations

import json
from pathlib import Path

from soccerviz.core.assets import sha256
from soccerviz.harness.engine import Harness, now


class DatasetCatalog:
    def __init__(self, store):
        self.harness = Harness(Path(store))
        with self.harness.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS dataset_records ("
                "id TEXT PRIMARY KEY REFERENCES objects(id), name TEXT NOT NULL, "
                "kind TEXT NOT NULL, split TEXT NOT NULL, created_at TEXT NOT NULL)"
            )

    def register(self, name, kind, split, report_path, artifacts=()):
        """Snapshot a report; hash explicit local artifacts without copying large datasets."""
        if not all(isinstance(value, str) and value.strip() for value in (name, kind, split)):
            raise ValueError("Name, kind and split description are required")
        report_path = Path(report_path).resolve()
        # Parse the same bytes whose hash we retain, avoiding a second report read.
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)
        if not isinstance(report, dict):
            raise TypeError("Dataset report must be a JSON object")
        import hashlib

        report_hash = hashlib.sha256(report_bytes).hexdigest()
        paths = sorted({report_path, *(Path(p).resolve() for p in artifacts)})
        files = []
        for path in paths:
            if not path.is_file():
                raise ValueError(f"Artifact is not a file: {path}")
            before = path.stat()
            file_hash = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Artifact changed during registration: {path}")
            if path == report_path and file_hash != report_hash:
                raise ValueError("Report changed during registration")
            files.append({"path": str(path), "sha256": file_hash, "bytes": after.st_size})
        record = {
            "schema": "dataset-experiment/v1",
            "name": name.strip(),
            "kind": kind.strip(),
            "split": split.strip(),
            "report": report,
            "artifacts": files,
            "verification_scope": "report_and_explicit_files" if len(files) > 1 else "report_only",
            "storage_policy": "Report snapshot is immutable; external files are hash-verified references",
            "interpretation": "Registration records provenance, not scientific validation or video-state alignment",
        }
        with self.harness.connect() as db:
            key = self.harness.put(db, record)
            db.execute(
                "INSERT OR IGNORE INTO dataset_records VALUES (?, ?, ?, ?, ?)",
                (key, record["name"], record["kind"], record["split"], now()),
            )
        return {"id": key, "record": record}

    def list(self):
        with self.harness.connect() as db:
            return [
                dict(row)
                for row in db.execute("SELECT * FROM dataset_records ORDER BY created_at DESC, id")
            ]

    def get(self, key):
        with self.harness.connect() as db:
            found = db.execute("SELECT id FROM dataset_records WHERE id=?", (key,)).fetchone()
        if not found:
            raise ValueError("Unknown dataset experiment")
        return self.harness.get(key)

    def verify(self, key):
        record = self.get(key)
        checked = []
        for entry in record["artifacts"]:
            path = Path(entry["path"])
            valid = (
                path.is_file()
                and path.stat().st_size == entry["bytes"]
                and sha256(path) == entry["sha256"]
            )
            checked.append({"path": str(path), "matches_registered_hash": valid})
        return {
            "id": key,
            "verification_scope": record.get(
                "verification_scope", "explicit_registered_files_only"
            ),
            "valid": all(row["matches_registered_hash"] for row in checked),
            "artifacts": checked,
        }
