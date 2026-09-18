"""Download Roboflow Universe datasets (COCO export) for the detector merge.

Fetches the latest generated version of each named project, skipping any
dataset already on disk (re-run = resume). Logs each project's declared
license — the due-diligence record for uploader-declared CC BY 4.0 sets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def fetch(api_key: str, projects: list[str], out_root: Path) -> dict:
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    out_root.mkdir(parents=True, exist_ok=True)
    results = {}
    started = time.monotonic()
    for ref in projects:
        workspace, slug = ref.split("/", 1)
        dest = out_root / slug
        if (dest / "train" / "_annotations.coco.json").exists():
            results[ref] = {"status": "already-present"}
            continue
        project = rf.workspace(workspace).project(slug)
        versions = project.versions()
        if not versions:
            results[ref] = {"status": "NO-VERSIONS"}
            continue
        latest = max(versions, key=lambda v: int(str(v.version).rsplit("/", 1)[-1]))
        latest.download("coco", location=str(dest))
        results[ref] = {
            "status": "downloaded",
            "version": str(latest.version),
            "license": getattr(project, "license", None),
            "images": getattr(latest, "images", None),
        }
    return {"projects": results, "wall_seconds": round(time.monotonic() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--project", action="append", required=True,
                        help="workspace/project-slug (repeatable)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    api_key = args.api_key_file.read_text().strip()
    print(json.dumps(fetch(api_key, args.project, args.out), indent=2))


if __name__ == "__main__":
    main()
