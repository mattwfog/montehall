"""Fetch explicit public specialist checkpoints and retain their source and hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

SOURCES = {
    "pnl-keypoints": "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp",
    "pnl-lines": "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines",
    "wasb-soccer": "https://drive.google.com/uc?id=1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l",
    "jersey-vitb": "https://drive.google.com/uc?id=16npJY-gyboRE_HNTQI1dC_fIxh3oxa0S",
}


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(name, destination):
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{name}.pt"
    manifest = destination / f"{name}.json"
    if manifest.exists():
        entry = json.loads(manifest.read_text())
        if entry["source"] != SOURCES[name] or checksum(path) != entry["sha256"]:
            raise ValueError(f"Previously recorded checkpoint changed: {name}")
        return entry
    if path.exists():
        raise ValueError(f"Checkpoint exists without provenance: {path}")
    temporary = path.with_suffix(".download")
    url = SOURCES[name]
    if "drive.google.com" in url:
        import gdown

        if not gdown.download(url, str(temporary), quiet=True):
            raise RuntimeError(f"Checkpoint download failed: {name}")
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "SoccerViz/0.2"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
    if temporary.stat().st_size < 100_000:
        raise ValueError(f"Unexpectedly small checkpoint; retained for inspection: {temporary}")
    with temporary.open("rb") as stream:
        prefix = stream.read(256).lower()
    if b"<html" in prefix or b"<!doctype" in prefix:
        raise ValueError(f"Checkpoint URL returned HTML: {name}")
    entry = {
        "schema": "public-checkpoint/v1",
        "name": name,
        "source": url,
        "sha256": checksum(temporary),
        "bytes": temporary.stat().st_size,
        "verification": "Locally computed hash, not an independently published checksum",
    }
    temporary.rename(path)
    manifest.write_text(json.dumps(entry, indent=2) + "\n")
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", choices=sorted(SOURCES), nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name in args.names:
        print(json.dumps(fetch(name, args.out)), flush=True)


if __name__ == "__main__":
    main()
