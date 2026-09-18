"""Public Roboflow soccer demo assets; optional download, source hashes recorded."""

import hashlib
import json
from pathlib import Path

SOURCES = {
    "football-ball-detection.pt": "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V",
    "football-player-detection.pt": "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q",
    "football-pitch-detection.pt": "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf",
    "2e57b9_0.mp4": "19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf",
}
EXPECTED_SHA256 = {
    "football-ball-detection.pt": "678fbad05134f19c5094cb8d273812ec9c6691228180d46832551ecf99ed2912",
    "football-player-detection.pt": "75b09c377fbf9d0791d23f6cfb689f5aed6eaa43a6818bd1fb884cf7507fffaf",
    "football-pitch-detection.pt": "28f68f7c4056d6d9b137efd2e7ab5f3c494039380c63831649126ced25628b36",
    "2e57b9_0.mp4": "7788719b32b944e42d8f1024fb4802f681bc2053d6301218a5bff6705b549f96",
}
UPSTREAM = "https://github.com/roboflow/sports/blob/42c80c06b6b65a7f89455b89fe31cdf4c38ba227/examples/soccer/setup.sh"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_demo(root: Path):
    import gdown

    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    prior = json.loads(manifest.read_text())["files"] if manifest.exists() else {}
    files = {}
    for name, source_id in SOURCES.items():
        path = root / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".download")
            result = gdown.download(id=source_id, output=str(temporary), quiet=False)
            if result is None:
                raise RuntimeError(f"Could not download public asset {name}")
            temporary.replace(path)
        digest = sha256(path)
        if digest != EXPECTED_SHA256[name]:
            raise ValueError(f"Pinned demo asset hash mismatch: {name}")
        if name in prior and digest != prior[name]["sha256"]:
            raise ValueError(f"Previously downloaded asset changed: {name}")
        files[name] = {
            "url": f"https://drive.google.com/uc?id={source_id}",
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    result = {
        "upstream_setup": UPSTREAM,
        "files": files,
        "purpose": "Local evaluation of the upstream public demo assets",
        "terms": "Upstream code MIT; video/model terms are separate. No redistribution license asserted.",
    }
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/demo"))
    print(json.dumps(fetch_demo(parser.parse_args().out), indent=2))
