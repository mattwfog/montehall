"""Bounded public SoccerNet archive downloads and provenance-preserving import.

Labels are benchmark targets, never model input. HTTP ranges avoid downloading
multi-gigabyte images archives merely to obtain one sequence's annotations.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import io
import json
import math
import re
import urllib.request
import zipfile
from pathlib import Path

REPOSITORIES = {"gsr": "SoccerNet/SN-GSR-2024", "tracking": "SoccerNet/SN-Tracking-2023"}
SPLITS = {
    "gsr": {"train", "valid", "test", "challenge"},
    "tracking": {"train", "test", "challenge"},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json_url(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


class RangeReader(io.RawIOBase):
    """Seekable read-only HTTP object with strict response and transfer limits."""

    def __init__(self, url, size, budget=100_000_000):
        self.url, self.size, self.budget = url, size, budget
        self.position, self.transferred = 0, 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = (
            offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        )
        if self.position < 0:
            raise ValueError("Negative archive offset")
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size)
        if size <= 0:
            return b""
        if self.transferred + size > self.budget:
            raise ValueError("Archive transfer exceeds max_bytes budget")
        start, end = self.position, self.position + size - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=90) as response:
            if (
                response.status != 206
                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}"
            ):
                raise ValueError(
                    "Server did not honor exact HTTP range; refusing full archive download"
                )
            body = response.read(size + 1)
        if len(body) != size:
            raise ValueError("Truncated or oversized HTTP range")
        self.position += size
        self.transferred += size
        return body


def download_sequence(
    kind, split, out, sequence=None, image_count=0, image_stride=1, max_bytes=100_000_000
):
    """Fetch official members, verify ZIP CRC, record member SHA256 and pinned origin."""
    if kind not in REPOSITORIES or split not in SPLITS[kind]:
        raise ValueError("Unsupported SoccerNet task/split")
    if image_count < 0 or image_stride < 1 or max_bytes < 1:
        raise ValueError("Invalid download bounds")
    out = Path(out)
    repo = REPOSITORIES[kind]
    metadata = read_json_url(f"https://huggingface.co/api/datasets/{repo}")
    revision = metadata["sha"]
    inventory = read_json_url(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}")
    archive_name = (
        "challenge2023.zip" if kind == "tracking" and split == "challenge" else f"{split}.zip"
    )
    entry = next(item for item in inventory if item["path"] == archive_name)
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{archive_name}"
    reader = RangeReader(url, entry["size"], max_bytes)
    files = []
    with zipfile.ZipFile(reader) as archive:
        names = archive.namelist()
        prefix = split + "/" if any(name.startswith(split + "/") for name in names) else ""
        relative = [name.removeprefix(prefix) for name in names]
        candidates = sorted(
            {name.split("/")[0] for name in relative if re.match(r"(?:SNGS|SNMOT)-\d+/", name)}
        )
        if not candidates:
            raise ValueError("No recognized SoccerNet sequences in archive")
        sequence = sequence or candidates[0]
        if sequence not in candidates:
            raise ValueError("Sequence is not a member of the requested official split")
        members = [
            name
            for name in names
            if name.startswith(prefix + sequence + "/")
            and name.endswith(
                ("Labels-GameState.json", "gt/gt.txt", "det/det.txt", "seqinfo.ini", "gameinfo.ini")
            )
        ]
        pictures = sorted(
            name
            for name in names
            if name.startswith(prefix + sequence + "/img1/")
            and name.lower().endswith((".jpg", ".png"))
        )
        members += pictures[::image_stride][:image_count]
        for name in members:
            info = archive.getinfo(name)
            if info.file_size > 100_000_000:
                raise ValueError("Individual archive member exceeds 100MB uncompressed limit")
            destination = out / split / name.removeprefix(prefix)
            if not destination.resolve().is_relative_to(out.resolve()):
                raise ValueError("Unsafe archive member path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
            files.append(
                {
                    "path": str(destination.resolve()),
                    "member": name,
                    "bytes": info.file_size,
                    "sha256": sha256(destination),
                    "zip_crc32": f"{info.CRC:08x}",
                }
            )
    manifest = {
        "schema": "soccernet-download/v1",
        "kind": kind,
        "split": split,
        "sequence": sequence,
        "repository": repo,
        "revision": revision,
        "archive_url": url,
        "archive_bytes": entry["size"],
        "archive_sha256_expected": entry.get("lfs", {}).get("oid"),
        "archive_hash_verified": False,
        "verification": "Member ZIP CRC32 and local SHA256; entire remote archive was not downloaded.",
        "transferred_bytes": reader.transferred,
        "files": files,
        "labels_are_model_input": False,
    }
    write_json(out / f"{kind}-{split}-{sequence}-download.json", manifest)
    return manifest


def validate_version(labels):
    version = str(labels.get("info", {}).get("version", "0"))
    if tuple(int(v) for v in version.split(".")) < (1, 3):
        raise ValueError("SoccerNet GSR annotation version >=1.3 is required")
    return version


def import_gsr(labels, split, sequence=None, fps=None):
    if split not in SPLITS["gsr"]:
        raise ValueError("Invalid official split or frame rate")
    path = Path(labels)
    source = json.loads(path.read_text())
    version = validate_version(source)
    source_fps = float(source["info"]["frame_rate"])
    if fps is not None and float(fps) != source_fps:
        raise ValueError("Frame rate must match source annotation metadata")
    fps = source_fps
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid source frame rate")
    if sequence and source["info"].get("name", sequence) != sequence:
        raise ValueError("Sequence name differs from source annotations")
    if path.parent.parent.name in SPLITS["gsr"] and path.parent.parent.name != split:
        raise ValueError("Requested split differs from source path")
    frames = []
    for image in source["images"]:
        frame = int(Path(image["file_name"]).stem)
        frames.append(
            {
                "image_id": image["image_id"],
                "frame_id": frame,
                "timestamp_s": (frame - 1) / fps,
                "width": image.get("width"),
                "height": image.get("height"),
                "scored": all(
                    image.get(k, False)
                    for k in ("has_labeled_pitch", "has_labeled_camera", "has_labeled_person")
                ),
            }
        )
    known = {f["image_id"] for f in frames}
    rows = []
    for annotation in source["annotations"]:
        if annotation.get("supercategory") != "object":
            continue
        if annotation["image_id"] not in known:
            raise ValueError("Annotation image missing from source frames")
        rows.append(
            {
                k: annotation.get(k)
                for k in (
                    "id",
                    "image_id",
                    "track_id",
                    "category_id",
                    "bbox_image",
                    "bbox_pitch",
                    "attributes",
                )
            }
        )
    return {
        "schema": "soccernet-benchmark/v1",
        "kind": "gsr",
        "split": split,
        "sequence": sequence or path.parent.name,
        "source": str(path.resolve()),
        "source_sha256": sha256(path),
        "annotation_version": version,
        "source_info": source["info"],
        "fps": fps,
        "clock": "Source image frame number is 1-based; timestamp_s=(frame_id-1)/fps. Clip-relative, not match time.",
        "coordinates": "Source pixel xywh and original SoccerNet bbox_pitch meters preserved without rescaling or flipping.",
        "frames": sorted(frames, key=lambda f: f["frame_id"]),
        "ground_truth": rows,
        "labels_are_model_input": False,
    }


def read_mot(path, ground_truth=False):
    rows, seen = [], set()
    with Path(path).open() as source:
        for fields in csv.reader(source):
            if not fields:
                continue
            if len(fields) < 7:
                raise ValueError("MOT rows require frame,id,x,y,w,h,confidence")
            frame, track = int(fields[0]), int(fields[1])
            x, y, width, height, confidence = map(float, fields[2:7])
            if (
                frame < 1
                or width <= 0
                or height <= 0
                or not all(math.isfinite(v) for v in (x, y, width, height, confidence))
            ):
                raise ValueError("Invalid MOT frame or bounding box")
            if ground_truth and confidence != 1:
                continue
            if ground_truth and (frame, track) in seen:
                raise ValueError("Duplicate ground-truth identity in frame")
            seen.add((frame, track))
            rows.append(
                {
                    "frame_id": frame,
                    "track_id": track,
                    "bbox": [x, y, x + width, y + height],
                    "confidence": confidence,
                }
            )
    return rows


def import_tracking(sequence_dir, split):
    if split not in SPLITS["tracking"]:
        raise ValueError("Unsupported official tracking split")
    path = Path(sequence_dir)
    if path.parent.name in SPLITS["tracking"] and path.parent.name != split:
        raise ValueError("Requested split differs from source path")
    config = configparser.ConfigParser()
    config.read(path / "seqinfo.ini")
    seq = config["Sequence"]
    fps, count = float(seq["frameRate"]), int(seq["seqLength"])
    gt = path / "gt/gt.txt"
    rows = read_mot(gt, True) if gt.exists() else []
    if any(row["frame_id"] > count for row in rows):
        raise ValueError("MOT annotations outside sequence length")
    sources = [p for p in (path / "seqinfo.ini", gt, path / "det/det.txt") if p.exists()]
    return {
        "schema": "soccernet-benchmark/v1",
        "kind": "tracking",
        "split": split,
        "sequence": path.name,
        "source": str(path.resolve()),
        "sources": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in sources],
        "fps": fps,
        "width": int(seq["imWidth"]),
        "height": int(seq["imHeight"]),
        "clock": "MOT source frame is 1-based; timestamp_s=(frame_id-1)/fps; clip-relative.",
        "frames": [{"frame_id": f, "timestamp_s": (f - 1) / fps} for f in range(1, count + 1)],
        "ground_truth": rows,
        "labels_available": gt.exists(),
        "labels_are_model_input": False,
    }


def run(request):
    if request.get("schema_version", 1) != 1:
        raise ValueError("Unsupported request schema_version")
    operation = request["operation"]
    if operation == "download":
        return download_sequence(
            **{k: v for k, v in request.items() if k not in {"operation", "schema_version"}}
        )
    if operation == "import-gsr":
        return import_gsr(
            **{k: v for k, v in request.items() if k not in {"operation", "schema_version"}}
        )
    if operation == "import-tracking":
        return import_tracking(
            **{k: v for k, v in request.items() if k not in {"operation", "schema_version"}}
        )
    raise ValueError(f"Unknown SoccerNet operation: {operation}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args()
    write_json(args.response, run(json.loads(args.request.read_text())))


if __name__ == "__main__":
    main()
