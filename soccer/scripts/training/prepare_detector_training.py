"""Freeze and export a bounded public TRAIN-only detector pilot; never run training."""

import argparse
import concurrent.futures
import hashlib
import json
import re
import zipfile
from pathlib import Path

from soccerviz.candidates.detector_training import (
    DATASET_REVISION,
    MAX_DOWNLOAD_BYTES,
    export_corpus,
    export_roboflow_corpus,
    freeze_protocol,
    freeze_roboflow_protocol,
    validate_partition,
    write_new_json,
)
from soccerviz.datasets.soccernet_adapter import (
    RangeReader,
    download_sequence,
    read_json_url,
    sha256,
)


def discover(root, external_manifest):
    """Read only JSON info prefixes for rejected games; fetch full selected labels."""
    repo = "SoccerNet/SN-GSR-2024"
    external = json.loads(external_manifest.read_text())
    excluded = {str(s["game_id"]) for s in external["sources"]}
    inventory = read_json_url(f"https://huggingface.co/api/datasets/{repo}/tree/{DATASET_REVISION}")
    entry = next(row for row in inventory if row["path"] == "train.zip")
    url = f"https://huggingface.co/datasets/{repo}/resolve/{DATASET_REVISION}/train.zip"
    reader = RangeReader(url, entry["size"], 40_000_000)
    selected, census, games = [], [], set()
    with zipfile.ZipFile(reader) as archive:
        for member in sorted(n for n in archive.namelist() if n.endswith("/Labels-GameState.json")):
            with archive.open(member) as source:
                prefix = source.read(65536).decode("utf-8")
            match = re.search(r'"info"\s*:\s*', prefix)
            if match is None:
                raise ValueError("Source info metadata was not found in bounded JSON prefix")
            info, _ = json.JSONDecoder().raw_decode(prefix, match.end())
            game = str(info["game_id"])
            row = {
                "sequence": info["name"],
                "game_id": game,
                "member": member,
                "selected": game not in excluded and game not in games,
            }
            census.append(row.copy())
            if not row["selected"]:
                continue
            data = archive.read(member)
            row["sha256"] = hashlib.sha256(data).hexdigest()
            path = root / "sources" / "train" / row["sequence"] / "Labels-GameState.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
            selected.append(row)
            games.add(game)
            print(f"Selected official TRAIN {row['sequence']} / game {game}", flush=True)
            if len(selected) == 3:
                break
    if len(selected) != 3:
        raise ValueError(
            "Could not find three eligible distinct training games in bounded discovery"
        )
    result = {
        "repository": repo,
        "revision": DATASET_REVISION,
        "archive": entry,
        "archive_url": url,
        "transferred_bytes": reader.transferred,
        "prior_probe_transfer_upper_bound_bytes": 0,
        "census": census,
        "selected": selected,
        "selection_note": "Sorted TRAIN metadata-prefix scan, first distinct games excluding "
        "the external development games; no model outputs used.",
    }
    write_new_json(root / "selection-inventory.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/detector-training"))
    parser.add_argument(
        "--external-manifest", type=Path, default=Path("artifacts/vision-benchmark/manifest.json")
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--roboflow-source",
        type=Path,
        help="A downloaded Roboflow Universe COCO export directory (with source.json); "
        "builds a video-disjoint corpus from it instead of SoccerNet",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "manifest.json").exists():
        raise FileExistsError("Completed frozen training corpus already exists")
    if args.roboflow_source:
        root.mkdir(parents=True, exist_ok=True)
        if not (root / "protocol.json").exists():
            freeze_roboflow_protocol(root, args.roboflow_source, args.external_manifest)
        result = export_roboflow_corpus(root)
        print(
            json.dumps({k: result[k] for k in ("frame_counts", "video_counts", "census")}, indent=2)
        )
        return
    inventory_path = root / "selection-inventory.json"
    inventory = (
        json.loads(inventory_path.read_text())
        if inventory_path.exists()
        else discover(root, args.external_manifest)
    )
    protocol_path = root / "protocol.json"
    protocol = (
        json.loads(protocol_path.read_text())
        if protocol_path.exists()
        else freeze_protocol(root, inventory, args.external_manifest)
    )
    validate_partition(protocol)
    if sha256(args.external_manifest) != protocol["external_development_manifest_sha256"]:
        raise ValueError("External development manifest changed since exclusion games were frozen")
    if args.download and not (root / "download-summary.json").exists():
        prior = (
            inventory.get("prior_probe_transfer_upper_bound_bytes", 0)
            + inventory["transferred_bytes"]
        )
        per_sequence_budget = (MAX_DOWNLOAD_BYTES - prior) // 3
        if per_sequence_budget <= 0:
            raise ValueError("No corpus download budget remains")

        def fetch(sequence):
            name = sequence["sequence"]
            output = root / "sources"
            cached = output / f"gsr-train-{name}-download.json"
            if cached.exists():
                result = json.loads(cached.read_text())
                for row in result["files"]:
                    if sha256(row["path"]) != row["sha256"]:
                        raise ValueError("Cached source file checksum mismatch")
            else:
                result = download_sequence(
                    "gsr",
                    "train",
                    output,
                    name,
                    image_count=75,
                    image_stride=10,
                    max_bytes=per_sequence_budget,
                )
            if result["revision"] != DATASET_REVISION or result["split"] != "train":
                raise ValueError("Downloader source differs from frozen official TRAIN revision")
            print(
                f"Verified {name}: {result['transferred_bytes']} source-transfer bytes", flush=True
            )
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(fetch, protocol["sequences"]))
        transferred = sum(row["transferred_bytes"] for row in results)
        if prior + transferred > MAX_DOWNLOAD_BYTES:
            raise ValueError("Corpus exceeded the total250MB download bound")
        files = [
            {**row, "relative_path": str(Path(row["path"]).relative_to(root))}
            for result in results
            for row in result["files"]
        ]
        write_new_json(
            root / "download-summary.json",
            {
                "schema": "detector-training-downloads/v1",
                "source_split": "train",
                "files": files,
                "image_and_selected_annotation_transfer_bytes": transferred,
                "discovery_and_prior_probe_accounted_bytes": prior,
                "total_accounted_transfer_upper_bound_bytes": prior + transferred,
                "max_download_bytes": MAX_DOWNLOAD_BYTES,
                "archive_hash_verified": False,
                "verification": "Official pinned archive membership, ZIP CRC32 and local SHA256; "
                "entire archive was not downloaded",
            },
        )
    result = export_corpus(root)
    print(
        json.dumps(
            {key: result[key] for key in ("frame_counts", "census", "masked_frames")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
