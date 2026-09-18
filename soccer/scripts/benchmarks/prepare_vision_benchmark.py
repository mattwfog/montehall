"""Download a frozen public GSR subset and prepare its inference manifest.

Run from the repository with PYTHONPATH=src. Fetches only the sequences and frames the
reviewed protocol.json names, from the official split each sequence belongs to; never
the challenge split, and never for training.
"""

import argparse
import concurrent.futures
import json
from pathlib import Path

from soccerviz.candidates.vision_benchmark import prepare_manifest, sequence_splits
from soccerviz.datasets.soccernet_adapter import (
    download_sequence,
    read_json_url,
    sha256,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/vision-benchmark"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    if (root / "manifest.json").exists():
        raise ValueError("Frozen manifest already exists; do not overwrite an evaluated corpus")
    if protocol["purpose"] != "development":
        raise ValueError("Only frozen development corpora are supported")
    splits = sequence_splits(protocol)
    if protocol["first_frame_id"] != 1:
        raise ValueError("Existing public downloader starts at source frame one")
    if args.download:
        metadata = read_json_url(f"https://huggingface.co/api/datasets/{protocol['dataset']}")
        if metadata["sha"] != protocol["revision"]:
            raise ValueError("Upstream revision changed; refusing to change the frozen corpus")
        sequence_budget = protocol["max_download_bytes"] // len(protocol["sequences"])

        def fetch(sequence):
            download_path = root / "sources" / f"gsr-{splits[sequence]}-{sequence}-download.json"
            if download_path.exists():
                result = json.loads(download_path.read_text())
                if result["revision"] != protocol["revision"]:
                    raise ValueError("Cached dataset revision mismatch")
                for member in result["files"]:
                    if sha256(member["path"]) != member["sha256"]:
                        raise ValueError("Cached source member checksum mismatch")
            else:
                print(f"Downloading {sequence}", flush=True)
                result = download_sequence(
                    "gsr",
                    splits[sequence],
                    root / "sources",
                    sequence,
                    image_count=protocol["frames_per_sequence"],
                    image_stride=protocol["frame_stride"],
                    max_bytes=sequence_budget,
                )
            if result["revision"] != protocol["revision"]:
                raise ValueError("Downloaded dataset revision mismatch")
            print(f"Verified {sequence}: {result['transferred_bytes']} bytes", flush=True)
            return {
                "sequence": sequence,
                "download_manifest": str(download_path),
                "transferred_bytes": result["transferred_bytes"],
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            outcomes = list(pool.map(fetch, protocol["sequences"]))
        total = sum(row["transferred_bytes"] for row in outcomes)
        if total > protocol["max_download_bytes"]:
            raise ValueError("Total source transfer exceeded the frozen budget")
        write_json(
            root / "download-summary.json",
            {"downloads": outcomes, "total_transferred_bytes": total},
        )
    print(json.dumps(prepare_manifest(root), indent=2))


if __name__ == "__main__":
    main()
