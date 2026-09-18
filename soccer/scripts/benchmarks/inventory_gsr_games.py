"""Inventory the official SoccerNet GSR split: every sequence's game id and clip metadata.

Reads only the head of each sequence's Labels-GameState.json (the `info` block sits
first in the file) through the bounded HTTP-range ZIP reader, so a whole split costs a
few megabytes, not the 11 GB archive. Results persist per sequence as they arrive and
a re-run skips what is already recorded. Needed to choose a game-disjoint evaluation:
the frozen development corpus uses games 2, 3 and 5.

usage: PYTHONPATH=src python scripts/benchmarks/inventory_gsr_games.py \
           artifacts/calibration-benchmark/gsr-valid-inventory.json [--split valid]
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

from soccerviz.datasets.soccernet_adapter import REPOSITORIES, RangeReader, read_json_url

HEAD_BYTES = 2048  # the info block is well inside the first 2 KB of the pretty-printed label file


def info_block(head):
    """Parse the `info` object out of a truncated JSON head."""
    start = head.index('"info"')
    brace = head.index("{", start)
    depth = 0
    for i in range(brace, len(head)):
        if head[i] == "{":
            depth += 1
        elif head[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(head[brace : i + 1])
    raise ValueError("info block not closed inside the fetched head")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--max-bytes", type=int, default=50_000_000)
    args = parser.parse_args()
    repo = REPOSITORIES["gsr"]
    metadata = read_json_url(f"https://huggingface.co/api/datasets/{repo}")
    revision = metadata["sha"]
    inventory = read_json_url(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}")
    entry = next(item for item in inventory if item["path"] == f"{args.split}.zip")
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{args.split}.zip"
    done = {}
    if args.out.exists():
        previous = json.loads(args.out.read_text())
        if previous["revision"] != revision:
            raise SystemExit("upstream revision changed; refusing to mix inventories")
        done = {row["sequence"]: row for row in previous["sequences"]}
    reader = RangeReader(url, entry["size"], args.max_bytes)
    with zipfile.ZipFile(reader) as archive:
        names = archive.namelist()
        prefix = args.split + "/" if any(n.startswith(args.split + "/") for n in names) else ""
        sequences = sorted(
            {
                n.removeprefix(prefix).split("/")[0]
                for n in names
                if re.match(rf"{prefix}SNGS-\d+/", n)
            }
        )
        image_counts = {
            s: sum(1 for n in names if n.startswith(f"{prefix}{s}/img1/") and n.endswith(".jpg"))
            for s in sequences
        }
        for sequence in sequences:
            if sequence in done:
                continue
            member = f"{prefix}{sequence}/Labels-GameState.json"
            with archive.open(member) as stream:
                head = stream.read(HEAD_BYTES).decode("utf-8", errors="replace")
            info = info_block(head)
            done[sequence] = {
                "sequence": sequence,
                "game_id": info.get("game_id"),
                "action_class": info.get("action_class"),
                "game_time_start": info.get("game_time_start"),
                "seq_length": int(info.get("seq_length", 0)),
                "images_in_archive": image_counts[sequence],
                "label_bytes": archive.getinfo(member).file_size,
            }
            args.out.write_text(
                json.dumps(
                    {
                        "schema": "gsr-split-inventory/v1",
                        "repository": repo,
                        "split": args.split,
                        "revision": revision,
                        "archive_bytes": entry["size"],
                        "transferred_bytes": reader.transferred,
                        "sequences": [done[s] for s in sorted(done)],
                    },
                    indent=1,
                )
            )
            print(
                sequence,
                "game",
                done[sequence]["game_id"],
                "transferred",
                reader.transferred,
                flush=True,
            )
    print("sequences", len(done), "games", len({r["game_id"] for r in done.values()}))


if __name__ == "__main__":
    main()
