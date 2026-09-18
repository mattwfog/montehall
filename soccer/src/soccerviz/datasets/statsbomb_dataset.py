"""Pinned public StatsBomb corpus with deterministic match-disjoint partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from soccerviz.core.assets import sha256
from soccerviz.core.data import write_json

DEFAULT_COMMIT = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
REPOSITORY = "https://github.com/hudl/open-data"


def split_matches(matches: list[dict], max_matches: int = 30) -> dict[str, list[int]]:
    """Select by fixed identifier hash; split before opening any event outcomes."""
    if max_matches < 10:
        raise ValueError("At least ten matches are required for train/dev/test")
    ids = [int(row["match_id"]) for row in matches]
    if len(ids) != len(set(ids)) or len(ids) < max_matches:
        raise ValueError("Match catalogue needs enough unique match IDs")
    ids.sort(key=lambda game: hashlib.sha256(f"soccerviz-v1:{game}".encode()).hexdigest())
    selected = ids[:max_matches]
    heldout = max(2, max_matches // 6)
    return {
        "train": selected[: max_matches - 2 * heldout],
        "dev": selected[max_matches - 2 * heldout : -heldout],
        "test": selected[-heldout:],
    }


def _fetch(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with urlopen(url, timeout=60) as response:
                data = response.read()
            if destination.suffix == ".json":
                json.loads(data)
            temporary = destination.with_suffix(destination.suffix + ".download")
            temporary.write_bytes(data)
            temporary.replace(destination)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def build_dataset(
    out: Path,
    *,
    commit: str = DEFAULT_COMMIT,
    competition_id: int = 55,
    season_id: int = 43,
    max_matches: int = 30,
) -> dict:
    import pandas as pd

    from soccerviz.providers.action_value import statsbomb_to_spadl

    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Source must be pinned to a full Git commit")
    if not all(isinstance(value, int) and value > 0 for value in (competition_id, season_id)):
        raise ValueError("Competition and season IDs must be positive integers")
    out = Path(out)
    configuration = {
        "commit": commit,
        "competition_id": competition_id,
        "season_id": season_id,
        "max_matches": max_matches,
    }
    config_path = out / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Existing corpus configuration differs; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    write_json(config_path, configuration)
    base = f"https://raw.githubusercontent.com/hudl/open-data/{commit}"
    catalogue_name = f"data/matches/{competition_id}/{season_id}.json"
    _fetch(f"{base}/{catalogue_name}", out / catalogue_name)
    matches = json.loads((out / catalogue_name).read_text())
    splits = split_matches(matches, max_matches)
    # This durable split is written before the event files are fetched or converted.
    split_record = {
        "schema_version": 1,
        "algorithm": "sha256(soccerviz-v1:match_id)",
        "configuration": configuration,
        "splits": splits,
    }
    split_path = out / "splits.json"
    if split_path.exists() and json.loads(split_path.read_text()) != split_record:
        raise ValueError("Frozen split manifest changed")
    write_json(split_path, split_record)
    names = [catalogue_name, "LICENSE.pdf", "README.md"]
    for game in [game for values in splits.values() for game in values]:
        names.extend([f"data/events/{game}.json", f"data/lineups/{game}.json"])
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda name: _fetch(f"{base}/{name}", out / name), names))
    sources = {
        name: {
            "url": f"{base}/{name}",
            "sha256": sha256(out / name),
            "bytes": (out / name).stat().st_size,
        }
        for name in names
    }
    prior = out / "manifest.json"
    if prior.exists() and json.loads(prior.read_text())["sources"] != sources:
        raise ValueError("Downloaded source changed from the recorded content hashes")
    homes = {row["match_id"]: row["home_team"]["home_team_id"] for row in matches}
    outputs = {}
    for split, ids in splits.items():
        frames = [
            statsbomb_to_spadl(
                out / f"data/events/{game}.json",
                out / f"data/lineups/{game}.json",
                game,
                homes[game],
            )
            for game in ids
        ]
        table = pd.concat(frames, ignore_index=True)
        # Penalty shootouts are not open-play action value training examples.
        table = table[table.period_id <= 4].reset_index(drop=True)
        path = out / f"{split}.parquet"
        table.to_parquet(path, index=False)
        outputs[split] = {
            "path": path.name,
            "sha256": sha256(path),
            "actions": len(table),
            "game_ids": ids,
        }
    report = {
        "schema_version": 1,
        "dataset": "StatsBomb Open Data",
        "repository": REPOSITORY,
        "configuration": configuration,
        "split_sha256": sha256(split_path),
        "splits": splits,
        "sources": sources,
        "outputs": outputs,
        "orientation": "attacking_left_to_right",
        "excluded_periods": [5],
        "terms_file": "LICENSE.pdf",
        "attribution": "Data: StatsBomb",
        "limitations": [
            "One competition-season; teams recur across match-disjoint splits",
            "Not a tracking or coaching-quality ground truth",
        ],
    }
    write_json(prior, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if request.pop("schema_version", 1) != 1:
        raise ValueError("Unsupported request schema_version")
    write_json(args.response, build_dataset(**request))


if __name__ == "__main__":
    main()
