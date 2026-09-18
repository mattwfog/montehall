"""Pinned Wyscout open match-event corpus (Pappalardo et al. 2019), CC BY 4.0.

figshare collection 4415000: seven competitions of 2017/18 (Serie A, Premier
League, La Liga, Ligue 1, Bundesliga) plus Euro 2016 and World Cup 2018. Every
article's licence was read as CC BY 4.0 on 2026-09-08; the files are pinned by
figshare file id and the MD5 figshare publishes for each one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

from soccerviz.core.assets import sha256
from soccerviz.core.data import write_json

COLLECTION = "https://doi.org/10.6084/m9.figshare.c.4415000"
LICENCE = "CC BY 4.0"
CITATION = (
    "Pappalardo, L., Cintia, P., Rossi, A. et al. A public data set of spatio-temporal "
    "match events in soccer competitions. Sci Data 6, 236 (2019)."
)
# name -> (figshare file id, published MD5, bytes)
FILES = {
    "events.zip": (14464685, "7c20e8647e7eda58d7838a0c7b1ec6ab", 77323413),
    "matches.zip": (14464622, "51d80beb17480919f69a53a0152c2d71", 645097),
    "players.json": (15073721, "f28ddf6326281efeda6488b2169f5609", 1737347),
    "teams.json": (15073697, "1381ff9449f21105090729cf0e086b5b", 27404),
    "competitions.json": (15073685, "3dc210a4805dda5337b0ff9f7eaa407a", 1209),
    "eventid2name.csv": (21385245, "46daf16100ece0c743eedc9adcfea162", 1001),
    "tags2name.csv": (21385239, "e7acb14918d00e40c80a898b1da8fc39", 1754),
    "referees.json": (15074030, "6b5b5612e128238f0f28a355cc13796a", 196709),
    "coaches.json": (15073868, "ee8afad14b5be622b1d3560d15248778", 70664),
}
DEFAULT_OUT = Path("artifacts/public-data/wyscout")


def download_url(file_id: int) -> str:
    return f"https://ndownloader.figshare.com/files/{file_id}"


def _fetch(url: str, destination: Path, expected_md5: str) -> None:
    """Resumable: an existing file with the published MD5 is never refetched."""
    if destination.exists() and md5(destination) == expected_md5:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with urlopen(url, timeout=120) as response:
                data = response.read()
            if hashlib.md5(data).hexdigest() != expected_md5:
                raise ValueError(f"{destination.name}: MD5 differs from the figshare record")
            temporary = destination.with_suffix(destination.suffix + ".download")
            temporary.write_bytes(data)
            temporary.replace(destination)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(archive: Path, folder: Path) -> list[str]:
    if not folder.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(folder)
    return sorted(p.name for p in folder.iterdir())


def build_dataset(out: Path = DEFAULT_OUT) -> dict:
    """Download every pinned file, verify MD5, extract the zips, record provenance."""
    records = {}
    for name, (file_id, expected_md5, size) in FILES.items():
        destination = out / name
        _fetch(download_url(file_id), destination, expected_md5)
        record = {
            "figshare_file_id": file_id,
            "url": download_url(file_id),
            "md5": md5(destination),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "expected_bytes": size,
        }
        if name.endswith(".zip"):
            record["extracted"] = extract(destination, out / name[:-4])
        records[name] = record
        print(f"{name}: {record['bytes']} bytes, sha256 {record['sha256'][:12]}", flush=True)
    summary = {
        "schema": "wyscout-open-data/v1",
        "collection": COLLECTION,
        "licence": LICENCE,
        "citation": CITATION,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": records,
    }
    write_json(out / "source.json", summary)
    return summary


LOADER_FILES = (
    ("competitions.json", "competitions.json"),
    ("players.json", "players.json"),
    ("teams.json", "teams.json"),
) + tuple(
    (f"{kind}/{kind}_{name}.json", f"{kind}_{name}.json")
    for kind in ("events", "matches")
    for name in (
        "England",
        "European_Championship",
        "France",
        "Germany",
        "Italy",
        "Spain",
        "World_Cup",
    )
)


def loader_root(source: Path) -> Path:
    """socceraction's public loader wants one flat folder; link the extracted files into it."""
    root = source / "loader-root"
    root.mkdir(exist_ok=True)
    for relative, name in LOADER_FILES:
        link, target = root / name, (source / relative).resolve()
        if not target.exists():
            raise FileNotFoundError(f"{target}: run build_dataset first")
        if not link.exists():
            link.symlink_to(target)
    return root


def wyscout_to_spadl(loader, game_id: int, home_team_id: int):
    """socceraction's own Wyscout converter, then the shared orientation and validation."""
    from socceraction.spadl import play_left_to_right, wyscout

    from soccerviz.providers.action_value import validate_actions

    converted = wyscout.convert_to_actions(loader.events(game_id), home_team_id=home_team_id)
    if home_team_id not in converted.team_id.unique():
        raise ValueError(f"home_team_id {home_team_id} absent from converted actions of {game_id}")
    return validate_actions(play_left_to_right(converted, home_team_id))


def build_spadl_dataset(out: Path, *, source: Path = DEFAULT_OUT, max_matches: int = 300) -> dict:
    """Match-disjoint train/dev/test SPADL tables from all seven competitions, same
    contract as the StatsBomb corpus so modeling.action_training consumes it unchanged."""
    import pandas as pd
    from socceraction.data.wyscout import PublicWyscoutLoader

    from soccerviz.datasets.statsbomb_dataset import split_matches

    source, out = Path(source), Path(out)
    provenance = json.loads((source / "source.json").read_text())
    loader = PublicWyscoutLoader(root=str(loader_root(source)), download=False)
    competitions = loader.competitions()
    games = pd.concat(
        [loader.games(row.competition_id, row.season_id) for row in competitions.itertuples()],
        ignore_index=True,
    )
    configuration = {
        "source": provenance["collection"],
        "source_files_sha256": {name: rec["sha256"] for name, rec in provenance["files"].items()},
        "competitions": sorted(int(c) for c in games.competition_id.unique()),
        "available_games": len(games),
        "max_matches": max_matches,
    }
    config_path = out / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("Existing corpus configuration differs; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    write_json(config_path, configuration)
    splits = split_matches([{"match_id": int(g)} for g in games.game_id], max_matches)
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
    homes = dict(zip(games.game_id.astype(int), games.home_team_id.astype(int)))
    outputs = {}
    for split, ids in splits.items():
        path = out / f"{split}.parquet"
        if not path.exists():
            frames = []
            for index, game in enumerate(ids, 1):
                frames.append(wyscout_to_spadl(loader, game, homes[game]))
                if index % 25 == 0:
                    print(f"{split}: {index}/{len(ids)} games converted", flush=True)
            table = pd.concat(frames, ignore_index=True)
            table = table[table.period_id <= 4].reset_index(drop=True)
            table.to_parquet(path, index=False)
        table = pd.read_parquet(path)
        outputs[split] = {
            "path": path.name,
            "sha256": sha256(path),
            "actions": len(table),
            "game_ids": ids,
        }
    report = {
        "schema_version": 1,
        "dataset": "Wyscout open match-event dataset (Pappalardo et al. 2019)",
        "repository": COLLECTION,
        "licence": LICENCE,
        "citation": CITATION,
        "configuration": configuration,
        "split_sha256": sha256(split_path),
        "splits": splits,
        "sources": provenance["files"],
        "outputs": outputs,
        "orientation": "attacking_left_to_right",
        "excluded_periods": [5],
        "terms_file": "figshare article licence: CC BY 4.0",
        "attribution": "Data: Wyscout via Pappalardo et al. 2019, CC BY 4.0",
        "limitations": [
            "Seven competitions of one season (2017/18 leagues, Euro 2016, World Cup 2018); teams recur across match-disjoint splits",
            "Wyscout event coordinates are percentages of the pitch; socceraction's converter scales them",
            "Not a tracking or coaching-quality ground truth",
        ],
    }
    write_json(out / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Raw download folder")
    parser.add_argument("--spadl-out", type=Path, help="Also build the SPADL corpus here")
    parser.add_argument("--max-matches", type=int, default=300)
    args = parser.parse_args()
    summary = build_dataset(args.out)
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2))
    if args.spadl_out:
        report = build_spadl_dataset(args.spadl_out, source=args.out, max_matches=args.max_matches)
        print(json.dumps({k: report[k] for k in ("splits", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
