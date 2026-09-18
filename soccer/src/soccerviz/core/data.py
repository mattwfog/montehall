"""Pinned Metrica CSV ingestion. Provider coordinates remain observations, never guesses."""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REVISION = "e706dd506b360d69d9d123d5b8026e7294b13996"
BASE_URL = f"https://raw.githubusercontent.com/metrica-sports/sample-data/{REVISION}/data"
LENGTH, WIDTH = 105.0, 68.0
SAMPLE_HZ = 5
SOURCE_FILES = {
    1: {
        "RawEventsData.csv": "3c426dd4f442dd433bccf89e3634f91aa310cd8f",
        "RawTrackingData_Home_Team.csv": "ce2e91c0e633426cec1146f4e728b72417454e3d",
        "RawTrackingData_Away_Team.csv": "ff409a5680fd75acc7cebcf786827214c687478b",
    },
    2: {
        "RawEventsData.csv": "decccd47c1d53960d290cb6679337eec83a8a97f",
        "RawTrackingData_Home_Team.csv": "8b4d3648888cc8b033e7142919ebaea04f6328e0",
        "RawTrackingData_Away_Team.csv": "e7334d8d762954a215c49e55f211855000243fdc",
    },
}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def git_blob_sha(path: Path) -> str:
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(root: Path, games: list[int]) -> None:
    """Download only verified files; restart incomplete files and reuse hash-verified files."""
    for game in games:
        if game not in SOURCE_FILES:
            raise ValueError("Only CSV games 1 and 2 are enabled. Game 3 remains reserved.")
        target = root / "raw" / f"game_{game}"
        target.mkdir(parents=True, exist_ok=True)
        entries = []
        for suffix, expected in SOURCE_FILES[game].items():
            name = f"Sample_Game_{game}_{suffix}"
            path = target / name
            url = f"{BASE_URL}/Sample_Game_{game}/{name}"
            if not path.exists() or git_blob_sha(path) != expected:
                tmp = path.with_suffix(".download")
                print(f"Downloading {name}", flush=True)
                with urllib.request.urlopen(url, timeout=90) as response, tmp.open("wb") as out:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        out.write(chunk)
                if git_blob_sha(tmp) != expected:
                    raise ValueError(f"Source hash mismatch: {name}")
                tmp.replace(path)
            entries.append(
                {"file": name, "url": url, "git_blob_sha1": expected, "bytes": path.stat().st_size}
            )
        write_json(
            target / "manifest.json",
            {
                "source": "Metrica Sports sample data",
                "revision": REVISION,
                "game": game,
                "files": entries,
            },
        )


@dataclass
class Match:
    game: int
    frames: np.ndarray
    times: np.ndarray
    periods: np.ndarray
    xy: np.ndarray  # [time, registered player, xy], NaN for absent players
    ball: np.ndarray  # [time, xy], NaN where source has no position
    teams: np.ndarray  # 0 home, 1 away
    players: list[str]
    direction: np.ndarray  # [time, team] attacking +x or -x
    events: pd.DataFrame


def read_tracking(path: Path, team: int) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    with path.open() as source:
        reader = csv.reader(source)
        next(reader)
        next(reader)
        header = next(reader)
    names = [
        f"{name}_{'x' if (i - 3) % 2 == 0 else 'y'}" if name else f"c{i}"
        for i, name in enumerate(header)
    ]
    table = pd.read_csv(path, skiprows=3, header=None, names=names)
    # Sample by source timestamp buckets, never synthesize timestamps from ordinals.
    bucket = np.floor(table.iloc[:, 2].to_numpy() * SAMPLE_HZ + 1e-7).astype(int)
    table = table.loc[~pd.Series(bucket).duplicated().to_numpy()].reset_index(drop=True)
    indices = [i for i in range(3, len(header), 2) if header[i].startswith("Player")]
    players = [f"{'Home' if team == 0 else 'Away'}:{header[i]}" for i in indices]
    xy = np.stack([table.iloc[:, [i, i + 1]].to_numpy(float) for i in indices], axis=1)
    return table, players, xy * [LENGTH, WIDTH]


def prepare(root: Path, game: int) -> dict:
    source = root / "raw" / f"game_{game}"
    for suffix, expected in SOURCE_FILES[game].items():
        path = source / f"Sample_Game_{game}_{suffix}"
        if not path.exists() or git_blob_sha(path) != expected:
            raise ValueError(f"Missing or modified source {path}; run fetch first")
    h, hp, hx = read_tracking(source / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv", 0)
    a, ap, ax = read_tracking(source / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv", 1)
    if not np.array_equal(h.iloc[:, :3].to_numpy(), a.iloc[:, :3].to_numpy()):
        raise ValueError("Home/Away tracking clocks disagree")
    ball = h.iloc[:, -2:].to_numpy(float) * [LENGTH, WIDTH]
    away_ball = a.iloc[:, -2:].to_numpy(float) * [LENGTH, WIDTH]
    if not np.allclose(ball, away_ball, equal_nan=True, atol=0.05):
        raise ValueError("Home/Away ball observations disagree")
    frames = h.iloc[:, 1].to_numpy(np.int64)
    times = h.iloc[:, 2].to_numpy(float)
    periods = h.iloc[:, 0].to_numpy(np.int8)
    if not np.all(np.diff(times) > 0):
        raise ValueError("Expected strictly increasing source timestamps")
    xy = np.concatenate([hx, ax], axis=1)
    teams = np.array([0] * len(hp) + [1] * len(ap), dtype=np.int8)
    events = pd.read_csv(source / f"Sample_Game_{game}_RawEventsData.csv")
    events = events.sort_values(["Start Frame"], kind="stable").reset_index(drop=True)
    directions = np.zeros((len(times), 2), dtype=np.int8)
    evidence = {}
    for period in np.unique(periods):
        # Initial team centers establish orientation, avoiding future event/shot labels.
        indices = np.flatnonzero(periods == period)[: 5 * SAMPLE_HZ]
        centers = [float(np.nanmedian(xy[indices][:, teams == team, 0])) for team in (0, 1)]
        if abs(centers[0] - centers[1]) < 1:
            raise ValueError("Cannot establish direction from opening team positions")
        home_direction = 1 if centers[0] < centers[1] else -1
        directions[periods == period] = [home_direction, -home_direction]
        evidence[str(int(period))] = {
            "home_attacks": home_direction,
            "opening_median_x_m": centers,
            "method": "first 5 seconds team positions; verify in replay",
        }
    out = root / "processed" / f"game_{game}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "tracking.npz",
        frames=frames,
        times=times,
        periods=periods,
        xy=xy.astype(np.float32),
        ball=ball.astype(np.float32),
        teams=teams,
        players=hp + ap,
        direction=directions,
    )
    events.to_parquet(out / "events.parquet", index=False)
    census = {
        "game": game,
        "revision": REVISION,
        "sample_hz": SAMPLE_HZ,
        "sampled_frames": len(times),
        "events": len(events),
        "registered_players": len(hp + ap),
        "observed_player_positions": int(np.isfinite(xy).all(axis=2).sum()),
        "ball_coverage": float(np.isfinite(ball).all(axis=1).mean()),
        "periods": [int(p) for p in np.unique(periods)],
        "direction_evidence": evidence,
        "coordinate_system": "metres, source top-left origin; x=105, y=68",
        "observation_source": "provider tracking, not SoccerViz CV",
    }
    write_json(out / "census.json", census)
    return census


def load_match(root: Path, game: int) -> Match:
    folder = root / "processed" / f"game_{game}"
    with np.load(folder / "tracking.npz", allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    arrays["players"] = arrays["players"].tolist()
    return Match(game=game, events=pd.read_parquet(folder / "events.parquet"), **arrays)
