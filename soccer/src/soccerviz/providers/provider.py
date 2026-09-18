"""Kloppy Metrica ingestion retaining source clocks and missing observations."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from soccerviz.core.assets import sha256
from soccerviz.core.data import write_json


def load_metrica(
    home: Path, away: Path, *, sample_rate: float = 0.2, limit: int | None = None
) -> tuple[pd.DataFrame, dict]:
    """Return long-form provider observations and source provenance.

    `sample_rate` is a fraction of source frames, not Hz. Kloppy's internal CSV
    clock uses frame_id / 25; we join its parsed identities/positions back to the
    actual CSV timestamps. Coordinates are returned in source top-left metres.
    """
    if not 0 < sample_rate <= 1 or (limit is not None and limit < 1):
        raise ValueError("sample_rate must be in (0,1]; limit must be positive")
    try:
        from kloppy import metrica
    except ImportError as exc:
        raise RuntimeError("Install kloppy>=3.19,<4 to import provider tracking") from exc
    home, away = Path(home), Path(away)
    clocks = [
        pd.read_csv(
            path,
            skiprows=3,
            header=None,
            usecols=[0, 1, 2],
            names=["period", "frame_id", "timestamp_s"],
        )
        for path in [home, away]
    ]
    if not clocks[0].equals(clocks[1]):
        raise ValueError("Home/away source clocks disagree")
    clock = clocks[0]
    if (
        clock.frame_id.duplicated().any()
        or not np.isfinite(clock.timestamp_s).all()
        or not (np.diff(clock.timestamp_s) > 0).all()
    ):
        raise ValueError("Source frame IDs must be unique and timestamps strictly increasing")
    clock = clock.set_index("frame_id")
    # Kloppy 3.19's limit counts sampled iterations against limit/sample_rate and
    # stops one early. Load enough, then apply our unambiguous output-frame limit.
    loader_limit = None if limit is None else max(2, int(np.ceil((limit + 2) * sample_rate)))
    dataset = metrica.load_tracking_csv(
        home_data=home,
        away_data=away,
        sample_rate=sample_rate,
        limit=loader_limit,
        coordinates="metrica",
    )
    rows = []
    for frame in dataset.records[:limit]:
        if frame.frame_id not in clock.index:
            raise ValueError("Kloppy returned a frame absent from the source clock")
        source = clock.loc[frame.frame_id]
        if int(source.period) != frame.period.id:
            raise ValueError("Kloppy period disagrees with source")
        common = {
            "frame_id": frame.frame_id,
            "timestamp_s": float(source.timestamp_s),
            "period": frame.period.id,
            "source": "provider_tracking",
            "coordinate_system": "metres_top_left_105x68",
            "kloppy_timestamp_s": frame.timestamp.total_seconds(),
        }
        for team in dataset.metadata.teams:
            for player in team.players:
                data = frame.players_data.get(player)
                point = data.coordinates if data else None
                observed = point is not None and np.isfinite([point.x, point.y]).all()
                rows.append(
                    {
                        **common,
                        "provider_entity_id": player.player_id,
                        "team_id": team.team_id,
                        "entity": "player",
                        "x_m": point.x * 105 if observed else None,
                        "y_m": (1 - point.y) * 68 if observed else None,
                        "status": "observed" if observed else "unavailable",
                    }
                )
        point = frame.ball_coordinates
        observed = point is not None and np.isfinite([point.x, point.y]).all()
        rows.append(
            {
                **common,
                "provider_entity_id": "ball",
                "team_id": None,
                "entity": "ball",
                "x_m": point.x * 105 if observed else None,
                "y_m": (1 - point.y) * 68 if observed else None,
                "status": "observed" if observed else "unavailable",
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No tracking frames parsed")
    report = {
        "schema_version": 1,
        "provider": "Metrica CSV",
        "loader": "kloppy",
        "kloppy_version": importlib.metadata.version("kloppy"),
        "source_sha256": {"home": sha256(home), "away": sha256(away)},
        "sample_rate_fraction": sample_rate,
        "frames": int(table.frame_id.nunique()),
        "observations": int((table.status == "observed").sum()),
        "missing_observations": int((table.status == "unavailable").sum()),
        "registered_players": sum(len(t.players) for t in dataset.metadata.teams),
        "coordinate_system": "metres, source top-left origin; x=105, y=68",
        "timestamp_policy": "Exact CSV Time [s] joined by frame ID; Kloppy clock retained separately",
        "limitations": [
            "Provider identities are provider IDs, not verified player names",
            "Pitch dimensions assume the existing SoccerViz 105x68 convention",
            "No interpolation of absent players or ball",
        ],
    }
    return table, report


def export_metrica(
    home: Path, away: Path, out: Path, *, sample_rate: float = 0.2, limit: int | None = None
) -> dict:
    out = Path(out)
    if out.exists():
        raise ValueError(f"Output already exists: {out}")
    table, report = load_metrica(home, away, sample_rate=sample_rate, limit=limit)
    out.mkdir(parents=True)
    table.to_parquet(out / "observations.parquet", index=False)
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if request.get("schema_version", 1) != 1:
        raise ValueError("Unsupported worker request schema_version")
    report = export_metrica(
        Path(request["home"]),
        Path(request["away"]),
        Path(request["out"]),
        sample_rate=request.get("sample_rate", 0.2),
        limit=request.get("limit"),
    )
    write_json(args.response, report)


if __name__ == "__main__":
    main()
