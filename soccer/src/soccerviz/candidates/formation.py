"""Run the real UnravelSports EFPI on complete, observed Metrica team shapes.

Worker: PYTHONPATH=src .venvs/formations/bin/python -m soccerviz.candidates.formation
       --game 1 --out artifacts/integrations/formation-game-1
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import platform
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from soccerviz.core.assets import sha256
from soccerviz.core.data import write_json
from soccerviz.providers.provider import load_metrica


def validate_observations(table: pd.DataFrame) -> None:
    required = {
        "frame_id",
        "timestamp_s",
        "period",
        "provider_entity_id",
        "team_id",
        "entity",
        "x_m",
        "y_m",
        "status",
        "coordinate_system",
    }
    if not required.issubset(table.columns) or table.empty:
        raise ValueError("Missing required nonempty provider observation table")
    if set(table.coordinate_system) != {"metres_top_left_105x68"}:
        raise ValueError("Coordinates must explicitly use metres_top_left_105x68")
    if table.duplicated(["period", "frame_id", "provider_entity_id"]).any():
        raise ValueError("Duplicate entity within a frame, possibly cross-team contamination")
    players = table[table.entity == "player"]
    if set(players.team_id) != {"home", "away"}:
        raise ValueError("Expected explicit home and away team IDs")
    if (players.groupby("provider_entity_id").team_id.nunique() != 1).any():
        raise ValueError("A provider player ID belongs to multiple teams")
    if table.groupby("frame_id").period.nunique().max() > 1:
        raise ValueError("Frame IDs must be unique across periods")
    if table.groupby("frame_id").timestamp_s.nunique().max() > 1:
        raise ValueError("Source clock differs across entities in a frame")
    clocks = table[["frame_id", "timestamp_s"]].drop_duplicates().sort_values("frame_id")
    if not np.isfinite(clocks.timestamp_s).all() or (clocks.timestamp_s.diff().dropna() <= 0).any():
        raise ValueError("Source timestamps must be finite and strictly increasing")


def goalkeeper_proxies(table: pd.DataFrame) -> dict:
    """Fixed per-period proxy: high-coverage player closest to either goal on median.

    Require >=80% coverage, a >=3 m gap to the next player's same-goal distance,
    and opposing inferred defended goals. These are geometry assumptions, not labels.
    """
    result = {}
    for period, period_table in table.groupby("period"):
        teams = {}
        total_frames = period_table.frame_id.nunique()
        for team in ("home", "away"):
            candidates = []
            observed = period_table[
                (period_table.team_id == team) & (period_table.status == "observed")
            ]
            for player, rows in observed.groupby("provider_entity_id"):
                points = rows[["x_m", "y_m"]].to_numpy(float)
                if rows.frame_id.nunique() / total_frames < 0.8 or not np.isfinite(points).all():
                    continue
                for goal_x in (0.0, 105.0):
                    dist = np.linalg.norm(points - [goal_x, 34.0], axis=1)
                    candidates.append((float(np.median(dist)), str(player), goal_x))
            candidates.sort()
            if not candidates:
                raise ValueError(f"No sufficiently observed goalkeeper proxy for {period}/{team}")
            distance, player, goal_x = candidates[0]
            rivals = [d for d, p, g in candidates if p != player and g == goal_x]
            if not rivals or min(rivals) - distance < 3.0:
                raise ValueError(f"Ambiguous goalkeeper/orientation proxy for {period}/{team}")
            teams[team] = {
                "provider_entity_id": player,
                "defended_goal_x_m": goal_x,
                "attacking_direction_source_x": 1 if goal_x == 0 else -1,
                "median_goal_distance_m": distance,
                "next_player_gap_m": min(rivals) - distance,
                "status": "geometry_proxy_unverified",
            }
        if teams["home"]["defended_goal_x_m"] == teams["away"]["defended_goal_x_m"]:
            raise ValueError("Team orientation proxies do not defend opposing goals")
        result[int(period)] = teams
    return result


def eligible_frames(table: pd.DataFrame, proxies: dict, possessions: pd.DataFrame):
    """Audit every sampled frame; never interpolate, merge substitutions or infer possession."""
    rows = []
    for (period, frame_id), frame in table.groupby(["period", "frame_id"], sort=True):
        timestamp = float(frame.timestamp_s.iloc[0])
        row = {"period": int(period), "frame_id": int(frame_id), "timestamp_s": timestamp}
        reasons = []
        for team in ("home", "away"):
            observed = frame[(frame.team_id == team) & (frame.status == "observed")]
            points = observed[["x_m", "y_m"]].to_numpy(float, copy=True)
            keeper = proxies[int(period)][team]["provider_entity_id"]
            complete = len(observed) == 11 and keeper in set(observed.provider_entity_id)
            row[f"{team}_observed_players"] = len(observed)
            row[f"{team}_complete"] = bool(complete)
            if not complete:
                reasons.append(f"{team}_incomplete_11_with_keeper_proxy")
            if not np.isfinite(points).all():
                reasons.append(f"{team}_nonfinite_coordinates")
            elif ((points < [0, 0]) | (points > [105, 68])).any():
                reasons.append(f"{team}_outside_pitch_coordinates")
        ball = frame[(frame.entity == "ball") & (frame.status == "observed")]
        if len(ball) != 1 or not np.isfinite(ball[["x_m", "y_m"]].to_numpy(float)).all():
            reasons.append("ball_unavailable")
        intervals = possessions[
            (possessions.period == period)
            & (possessions.start_s <= timestamp)
            & (possessions.end_s > timestamp)
        ]
        if len(intervals) > 1:
            raise ValueError("Overlapping provider-event possession intervals")
        owner = None
        if len(intervals) == 1:
            owner = {0: "home", 1: "away"}.get(int(intervals.team.iloc[0]))
        if owner is None:
            reasons.append("provider_event_possession_unavailable")
        row.update(
            ball_owning_team_id=owner, eligible=not reasons, omitted_reasons=";".join(reasons)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def to_unravel(table: pd.DataFrame, audit: pd.DataFrame, proxies: dict, game: int):
    """Use public Kloppy and Unravel constructors; assert clocks and coordinates survive."""
    from kloppy import domain as d
    from unravel.soccer import KloppyPolarsDataset

    selected = audit[audit.eligible]
    if selected.empty:
        raise ValueError("No complete, observed frames with provider-event possession")
    # Keeping this experiment within one period avoids ambiguous role metadata at halftime.
    if selected.period.nunique() != 1:
        raise ValueError("Run each period separately; goalkeeper proxies are period-specific")
    period_id = int(selected.period.iloc[0])
    period = d.Period(period_id, timedelta(0), timedelta(seconds=float(selected.timestamp_s.max())))
    teams = {
        name: d.Team(name, name.title(), d.Ground.HOME if name == "home" else d.Ground.AWAY)
        for name in ("home", "away")
    }
    players = {}
    for row in table[table.entity == "player"].drop_duplicates("provider_entity_id").itertuples():
        player = d.Player(
            player_id=row.provider_entity_id,
            team=teams[row.team_id],
            jersey_no=None,
            starting_position=d.PositionType.Goalkeeper
            if row.provider_entity_id == proxies[period_id][row.team_id]["provider_entity_id"]
            else d.PositionType.Unknown,
        )
        teams[row.team_id].players.append(player)
        players[row.provider_entity_id] = player
    home_direction = proxies[period_id]["home"]["attacking_direction_source_x"]
    frames, expected = [], []
    by_frame = table.groupby("frame_id")
    for meta in selected.itertuples():
        frame = by_frame.get_group(meta.frame_id)
        data, ball = {}, None
        for row in frame[frame.status == "observed"].itertuples():
            # Source top-left metres -> centred Cartesian, then static home attacks right.
            x, y = (row.x_m - 52.5) * home_direction, (34.0 - row.y_m) * home_direction
            if row.entity == "player":
                data[players[row.provider_entity_id]] = d.PlayerData(d.Point(x, y))
            else:
                # Metrica CSV has no ball height. EFPI uses x/y only, z=0 is an API placeholder.
                ball = d.Point3D(x, y, 0.0)
            flip = 1 if meta.ball_owning_team_id == "home" else -1
            expected.append(
                {
                    "frame_id": meta.frame_id,
                    "id": row.provider_entity_id,
                    "x": x * flip,
                    "y": y * flip,
                    "timestamp_s": meta.timestamp_s,
                }
            )
        frames.append(
            d.Frame(
                period=period,
                timestamp=timedelta(seconds=meta.timestamp_s),
                statistics=[],
                ball_owning_team=teams[meta.ball_owning_team_id],
                ball_state=None,
                frame_id=meta.frame_id,
                players_data=data,
                other_data={},
                ball_coordinates=ball,
            )
        )
    coordinate_system = d.SecondSpectrumCoordinateSystem(pitch_length=105, pitch_width=68)
    metadata = d.Metadata(
        periods=[period],
        teams=list(teams.values()),
        coordinate_system=coordinate_system,
        pitch_dimensions=coordinate_system.pitch_dimensions,
        orientation=d.Orientation.STATIC_HOME_AWAY,
        flags=d.DatasetFlag.BALL_OWNING_TEAM,
        provider=d.Provider.METRICA,
        frame_rate=1.0,
        game_id=f"metrica-game-{game}",
    )
    converted = KloppyPolarsDataset(
        kloppy_dataset=d.TrackingDataset(records=frames, metadata=metadata),
        add_smoothing=False,
        orient_ball_owning=True,
    )
    actual = converted.data.to_pandas()
    joined = pd.DataFrame(expected).merge(
        actual, on=["frame_id", "id"], suffixes=("_expected", ""), validate="one_to_one"
    )
    if len(joined) != len(expected) or len(actual) != len(expected):
        raise RuntimeError("Unravel conversion dropped or added source observations")
    clock_error = np.max(np.abs(joined.timestamp.dt.total_seconds() - joined.timestamp_s))
    coordinate_error = np.max(
        np.abs(joined[["x", "y"]].to_numpy() - joined[["x_expected", "y_expected"]].to_numpy())
    )
    if coordinate_error > 1e-7 or clock_error > 1e-7:
        raise RuntimeError("Unravel conversion changed audited coordinates or source timestamps")
    return converted, {
        "max_coordinate_error_m": float(coordinate_error),
        "max_timestamp_error_s": float(clock_error),
        "rows_verified": len(joined),
    }


def summarize_shapes(assignments: pd.DataFrame, audit: pd.DataFrame) -> dict:
    """Stability only compares eligible consecutive sampled frames within the same phase."""
    shapes = (
        assignments[assignments.team_id.isin(["home", "away"])]
        .drop_duplicates(["frame_id", "team_id"])
        .merge(audit[["frame_id", "timestamp_s", "ball_owning_team_id"]], on="frame_id")
    )
    result = {}
    for team, rows in shapes.groupby("team_id"):
        rows = rows.sort_values("timestamp_s")
        adjacent = np.isclose(rows.timestamp_s.diff(), 1.0) & (
            rows.ball_owning_team_id == rows.ball_owning_team_id.shift()
        )
        changed = rows.formation != rows.formation.shift()
        counts = rows.formation.value_counts()
        result[str(team)] = {
            "assigned_frames": len(rows),
            "formation_counts": counts.to_dict(),
            "dominant_formation": str(counts.index[0]),
            "dominant_share": float(counts.iloc[0] / len(rows)),
            "comparable_adjacent_pairs_same_phase": int(adjacent.sum()),
            "changed_pairs_same_phase": int((adjacent & changed).sum()),
            "same_phase_adjacent_stability": float((adjacent & ~changed).sum() / adjacent.sum())
            if adjacent.any()
            else None,
        }
    return result


def run(game: int, out: Path, root: Path = Path("data"), duration_s: int = 300) -> dict:
    import polars as pl
    from unravel.soccer import EFPI

    from soccerviz.modeling.tactics import shape_assignment

    if duration_s < 2 or duration_s > 1800:
        raise ValueError("duration_s must be between 2 and 1800")
    if out.exists():
        raise ValueError(f"Output already exists: {out}")
    out.mkdir(parents=True)
    started = time.perf_counter()
    paths = {
        name: root
        / "raw"
        / f"game_{game}"
        / f"Sample_Game_{game}_RawTrackingData_{name.title()}_Team.csv"
        for name in ("home", "away")
    }
    table, source_report = load_metrica(
        paths["home"], paths["away"], sample_rate=0.04, limit=duration_s + 1
    )
    table = table[(table.period == 1) & (table.timestamp_s < duration_s)].copy()
    validate_observations(table)
    possessions_path = root / "processed" / f"game_{game}" / "possessions.parquet"
    possessions = pd.read_parquet(possessions_path)
    proxies = goalkeeper_proxies(table)
    audit = eligible_frames(table, proxies, possessions)
    table.to_parquet(out / "source-observations.parquet", index=False)
    audit.to_parquet(out / "frame-eligibility.parquet", index=False)
    dataset, conversion_report = to_unravel(table, audit, proxies, game)
    dataset.data.write_parquet(out / "efpi-input.parquet")
    model = EFPI(dataset=dataset).fit(
        every="frame", change_threshold=None, change_after_possession=False
    )
    assignments = model.output.to_pandas()
    # Keep only stable source identity/clock fields needed for joining with observations.
    assignments = assignments.drop(columns=["ball_owning_team_id"], errors="ignore")
    assignments.to_parquet(out / "frame-assignments.parquet", index=False)
    frames = summarize_shapes(assignments, audit)
    intervals = EFPI(dataset=dataset).fit(
        every="1m", change_threshold=None, change_after_possession=False, substitutions="drop"
    )
    intervals.output.write_parquet(out / "minute-assignments.parquet")
    intervals.segments.write_parquet(out / "minute-segments.parquet")
    baseline = []
    for frame_id in audit.loc[audit.eligible, "frame_id"]:
        frame = table[table.frame_id == frame_id]
        for team in ("home", "away"):
            observed = frame[(frame.team_id == team) & (frame.status == "observed")]
            points = observed[["x_m", "y_m"]].to_numpy(float, copy=True)
            if proxies[1][team]["attacking_direction_source_x"] == -1:
                points[:, 0] = 105 - points[:, 0]
            result = shape_assignment(points)
            baseline.append(
                {
                    "frame_id": int(frame_id),
                    "team_id": team,
                    "formation": result["shape"],
                    "template_error": result["template_error"],
                }
            )
    baseline = pd.DataFrame(baseline)
    baseline.to_parquet(out / "baseline-assignments.parquet", index=False)
    compared = baseline.merge(
        assignments[assignments.team_id.isin(["home", "away"])].drop_duplicates(
            ["frame_id", "team_id"]
        )[["frame_id", "team_id", "formation"]],
        on=["frame_id", "team_id"],
        suffixes=("_baseline", "_efpi"),
    )
    agreement = compared.formation_baseline.str.replace("-", "") == compared.formation_efpi
    package_names = (
        "unravelsports",
        "kloppy",
        "mplsoccer",
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "pyarrow",
        "scikit-learn",
    )
    report = {
        "schema_version": 1,
        "status": "executed",
        "method": "UnravelSports EFPI public API",
        "game": game,
        "duration_s_requested": duration_s,
        "sample_frequency_hz": 1,
        "sampled_frames": len(audit),
        "both_teams_complete_frames": int((audit.home_complete & audit.away_complete).sum()),
        "home_complete_frames": int(audit.home_complete.sum()),
        "away_complete_frames": int(audit.away_complete.sum()),
        "eligible_frames": int(audit.eligible.sum()),
        "omitted_frames": int((~audit.eligible).sum()),
        "omitted_reason_counts_nonexclusive": audit.loc[~audit.eligible, "omitted_reasons"]
        .str.split(";")
        .explode()
        .value_counts()
        .to_dict(),
        "source": source_report,
        "goalkeeper_and_orientation_proxies": proxies,
        "conversion_verification": conversion_report,
        "frame_efpi": frames,
        "baseline_three_template": summarize_shapes(baseline, audit),
        "baseline_name_agreement": float(agreement.mean()),
        "comparison_note": "Name agreement is not accuracy; template sets, scaling and GK rules differ",
        "minute_formations": intervals.output.filter(pl.col("team_id") != "ball")
        .select(["1m_id", "team_id", "formation", "is_attacking"])
        .unique()
        .sort(["1m_id", "team_id", "is_attacking"])
        .to_pandas()
        .assign(**{"1m_id": lambda f: f["1m_id"].dt.total_seconds()})
        .to_dict("records"),
        "packages": {name: importlib.metadata.version(name) for name in package_names},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "efpi_implementation_sha256": sha256(Path(inspect.getfile(EFPI))),
        "adapter_sha256": sha256(Path(__file__)),
        "provider_adapter_sha256": sha256(Path(inspect.getfile(load_metrica))),
        "baseline_implementation_sha256": sha256(Path(inspect.getfile(shape_assignment))),
        "possession_parquet_sha256": sha256(possessions_path),
        "events_sha256": sha256(
            root / "raw" / f"game_{game}" / f"Sample_Game_{game}_RawEventsData.csv"
        ),
        "elapsed_s": time.perf_counter() - started,
        "limitations": [
            "Descriptive template assignments only; no analyst formation/role labels or accuracy claim",
            "GK and attacking direction are fixed segment geometry proxies, not provider role truth",
            "105x68 pitch dimensions are the existing project assumption",
            "Possession is existing retrospective provider-event segmentation; gaps are omitted",
            "Only complete observed 11-player teams, finite on-pitch player coordinates and observed ball",
            "No interpolation or inferred off-screen players; only first five minutes at 1 Hz by default",
            "Ball z=0 only satisfies Kloppy API; Metrica CSV provides no observed height; EFPI uses x/y",
            "Minute fits average observed positions separately by owning team; not contiguous possessions",
            "Baseline chooses deepest player independently per frame; EFPI uses fixed GK proxy",
        ],
        "primary_sources": [
            "https://github.com/UnravelSports/unravelsports",
            "https://github.com/metrica-sports/sample-data",
            "https://arxiv.org/abs/2506.23843",
        ],
    }
    write_json(out / "report.json", report)
    write_json(
        out / "artifact-sha256.json",
        {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=int, choices=[1, 2], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--duration-s", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(run(args.game, args.out, args.root, args.duration_s), indent=2, default=str))


if __name__ == "__main__":
    main()
