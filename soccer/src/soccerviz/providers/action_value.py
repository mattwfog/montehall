"""Isolated socceraction SPADL, xT, and VAEP adapters with explicit training lineage."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from soccerviz.core.assets import sha256
from soccerviz.core.data import write_json


def validate_actions(actions: pd.DataFrame) -> pd.DataFrame:
    try:
        from socceraction import spadl
    except ImportError as exc:
        raise RuntimeError(
            "Use the isolated socceraction runtime in docs/guides/data-tracking-adapters.md"
        ) from exc
    actions = actions.copy()
    # Add canonical labels through the library rather than maintaining a copy.
    actions = spadl.add_names(
        actions.drop(
            columns=[c for c in ["type_name", "result_name", "bodypart_name"] if c in actions]
        )
    )
    actions = spadl.SPADLSchema.validate(actions)
    if actions.empty or actions[["game_id", "action_id"]].duplicated().any():
        raise ValueError("SPADL must contain unique (game_id, action_id) rows")
    if actions[["game_id", "team_id", "player_id"]].isna().any().any():
        raise ValueError("SPADL game/team/player IDs must be known")
    for _, game in actions.groupby("game_id", sort=False):
        expected = game.sort_values(["period_id", "time_seconds", "action_id"], kind="stable")
        if not game.index.equals(expected.index):
            raise ValueError("SPADL actions must be ordered by period and source time")
    return actions.reset_index(drop=True)


def statsbomb_to_spadl(
    events: Path, lineup: Path, game_id: int | str, home_team_id: int | str
) -> pd.DataFrame:
    """Use socceraction's supported local StatsBomb loader and SPADL converter.

    The direct loader avoids a verified incompatibility between socceraction
    1.5.3's Kloppy bridge and Kloppy 3.19 coordinate-system construction.
    """
    from socceraction.data.statsbomb import StatsBombLoader
    from socceraction.spadl import play_left_to_right, statsbomb

    if isinstance(game_id, bool) or not str(game_id).isdigit():
        raise ValueError("StatsBomb game_id must be an integer")
    game_id = int(game_id)
    teams = {team["team_id"] for team in json.loads(Path(lineup).read_text())}
    if home_team_id not in teams:
        raise ValueError("home_team_id absent from lineup")
    with tempfile.TemporaryDirectory(prefix="soccerviz-statsbomb-") as folder:
        root = Path(folder)
        (root / "events").mkdir()
        (root / "events" / f"{game_id}.json").symlink_to(Path(events).resolve())
        loaded = StatsBombLoader(getter="local", root=str(root)).events(game_id)
    converted = statsbomb.convert_to_actions(loaded, home_team_id=home_team_id)
    if home_team_id not in converted.team_id.unique():
        raise ValueError("home_team_id absent from converted actions")
    return validate_actions(play_left_to_right(converted, home_team_id))


def vaep_training_tables(actions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Library feature/label generation, confined to each match and period.

    Labels use future outcomes and must be kept out of inference inputs. This
    function does not fit a classifier or invent scoring probabilities.
    """
    from socceraction.vaep import features, labels

    actions = validate_actions(actions)
    feature_rows, label_rows = [], []
    for _, group in actions.groupby(["game_id", "period_id"], sort=False):
        group = group.reset_index(drop=True)
        keys = group[["game_id", "action_id"]]
        feature_rows.append(
            pd.concat([keys, features.actiontype(group), features.startlocation(group)], axis=1)
        )
        label_rows.append(pd.concat([keys, labels.scores(group), labels.concedes(group)], axis=1))
    return pd.concat(feature_rows, ignore_index=True), pd.concat(label_rows, ignore_index=True)


def vaep_values(
    actions: pd.DataFrame, probabilities: pd.DataFrame, model_provenance: dict
) -> pd.DataFrame:
    """Apply socceraction's VAEP formula to externally trained probabilities."""
    from socceraction.vaep import formula

    actions = validate_actions(actions)
    if not model_provenance.get("model_sha256") or not model_provenance.get("training_game_ids"):
        raise ValueError("VAEP requires a model hash and training_game_ids")
    if set(map(str, actions.game_id)) & set(map(str, model_provenance["training_game_ids"])):
        raise ValueError("VAEP inference games overlap model training games")
    keys = ["game_id", "action_id"]
    if probabilities[keys].duplicated().any():
        raise ValueError("Duplicate probability action keys")
    joined = actions.merge(
        probabilities[keys + ["scores", "concedes"]],
        on=keys,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    values = joined[["scores", "concedes"]].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Every action needs finite scoring/conceding probabilities in [0,1]")
    rows = []
    for _, group in joined.groupby(["game_id", "period_id"], sort=False):
        group = group.reset_index(drop=True)
        rows.append(
            pd.concat([group[keys], formula.value(group, group.scores, group.concedes)], axis=1)
        )
    return pd.concat(rows, ignore_index=True)


def run_baseline(
    train: Path,
    evaluate: Path,
    out: Path,
    *,
    orientation: str,
    data_kind: str = "observed",
    grid: tuple[int, int] = (16, 12),
    probabilities: Path | None = None,
    vaep_provenance: dict | None = None,
) -> dict:
    """Fit xT on training matches, rate separate matches, and prepare VAEP tables."""
    from socceraction.xthreat import ExpectedThreat

    if orientation != "attacking_left_to_right":
        raise ValueError("xT requires explicit attacking_left_to_right SPADL coordinates")
    if data_kind not in {"observed", "synthetic"}:
        raise ValueError("data_kind must be observed or synthetic")
    if len(grid) != 2 or any(not isinstance(x, int) or not 1 <= x <= 100 for x in grid):
        raise ValueError("grid must contain two positive integers <=100")
    train, evaluate, out = Path(train), Path(evaluate), Path(out)
    if out.exists():
        raise ValueError(f"Output already exists: {out}")
    training = validate_actions(pd.read_parquet(train))
    testing = validate_actions(pd.read_parquet(evaluate))
    if set(map(str, training.game_id)) & set(map(str, testing.game_id)):
        raise ValueError("Training and evaluation games must be disjoint")
    if not training.type_name.eq("shot").any():
        raise ValueError("xT training needs recorded shots and outcomes")
    model = ExpectedThreat(l=grid[0], w=grid[1]).fit(training)
    rated = testing.copy()
    rated["xT"] = model.rate(testing)
    features, labels = vaep_training_tables(training)
    vaep = None
    if probabilities is not None:
        vaep = vaep_values(testing, pd.read_parquet(probabilities), vaep_provenance or {})
    report = {
        "schema_version": 1,
        "socceraction_version": importlib.metadata.version("socceraction"),
        "data_kind": data_kind,
        "orientation": orientation,
        "source_sha256": {"train": sha256(train), "evaluate": sha256(evaluate)},
        "training_game_ids": sorted(map(str, training.game_id.unique())),
        "evaluation_game_ids": sorted(map(str, testing.game_id.unique())),
        "training_actions": len(training),
        "evaluation_actions": len(testing),
        "rated_successful_moves": int(rated.xT.notna().sum()),
        "xT_grid": list(grid),
        "vaep_status": "external_model_probabilities" if vaep is not None else "untrained",
        "real_world_value_accuracy": None,
        "limitations": [
            "xT estimates action value; fit/rate success is not an accuracy measurement",
            "xT leaves non-successful-movement actions unrated according to upstream semantics",
            "VAEP features and future-outcome labels are separate artifacts; no classifier trained here",
            "Metrica CSV events are not a supported socceraction Kloppy conversion provider; no guessed mapping",
        ],
    }
    out.mkdir(parents=True)
    rated.to_parquet(out / "action-values.parquet", index=False)
    features.to_parquet(out / "vaep-training-features.parquet", index=False)
    labels.to_parquet(out / "vaep-training-labels.parquet", index=False)
    np.savez_compressed(out / "xt-model.npz", xT=model.xT)
    report["model_sha256"] = sha256(out / "xt-model.npz")
    if vaep is not None:
        vaep.to_parquet(out / "vaep-values.parquet", index=False)
        report["vaep_model_provenance"] = vaep_provenance
        report["vaep_probability_sha256"] = sha256(probabilities)
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
    if request.get("operation") == "statsbomb-to-spadl":
        table = statsbomb_to_spadl(
            Path(request["events"]),
            Path(request["lineup"]),
            request["game_id"],
            request["home_team_id"],
        )
        out = Path(request["out"])
        if out.exists():
            raise ValueError(f"Output already exists: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(out, index=False)
        report = {
            "actions": len(table),
            "output_sha256": sha256(out),
            "orientation": "attacking_left_to_right",
            "source_sha256": {
                "events": sha256(Path(request["events"])),
                "lineup": sha256(Path(request["lineup"])),
            },
        }
    else:
        report = run_baseline(
            Path(request["train"]),
            Path(request["evaluate"]),
            Path(request["out"]),
            orientation=request["orientation"],
            data_kind=request.get("data_kind", "observed"),
            grid=tuple(request.get("grid", [16, 12])),
            probabilities=Path(request["probabilities"]) if request.get("probabilities") else None,
            vaep_provenance=request.get("vaep_provenance"),
        )
    write_json(args.response, report)


if __name__ == "__main__":
    main()
