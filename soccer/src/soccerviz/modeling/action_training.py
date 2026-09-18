"""Match-disjoint xT and modest VAEP probability baselines on pinned public data."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from soccerviz.core.assets import sha256
from soccerviz.core.data import write_json
from soccerviz.providers.action_value import run_baseline, vaep_values, validate_actions


def state_tables(
    actions: pd.DataFrame, *, include_labels: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Current completed action features; future outcome labels remain separate."""
    from socceraction.vaep import features, labels

    actions = validate_actions(actions)
    feature_rows, label_rows = [], []
    transformers = [
        features.actiontype_onehot,
        features.result_onehot,
        features.bodypart_onehot,
        features.startlocation,
        features.endlocation,
        features.startpolar,
        features.endpolar,
        features.movement,
    ]
    for _, group in actions.groupby(["game_id", "period_id"], sort=False):
        group = group.reset_index(drop=True)
        feature_rows.append(pd.concat([transform(group) for transform in transformers], axis=1))
        if include_labels:
            label_rows.append(
                pd.concat(
                    [labels.scores(group, nr_actions=10), labels.concedes(group, nr_actions=10)],
                    axis=1,
                )
            )
    x = pd.concat(feature_rows, ignore_index=True).astype(float)
    y = pd.concat(label_rows, ignore_index=True).astype(int) if include_labels else pd.DataFrame()
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("VAEP features must all be finite")
    return x, y


def probability_metrics(y, probabilities) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss

    y, probabilities = np.asarray(y), np.asarray(probabilities, dtype=float)
    if y.shape != probabilities.shape or not np.isfinite(probabilities).all():
        raise ValueError("Probability rows must match finite target rows")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be within [0,1]")
    return {
        "brier": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "positive_count": int(y.sum()),
        "actions": len(y),
        "observed_prevalence": float(y.mean()),
        "mean_predicted_probability": float(probabilities.mean()),
    }


def predict_probabilities(
    model_dir: Path, actions: pd.DataFrame, *, orientation: str
) -> pd.DataFrame:
    """Replay portable fitted coefficients on games untouched by model selection."""
    from scipy.special import expit

    if orientation != "attacking_left_to_right":
        raise ValueError("VAEP inference requires explicit attacking_left_to_right coordinates")
    model_dir = Path(model_dir)
    provenance = json.loads((model_dir / "model-provenance.json").read_text())
    if sha256(model_dir / "vaep-model.npz") != provenance["model_sha256"]:
        raise ValueError("VAEP model content hash mismatch")
    actions = validate_actions(actions)
    excluded = set(map(str, provenance["training_game_ids"] + provenance["development_game_ids"]))
    if set(map(str, actions.game_id.unique())) & excluded:
        raise ValueError("Inference games overlap training or development/model-selection games")
    features, _ = state_tables(actions, include_labels=False)
    if list(features.columns) != provenance["features"]:
        raise ValueError("VAEP feature contract differs from trained model")
    result = actions[["game_id", "action_id"]].copy()
    with np.load(model_dir / "vaep-model.npz", allow_pickle=False) as model:
        for target in ("scores", "concedes"):
            standardized = (features.to_numpy() - model[f"{target}_mean"]) / model[
                f"{target}_scale"
            ]
            result[target] = expit(
                standardized @ model[f"{target}_coef"].T + model[f"{target}_intercept"]
            ).ravel()
    return result


def load_splits(dataset: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    dataset = Path(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest["orientation"] != "attacking_left_to_right":
        raise ValueError("Dataset must explicitly use attacking_left_to_right coordinates")
    if sha256(dataset / "splits.json") != manifest["split_sha256"]:
        raise ValueError("Frozen split manifest hash mismatch")
    frozen = json.loads((dataset / "splits.json").read_text())
    if frozen["splits"] != manifest["splits"]:
        raise ValueError("Dataset manifest differs from predeclared split")
    tables, seen = {}, set()
    for split in ("train", "dev", "test"):
        record = manifest["outputs"][split]
        path = dataset / record["path"]
        if path.resolve().parent != dataset.resolve():
            raise ValueError("Split file must be directly inside the dataset directory")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"{split} action content hash mismatch")
        table = validate_actions(pd.read_parquet(path))
        ids = set(map(int, table.game_id.unique()))
        if ids != set(manifest["splits"][split]) or ids != set(record["game_ids"]):
            raise ValueError("Split action game IDs differ from predeclared manifest")
        if ids & seen:
            raise ValueError("Training, development, and test games must be disjoint")
        if (table.period_id > 4).any():
            raise ValueError("Penalty shootout actions must be excluded")
        seen |= ids
        tables[split] = table
    return manifest, tables


def train_baselines(dataset: Path, out: Path) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    dataset, out = Path(dataset), Path(out)
    if out.exists():
        raise ValueError("Training output already exists; use a new immutable experiment directory")
    manifest, tables = load_splits(dataset)
    out.mkdir(parents=True)
    protocol = {
        "schema_version": 1,
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "splits": manifest["splits"],
        "candidates_C": [0.1, 1.0],
        "selection": "minimum dev log_loss, independently per target; ties use smaller C",
        "feature_scope": "current completed action only; no player/team/game IDs",
        "label_horizon": "current and next nine actions, confined to match and period",
        "test_policy": "one final evaluation after model choice; no refit on dev",
        "baseline": "constant probability equal to train target prevalence",
        "random_state": 0,
    }
    write_json(out / "protocol.json", protocol)
    x_train, y_train = state_tables(tables["train"])
    x_dev, y_dev = state_tables(tables["dev"])
    models, selections, arrays, prevalences = {}, {}, {}, {}
    for target in ("scores", "concedes"):
        if y_train[target].nunique() != 2:
            raise ValueError(f"Training target {target} requires positive and negative examples")
        prevalence = float(y_train[target].mean())
        prevalences[target] = prevalence
        candidates = []
        for c in protocol["candidates_C"]:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c, max_iter=1500, solver="lbfgs", random_state=0),
            )
            model.fit(x_train, y_train[target])
            probabilities = model.predict_proba(x_dev)[:, 1]
            score = probability_metrics(y_dev[target], probabilities)
            candidates.append((score["log_loss"], c, model, score))
        _, c, selected, _ = min(candidates, key=lambda value: (value[0], value[1]))
        scaler, classifier = selected.steps[0][1], selected.steps[1][1]
        if int(classifier.n_iter_.max()) >= classifier.max_iter:
            raise RuntimeError("Selected classifier failed to converge")
        models[target] = selected
        selections[target] = {
            "selected_C": c,
            "dev_candidates": [{"C": item[1], **item[3]} for item in candidates],
            "dev_prevalence_baseline": probability_metrics(
                y_dev[target], np.full(len(y_dev), prevalence)
            ),
        }
        arrays.update(
            {
                f"{target}_coef": classifier.coef_,
                f"{target}_intercept": classifier.intercept_,
                f"{target}_mean": scaler.mean_,
                f"{target}_scale": scaler.scale_,
            }
        )
    np.savez_compressed(out / "vaep-model.npz", **arrays)
    provenance = {
        "model_sha256": sha256(out / "vaep-model.npz"),
        "training_game_ids": manifest["splits"]["train"],
        "development_game_ids": manifest["splits"]["dev"],
        "features": list(x_train.columns),
        "train_prevalence": prevalences,
        "selection": selections,
        "protocol_sha256": sha256(out / "protocol.json"),
    }
    # Freeze selected model before constructing test labels or computing test scores.
    write_json(out / "model-provenance.json", provenance)
    x_test, y_test = state_tables(tables["test"])
    probabilities = tables["test"][["game_id", "action_id"]].copy()
    metrics = {}
    for target, model in models.items():
        p = model.predict_proba(x_test)[:, 1]
        probabilities[target] = p
        fitted = probability_metrics(y_test[target], p)
        baseline = probability_metrics(y_test[target], np.full(len(y_test), prevalences[target]))
        metrics[target] = {
            "model": fitted,
            "train_prevalence_baseline": baseline,
            "brier_skill_vs_prevalence": 1 - fitted["brier"] / baseline["brier"],
        }
    probabilities.to_parquet(out / "test-probabilities.parquet", index=False)
    y_test.to_parquet(out / "test-labels.parquet", index=False)
    values = vaep_values(tables["test"], probabilities, provenance)
    values.to_parquet(out / "test-vaep-values.parquet", index=False)
    xt = run_baseline(
        dataset / manifest["outputs"]["train"]["path"],
        dataset / manifest["outputs"]["test"]["path"],
        out / "xt",
        orientation="attacking_left_to_right",
    )
    report = {
        "schema_version": 1,
        "dataset_manifest_sha256": protocol["dataset_manifest_sha256"],
        "source_dataset": manifest.get("dataset"),
        "source_repository": manifest["repository"],
        "source_commit": manifest["configuration"].get("commit"),
        "source_licence": manifest.get("licence"),
        "game_counts": {key: len(value) for key, value in manifest["splits"].items()},
        "action_counts": {key: len(value) for key, value in tables.items()},
        "splits": manifest["splits"],
        "vaep_model": provenance,
        "test_metrics": metrics,
        "xT": xt,
        "artifacts": {},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("socceraction", "scikit-learn", "numpy", "pandas")
        },
        "limitations": [
            "Preliminary current-action logistic model, not the full three-action VAEP architecture",
            "One tournament with few independent matches; actions within matches are correlated",
            "Match-disjoint evaluation does not imply disjoint teams or chronological deployment",
            "Labels include the current action and truncate the horizon at period boundaries",
            "Brier/log-loss measure outcome probabilities, not causal action value or coaching quality",
            "xT fitting and scoring does not establish xT accuracy",
            "No video tracking, tactical truth, or commercial license is inferred from these results",
        ],
    }
    for path in sorted(out.rglob("*")):
        if path.is_file():
            report["artifacts"][str(path.relative_to(out))] = sha256(path)
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if request.pop("schema_version", 1) != 1:
        raise ValueError("Unsupported request schema_version")
    operation = request.pop("operation", "train")
    if operation == "train":
        result = train_baselines(**request)
    elif operation == "predict":
        output = Path(request["out"])
        if output.exists():
            raise ValueError("Prediction output already exists")
        predictions = predict_probabilities(
            Path(request["model_dir"]),
            pd.read_parquet(request["actions"]),
            orientation=request["orientation"],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(output, index=False)
        result = {
            "schema_version": 1,
            "actions": len(predictions),
            "output_sha256": sha256(output),
            "output": str(output),
        }
    else:
        raise ValueError(f"Unknown action training operation: {operation}")
    write_json(args.response, result)


if __name__ == "__main__":
    main()
