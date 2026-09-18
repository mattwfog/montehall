"""A fixed Random Forest baseline with game-disjoint evaluation and probability SHAP."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from soccerviz.core.analysis import FEATURES, TARGET
from soccerviz.core.data import REVISION, write_json


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
    }


def explain(model, background: pd.DataFrame, values: pd.DataFrame) -> tuple[np.ndarray, float]:
    import shap

    explainer = shap.TreeExplainer(
        model, data=background, model_output="probability", feature_perturbation="interventional"
    )
    contribution = explainer.shap_values(values, check_additivity=False)
    contribution = np.asarray(contribution)[:, :, list(model.classes_).index(1)]
    base = float(np.asarray(explainer.expected_value)[list(model.classes_).index(1)])
    probability = model.predict_proba(values)[:, list(model.classes_).index(1)]
    if not np.allclose(base + contribution.sum(axis=1), probability, atol=1e-5):
        raise ValueError("SHAP contributions do not reconstruct the model probability")
    return contribution, base


def train_baseline(data: Path, out: Path, train_game: int = 1, eval_game: int = 2) -> dict:
    if train_game == eval_game:
        raise ValueError("Training and evaluation must be different matches")
    train_path = data / "processed" / f"game_{train_game}" / "samples.parquet"
    eval_path = data / "processed" / f"game_{eval_game}" / "samples.parquet"
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(eval_path)
    if set(train.game) & set(test.game):
        raise ValueError("Game leakage")
    if train[TARGET].nunique() != 2:
        raise ValueError("Training target needs both classes")
    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=6,
        min_samples_leaf=15,
        max_features=0.8,
        n_jobs=4,
        random_state=20260907,
    )
    model.fit(train[FEATURES], train[TARGET])
    background = train[FEATURES].sample(n=min(100, len(train)), random_state=20260907)
    prior = float(train[TARGET].mean())
    probs = model.predict_proba(test[FEATURES])[:, 1]
    model_metrics = metrics(test[TARGET].to_numpy(), probs)
    prior_metrics = metrics(test[TARGET].to_numpy(), np.full(len(test), prior))
    # Cluster-resample entire possessions; multiple samples in one possession are dependent.
    clusters = [
        np.flatnonzero(test.possession_id.to_numpy() == key) for key in test.possession_id.unique()
    ]
    rng = np.random.default_rng(20260907)
    labels = test[TARGET].to_numpy()
    delta = (probs - labels) ** 2 - (prior - labels) ** 2
    draws = [
        float(
            delta[
                np.concatenate(
                    [clusters[i] for i in rng.integers(0, len(clusters), size=len(clusters))]
                )
            ].mean()
        )
        for _ in range(500)
    ]
    confidence = np.quantile(draws, [0.025, 0.975]).tolist()
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "background": background,
            "features": FEATURES,
            "train_game": train_game,
            "eval_game": eval_game,
        },
        out / "baseline.joblib",
    )
    predictions = []
    for table, split in [(train, "training"), (test, "evaluation")]:
        values, base = explain(model, background, table[FEATURES])
        result = table[["sample_id", "game", "possession_id", "index", "time_s", TARGET]].copy()
        result["probability"] = model.predict_proba(table[FEATURES])[:, 1]
        result["base_probability"] = base
        result["split"] = split
        for j, feature in enumerate(FEATURES):
            result[f"shap_{feature}"] = values[:, j]
        predictions.append(result)
    pd.concat(predictions).to_parquet(out / "predictions.parquet", index=False)
    report = {
        "target": TARGET,
        "train_game": train_game,
        "eval_game": eval_game,
        "training_samples": len(train),
        "evaluation_samples": len(test),
        "training_positive_rate": prior,
        "evaluation_positive_rate": float(test[TARGET].mean()),
        "random_forest": model_metrics,
        "training_prior_baseline": prior_metrics,
        "brier_difference_95pct_possession_bootstrap": confidence,
        "beats_prior_brier": bool(model_metrics["brier"] < prior_metrics["brier"]),
        "source_revision": REVISION,
        "seed": 20260907,
        "input_sha256": {
            "train": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "eval": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
        },
        "versions": {p: importlib.metadata.version(p) for p in ["scikit-learn", "shap", "numpy"]},
        "limitations": [
            "One training match and one held-out development match; not elite-level validation",
            "Progression is a spatial surrogate, not possession value or causal tactical credit",
            "Provider events supply possession intervals; not a video-only model",
            "SHAP explains predictions relative to training background; dependent features share credit",
            "Manual reviews are stored separately and never silently enter training/evaluation",
        ],
        "shap": {
            "output": "positive-class probability",
            "background": "100 training rows",
            "additivity_verified": True,
        },
    }
    write_json(out / "baseline-report.json", report)
    return report
