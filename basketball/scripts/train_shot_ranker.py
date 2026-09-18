"""Train the learned shot-attribution ranker (structural fix B, 2026-08-25).

Logistic regression over the geometry features extract_rank_features.py
emits, weak-labeled by the PBP shooter jersey. Plain numpy gradient
descent — no sklearn dependency in cvbench. Split is BY GAME (hash), so
validation anchors come from unseen games. The metric that matters is
per-anchor top-1: rank each anchor's candidates by model score, count the
anchors whose top-scored candidate is a positive.

The output JSON matches shot_attribution._learned_score's contract exactly:
{weights, bias, mean, std, impute} over the FEATURE_ORDER names, fit on
RAW feature values standardized by mean/std — no transform on either side.

Usage (inside cvbench or anywhere with numpy):
    python scripts/train_shot_ranker.py \
        --features-dir /work/models/ranker-v1/features \
        --out /work/models/ranker-v1/ranker.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FEATURE_ORDER = ("release_prox_px", "whole_prox_px", "n_obs", "max_h_px")
IMPUTE = {"release_prox_px": 3000.0, "whole_prox_px": 3000.0,
          "n_obs": 0.0, "max_h_px": 0.0}
VALID_FRAC = 0.2
EPOCHS = 400
LR = 0.05
L2 = 1e-3


def _load(features_dir: Path):
    games = sorted(features_dir.glob("*.json"))
    rows, anchor_keys, game_of = [], [], []
    for gp in games:
        doc = json.loads(gp.read_text())
        for r in doc["rows"]:
            rows.append(r)
            anchor_keys.append((doc["game"], r["anchor"]))
            game_of.append(doc["game"])
    return rows, anchor_keys, game_of, [json.loads(g.read_text())
                                        for g in games]


def _matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(rows), len(FEATURE_ORDER)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, name in enumerate(FEATURE_ORDER):
            v = r.get(name)
            x[i, j] = IMPUTE[name] if v is None else float(v)
        y[i] = 1.0 if r["label"] else 0.0
    return x, y


def _top1(scores: np.ndarray, y: np.ndarray, anchors: list) -> float:
    by_anchor: dict = {}
    for i, key in enumerate(anchors):
        cur = by_anchor.get(key)
        if cur is None or scores[i] > cur[0]:
            by_anchor[key] = (scores[i], y[i])
    hits = sum(1 for _s, label in by_anchor.values() if label > 0.5)
    return hits / max(len(by_anchor), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows, anchor_keys, game_of, docs = _load(args.features_dir)
    if not rows:
        raise SystemExit("no feature files")
    x, y = _matrix(rows)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    xs = (x - mean) / std

    def is_valid(game: str) -> bool:
        h = int(hashlib.sha1(game.encode()).hexdigest(), 16)
        return (h % 100) < VALID_FRAC * 100

    valid = np.array([is_valid(g) for g in game_of])
    train = ~valid

    w = np.zeros(xs.shape[1])
    b = 0.0
    pos_w = float((y[train] == 0).sum() / max((y[train] == 1).sum(), 1))
    for _ in range(EPOCHS):
        z = xs[train] @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        sample_w = np.where(y[train] > 0.5, pos_w, 1.0)
        g = (p - y[train]) * sample_w
        gw = xs[train].T @ g / train.sum() + L2 * w
        gb = float(g.mean())
        w -= LR * gw
        b -= LR * gb

    scores = xs @ w + b
    baseline = -x[:, 0]  # release-proximity heuristic (imputed, never inf)
    report = {
        "games": len(docs),
        "rows": len(rows),
        "anchors": len(set(anchor_keys)),
        "anchors_dropped_no_positive": sum(
            d["anchors_dropped_no_positive"] for d in docs),
        "pos_rate": round(float(y.mean()), 4),
        "top1_train": round(_top1(
            scores[train], y[train],
            [k for k, t in zip(anchor_keys, train) if t]), 4),
        "top1_valid": round(_top1(
            scores[valid], y[valid],
            [k for k, v in zip(anchor_keys, valid) if v]), 4),
        "top1_valid_release_baseline": round(_top1(
            baseline[valid], y[valid],
            [k for k, v in zip(anchor_keys, valid) if v]), 4),
        "valid_games": sorted({g for g in game_of if is_valid(g)}),
    }

    model = {
        "kind": "logistic-v1",
        "trained": "extract_rank_features corpus (weak PBP labels)",
        "weights": {name: float(w[j])
                    for j, name in enumerate(FEATURE_ORDER)},
        "bias": float(b),
        "mean": {name: float(mean[j])
                 for j, name in enumerate(FEATURE_ORDER)},
        "std": {name: float(std[j])
                for j, name in enumerate(FEATURE_ORDER)},
        "impute": dict(IMPUTE),
        "report": report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
