"""Calibration and selective-prediction metrics for adjudicator verdicts.

An adjudicator reports a confidence with each outcome. Two questions decide
whether that number can drive abstention:

- Is it calibrated? Of the verdicts reported at 0.8, do about 80% hold up?
  (reliability bins, expected calibration error, Brier score)
- Does thresholding it buy precision? As the threshold rises, coverage falls;
  precision should rise. (precision-at-coverage curve)

Inputs are plain sequences: a confidence per verdict and whether the verdict
matched truth. Truth is never model-descended (see eval/holdouts.py).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _arrays(confidence: Sequence[float], correct: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(confidence, dtype=float)
    hit = np.asarray(correct, dtype=float)
    if conf.shape != hit.shape or conf.ndim != 1:
        raise ValueError("confidence and correct must be 1-D and the same length")
    if conf.size == 0:
        raise ValueError("no verdicts to score")
    if np.any((conf < 0) | (conf > 1)):
        raise ValueError("confidence must lie in [0, 1]")
    return conf, hit


def reliability_bins(
    confidence: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> list[dict]:
    """Equal-width bins over [0, 1]; empty bins are omitted."""
    conf, hit = _arrays(confidence, correct)
    index = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = index == b
        if not mask.any():
            continue
        bins.append(
            {
                "lo": b / n_bins,
                "hi": (b + 1) / n_bins,
                "n": int(mask.sum()),
                "mean_confidence": float(conf[mask].mean()),
                "accuracy": float(hit[mask].mean()),
            }
        )
    return bins


def expected_calibration_error(
    confidence: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> float:
    bins = reliability_bins(confidence, correct, n_bins)
    total = sum(b["n"] for b in bins)
    return float(sum(b["n"] / total * abs(b["accuracy"] - b["mean_confidence"]) for b in bins))


def brier_score(confidence: Sequence[float], correct: Sequence[bool]) -> float:
    conf, hit = _arrays(confidence, correct)
    return float(np.mean((conf - hit) ** 2))


def precision_at_coverage(confidence: Sequence[float], correct: Sequence[bool]) -> list[dict]:
    """One point per distinct threshold: keep verdicts with confidence >= t."""
    conf, hit = _arrays(confidence, correct)
    curve = []
    for threshold in sorted(set(conf.tolist())):
        kept = conf >= threshold
        curve.append(
            {
                "threshold": float(threshold),
                "coverage": float(kept.mean()),
                "precision": float(hit[kept].mean()),
                "n": int(kept.sum()),
            }
        )
    return curve


def summarize(confidence: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> dict:
    conf, hit = _arrays(confidence, correct)
    return {
        "n": int(conf.size),
        "accuracy": float(hit.mean()),
        "mean_confidence": float(conf.mean()),
        "ece": expected_calibration_error(confidence, correct, n_bins),
        "brier": brier_score(confidence, correct),
        "reliability": reliability_bins(confidence, correct, n_bins),
        "precision_at_coverage": precision_at_coverage(confidence, correct),
    }


def fit_isotonic(confidence: Sequence[float], correct: Sequence[bool]) -> list[tuple[float, float]]:
    """Monotone map from reported confidence to observed accuracy (pool adjacent
    violators). Returns (confidence upper edge, calibrated value) steps; fit it
    on one set of verdicts and apply it to another."""
    conf, hit = _arrays(confidence, correct)
    order = np.argsort(conf, kind="stable")
    blocks = [[float(conf[i]), float(hit[i]), 1.0] for i in order]  # [max conf, mean hit, weight]
    merged: list[list[float]] = []
    for block in blocks:
        merged.append(block)
        while len(merged) > 1 and merged[-2][1] > merged[-1][1]:
            hi, lo = merged.pop(), merged.pop()
            weight = hi[2] + lo[2]
            merged.append([hi[0], (hi[1] * hi[2] + lo[1] * lo[2]) / weight, weight])
    return [(edge, value) for edge, value, _ in merged]


def apply_isotonic(steps: list[tuple[float, float]], confidence: Sequence[float]) -> list[float]:
    edges = np.array([edge for edge, _ in steps])
    values = [value for _, value in steps]
    index = np.minimum(np.searchsorted(edges, np.asarray(confidence, dtype=float), side="left"), len(values) - 1)
    return [values[i] for i in index]
