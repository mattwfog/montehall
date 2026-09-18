"""Two follow-up measurements on simulated possessions (see sim_possessions.py).

1. SENSOR SWEEP. Same games, same truth, only the two sensor rates change: how
   often a shot is detected and how often its made flag is right. If the
   adjudicator's accuracy is limited by the sensors, it should climb with them
   and meet the no-model rule at perfect sensors.
2. RECALIBRATION. Fit a monotone confidence->accuracy map on one seed's
   verdicts and apply it to another seed's. If the model's overconfidence is
   systematic, held-out calibration error should drop.

    python -m montehall_cv.harness.sim_sweep --report-dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from montehall_cv.harness import calibration, sim_possessions
from montehall_cv.harness.compare import compare_traces
from montehall_cv.harness.sim_eval import _naive_rule

DETECT_GRID = (0.82, 1.0)
FLAG_GRID = (0.80, 0.875, 0.95, 1.0)


def _run(n: int, seed: int, detect: float, flag: float, cache: Path, backend: str) -> tuple[dict, dict]:
    pairs = sim_possessions.sample(n, seed, shot_detect_p=detect, made_flag_p=flag)
    traces = [trace for trace, _ in pairs]
    truth = {trace["possession_id"]: t["outcome"] for trace, t in pairs}
    report = compare_traces(traces, truth, [backend], cache, keep_rows=True)
    return report["backends"][backend], _naive_rule(pairs)


def sweep(n: int, seed: int, cache: Path, backend: str) -> dict:
    cells = []
    for detect in DETECT_GRID:
        for flag in FLAG_GRID:
            entry, naive = _run(n, seed, detect, flag, cache, backend)
            cells.append(
                {
                    "shot_detect_p": detect,
                    "made_flag_p": flag,
                    "naive_rule_accuracy": naive["accuracy"],
                    "accuracy": entry["accuracy"],
                    "coverage": entry["coverage"],
                    "mean_confidence": entry["mean_confidence"],
                    "ece": entry["ece"],
                }
            )
    return {"backend": backend, "possessions": n, "seed": seed, "cells": cells}


def recalibration(n: int, fit_seed: int, test_seed: int, cache: Path, backend: str) -> dict:
    detect, flag = sim_possessions.SHOT_DETECT_P, sim_possessions.MADE_FLAG_P
    fit_entry, _ = _run(n, fit_seed, detect, flag, cache, backend)
    test_entry, _ = _run(n, test_seed, detect, flag, cache, backend)
    steps = calibration.fit_isotonic(*zip(*fit_entry["rows"], strict=True))
    confidence, correct = zip(*test_entry["rows"], strict=True)
    recalibrated = calibration.apply_isotonic(steps, confidence)
    return {
        "backend": backend,
        "fit_seed": fit_seed,
        "test_seed": test_seed,
        "possessions_each": n,
        "map": [{"confidence_up_to": edge, "calibrated": value} for edge, value in steps],
        "test_before": {
            "ece": calibration.expected_calibration_error(confidence, correct),
            "brier": calibration.brier_score(confidence, correct),
            "mean_confidence": sum(confidence) / len(confidence),
            "reliability": calibration.reliability_bins(confidence, correct),
        },
        "test_after": {
            "ece": calibration.expected_calibration_error(recalibrated, correct),
            "brier": calibration.brier_score(recalibrated, correct),
            "mean_confidence": sum(recalibrated) / len(recalibrated),
            "reliability": calibration.reliability_bins(recalibrated, correct),
        },
        "accuracy": test_entry["accuracy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--possessions", type=int, default=300)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--fit-seed", type=int, default=7)
    parser.add_argument("--backend", default="jev")
    parser.add_argument("--cache", type=Path, default=Path(".sim_eval_cache"))
    parser.add_argument("--report-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "adjudicators-sim-sweep.json": sweep(args.possessions, args.seed, args.cache, args.backend),
        "adjudicators-sim-recalibration.json": recalibration(
            args.possessions, args.fit_seed, args.seed, args.cache, args.backend
        ),
    }
    for name, payload in outputs.items():
        (args.report_dir / name).write_text(json.dumps(payload, indent=2) + "\n")
        print("wrote", args.report_dir / name)


if __name__ == "__main__":
    main()
