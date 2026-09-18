"""Score adjudicator backends on simulated possessions with exact truth.

    python -m montehall_cv.harness.sim_eval --possessions 300 --seed 7 \
        --backends generative jev --report results/adjudicators-sim.json

See harness/sim_possessions.py for what is simulated and what the noise is.
Verdicts are cached under --cache, so an interrupted run resumes for free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from montehall_cv.harness import sim_possessions
from montehall_cv.harness.adjudicators import BACKENDS, JevAdjudicator
from montehall_cv.harness.compare import compare_traces


def naive_outcome(trace: dict) -> str:
    shots = trace["shot_events"]
    return "turnover" if not shots else ("made_fg" if shots[-1]["made"] else "missed_fg_dreb")


def _naive_rule(pairs: list[tuple[dict, dict]]) -> dict:
    """The no-model reference: trust the last detected shot's made flag, call a
    possession with no detected shot a turnover, name the last ball handler."""
    right = named = named_right = makes = 0
    for trace, truth in pairs:
        outcome = naive_outcome(trace)
        right += outcome == truth["outcome"]
        if truth["outcome"] == "made_fg":
            makes += 1
            if outcome == "made_fg" and trace["ball_controls"]:
                named += 1
                named_right += trace["ball_controls"][-1]["entity"] == truth["scorer_entity"]
    return {
        "accuracy": right / len(pairs),
        "coverage": 1.0,
        "scorer": {
            "true_makes": makes,
            "named": named,
            "named_correct": named_right,
            "precision": named_right / named if named else None,
            "coverage": named / makes if makes else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--possessions", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backends", nargs="+", choices=sorted(BACKENDS), default=sorted(BACKENDS))
    parser.add_argument(
        "--jev-informed",
        action="store_true",
        help="also run jev with the sensors' measured error rates in its state",
    )
    parser.add_argument("--cache", type=Path, default=Path(".sim_eval_cache"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    pairs = sim_possessions.sample(args.possessions, args.seed)
    traces = [trace for trace, _ in pairs]
    truth = {trace["possession_id"]: t["outcome"] for trace, t in pairs}
    scorers = {trace["possession_id"]: t["scorer_entity"] for trace, t in pairs}
    backends: list = list(args.backends)
    if args.jev_informed:
        informed = JevAdjudicator(
            sensor_reliability={
                "shot_detection_recall": sim_possessions.SHOT_DETECT_P,
                "made_flag_accuracy": sim_possessions.MADE_FLAG_P,
            }
        )
        informed.name = "jev-informed"
        backends.append(informed)
    report = compare_traces(
        traces, truth, backends, args.cache, scorer_truth=scorers, reference=naive_outcome
    )
    report["naive_rule"] = _naive_rule(pairs)
    report["setup"] = {
        "possessions": args.possessions,
        "seed": args.seed,
        "shot_detect_p": sim_possessions.SHOT_DETECT_P,
        "made_flag_p": sim_possessions.MADE_FLAG_P,
        "truth_mix": {k: sum(v == k for v in truth.values()) for k in sorted(set(truth.values()))},
    }
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
