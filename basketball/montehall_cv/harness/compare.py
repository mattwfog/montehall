"""Score adjudicator backends against each other on the same possession traces.

Runs each backend over one job's traces and, given possession-outcome truth,
reports accuracy, calibration (ECE, Brier, reliability bins) and the
precision-at-coverage curve per backend. The question it answers: whose
confidence can be trusted to drive abstention?

Truth is a JSON object {possession_id: outcome} that no model produced (hand
labels, or outcomes derived from aligned play-by-play).

Usage:
    python -m montehall_cv.harness.compare --out /data/out --job-id <id> \
        --truth truth_outcomes.json --backends generative jev --report compare.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from montehall_cv.harness.adjudicators import BACKENDS, make_adjudicator
from montehall_cv.harness.calibration import summarize
from montehall_cv.harness.traces import build_traces
from montehall_cv.store.vlm_cache import VlmCache


def compare(job_dir: Path, truth: dict[int, str], backends: list[str]) -> dict:
    traces = [t for t in build_traces(job_dir) if t["possession_id"] in truth]
    if not traces:
        raise SystemExit("no possession in the job has a truth outcome")
    report = compare_traces(traces, truth, backends, job_dir / "_vlm_cache")
    report["job_dir"] = str(job_dir)
    return report


def compare_traces(
    traces: list[dict],
    truth: dict[int, str],
    backends: list,
    cache_dir: Path,
    scorer_truth: dict[int, int | None] | None = None,
) -> dict:
    """Every backend over the same traces. Outcome accuracy and calibration are
    scored on the verdicts a backend commits to; abstentions ("unclear") are
    counted, never scored as wrong. With scorer_truth, scorer attribution is
    scored on possessions that truly ended in a make."""
    report: dict = {"n_possessions": len(traces), "backends": {}}
    for backend in backends:
        adjudicator = make_adjudicator(backend) if isinstance(backend, str) else backend
        name = adjudicator.name
        cache = VlmCache(cache_dir / f"compare_{name}.jsonl")
        confidence, correct, abstained = [], [], 0
        named = named_right = makes = 0
        for trace in traces:
            pid = trace["possession_id"]
            verdict = adjudicator.adjudicate(trace, cache=cache)
            outcome = verdict.get("outcome")
            if scorer_truth is not None and truth[pid] == "made_fg":
                makes += 1
                if outcome == "made_fg" and verdict.get("scorer_entity") is not None:
                    named += 1
                    named_right += verdict["scorer_entity"] == scorer_truth[pid]
            if outcome in (None, "unclear") or verdict.get("confidence") is None:
                abstained += 1
                continue
            confidence.append(float(verdict["confidence"]))
            correct.append(outcome == truth[pid])
        entry: dict = {
            "model": getattr(adjudicator, "model", None) or getattr(adjudicator, "_model", None),
            "abstained": abstained,
            "answered": len(confidence),
            "coverage": len(confidence) / len(traces),
        }
        if confidence:
            entry.update(summarize(confidence, correct))
        if scorer_truth is not None:
            entry["scorer"] = {
                "true_makes": makes,
                "named": named,
                "named_correct": named_right,
                "precision": named_right / named if named else None,
                "coverage": named / makes if makes else None,
            }
        report["backends"][name] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", choices=sorted(BACKENDS), default=sorted(BACKENDS))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    truth = {int(k): v for k, v in json.loads(args.truth.read_text()).items()}
    report = compare(args.out / args.job_id, truth, args.backends)
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
