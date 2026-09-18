"""Score adjudicator backends against each other on the same possession traces.

Runs each backend over one job's traces and, given possession-outcome truth,
reports accuracy, calibration (ECE, Brier, reliability bins) and the
precision-at-coverage curve per backend. The question it answers: whose
confidence can be trusted to drive abstention?

Truth is a JSON object {possession_id: outcome} that no model produced (hand
labels, or outcomes derived from aligned play-by-play).

Usage:
    python -m montehall_cv.harness.compare --out /data/out --job-id <id> \
        --truth truth_outcomes.json --backends haiku jev --report compare.json
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
    report: dict = {"job_dir": str(job_dir), "n_possessions": len(traces), "backends": {}}
    for name in backends:
        adjudicator = make_adjudicator(name)
        cache = VlmCache(job_dir / "_vlm_cache" / f"compare_{name}.jsonl")
        confidence, correct, abstained = [], [], 0
        for trace in traces:
            verdict = adjudicator.adjudicate(trace, cache=cache)
            outcome = verdict.get("outcome")
            if outcome in (None, "unclear") or verdict.get("confidence") is None:
                abstained += 1
                continue
            confidence.append(float(verdict["confidence"]))
            correct.append(outcome == truth[trace["possession_id"]])
        entry: dict = {"abstained": abstained, "answered": len(confidence)}
        if confidence:
            entry.update(summarize(confidence, correct))
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
