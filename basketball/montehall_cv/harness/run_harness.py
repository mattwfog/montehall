"""Possession harness: rulebook adjudication over possession traces.

Tier-S of the harness design: one small-model call per possession
applies the NCAA/FIBA statistician conventions to the symbolic trace —
outcome classification, assist judgment via the FIBA mechanical test, and
anomaly flags for the escalation tier. Every verdict persists with its
cited rule as Evidence. The model behind the call is a swappable backend
(harness/adjudicators.py): `haiku` writes a JSON verdict, `jev` answers typed
questions with probabilities.

Output: <out>/<job_id>/possession_verdicts/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow as pa

from montehall_cv.harness.adjudicators import BACKENDS, make_adjudicator
from montehall_cv.harness.traces import build_traces
from montehall_cv.store.artifacts import ArtifactWriter, stage_complete
from montehall_cv.store.vlm_cache import VlmCache

VERDICTS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("possession_id", pa.int32()),
        pa.field("outcome", pa.string(), nullable=True),
        pa.field("scorer_entity", pa.int32(), nullable=True),
        pa.field("assist_entity", pa.int32(), nullable=True),
        pa.field("confidence", pa.float32(), nullable=True),
        pa.field("anomalies", pa.string(), nullable=True),
        pa.field("rationale", pa.string(), nullable=True),
        pa.field("backend", pa.string(), nullable=True),
        # JSON {outcome: probability}; null for backends that report no distribution
        pa.field("outcome_probabilities", pa.string(), nullable=True),
    ]
)


def run(out_root: Path, job_id: str, backend: str = "haiku") -> dict:
    job_dir = out_root / job_id
    if stage_complete(job_dir / "possession_verdicts"):
        return {"job_id": job_id, "skipped": True, "reason": "stage already complete"}
    started = time.monotonic()

    adjudicator = make_adjudicator(backend)

    traces = build_traces(job_dir)
    cache = VlmCache(job_dir / "_vlm_cache" / "possession_verdicts.jsonl")
    writer = ArtifactWriter(job_dir / "possession_verdicts", VERDICTS_SCHEMA)
    outcomes: dict[str, int] = {}
    n_assists = 0
    for trace in traces:
        verdict = adjudicator.adjudicate(trace, cache=cache)
        outcome = verdict.get("outcome")
        outcomes[outcome or "none"] = outcomes.get(outcome or "none", 0) + 1
        if verdict.get("assist_entity") is not None:
            n_assists += 1
        writer.add(
            {
                "job_id": job_id,
                "possession_id": trace["possession_id"],
                "outcome": outcome,
                "scorer_entity": verdict.get("scorer_entity"),
                "assist_entity": verdict.get("assist_entity"),
                "confidence": verdict.get("confidence"),
                "anomalies": json.dumps(verdict.get("anomalies", [])),
                "rationale": verdict.get("rationale"),
                "backend": adjudicator.name,
                "outcome_probabilities": (
                    json.dumps(verdict["outcome_probabilities"])
                    if verdict.get("outcome_probabilities")
                    else None
                ),
            }
        )
    writer.close()

    return {
        "job_id": job_id,
        "backend": adjudicator.name,
        "possessions_adjudicated": len(traces),
        "outcomes": outcomes,
        "assists_found": n_assists,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="haiku")
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.job_id, backend=args.backend), indent=2))


if __name__ == "__main__":
    main()
