"""LLM possession harness: rulebook adjudication over possession traces.

Tier-S of the ratified harness design: one small-model call per possession
applies the NCAA/FIBA statistician conventions to the symbolic trace —
outcome classification, assist judgment via the FIBA mechanical test, and
anomaly flags for the escalation tier. Every verdict persists with its
cited rule as Evidence.

Output: <out>/<job_id>/possession_verdicts/
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pyarrow as pa

from montehall_cv.harness.traces import build_traces, trace_json
from montehall_cv.store.artifacts import ArtifactWriter, stage_complete
from montehall_cv.store.vlm_cache import VlmCache, content_key

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
    ]
)

MODEL = "claude-haiku-4-5-20251001"

SYSTEM = """You are a basketball statistician applying official scoring conventions to \
possession traces from a computer-vision pipeline. Rules you enforce:

- OUTCOME per FIBA possession definition: a possession ends by made FG, defensive \
rebound after a miss, or turnover; an offensive rebound CONTINUES the possession.
- ASSIST per the FIBA mechanical test: only the LAST pass before the shot counts; \
a pass to a player who scores from the paint is always an assist; a pass to a \
player outside the paint who scores without dribbling is always an assist; with \
dribbles it is an assist only if the shooter did not have to beat their own \
defender. If ball-control data cannot establish a qualifying last pass, there is \
NO assist.
- DOUBT DEFAULTS: doubt about rebound control -> assume control; doubt about act \
of shooting -> assume NOT shooting; unattributable stats belong to the TEAM, \
never to a guessed player.
- The trace may be incomplete (sparse ball detection). Flag anomalies rather than \
inventing facts. Trust shot_events (VLM-adjudicated) over raw ball_controls when \
they conflict.

Respond ONLY with JSON:
{"outcome": "made_fg"|"missed_fg_dreb"|"missed_fg_oreb"|"turnover"|"unclear",
 "scorer_entity": int|null, "assist_entity": int|null,
 "confidence": 0.0-1.0, "anomalies": ["..."], "rationale": "<=2 sentences"}"""


def run(out_root: Path, job_id: str) -> dict:
    job_dir = out_root / job_id
    if stage_complete(job_dir / "possession_verdicts"):
        return {"job_id": job_id, "skipped": True, "reason": "stage already complete"}
    started = time.monotonic()

    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)

    traces = build_traces(job_dir)
    cache = VlmCache(job_dir / "_vlm_cache" / "possession_verdicts.jsonl")
    writer = ArtifactWriter(job_dir / "possession_verdicts", VERDICTS_SCHEMA)
    outcomes: dict[str, int] = {}
    n_assists = 0
    for trace in traces:
        verdict = _adjudicate(client, trace, cache=cache)
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
            }
        )
    writer.close()

    return {
        "job_id": job_id,
        "possessions_adjudicated": len(traces),
        "outcomes": outcomes,
        "assists_found": n_assists,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def _adjudicate(client, trace: dict, cache: VlmCache | None = None) -> dict:
    payload = trace_json(trace)
    key = content_key(MODEL, SYSTEM, payload)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM,
        messages=[{"role": "user", "content": payload}],
    )
    text = next((b.text for b in response.content if hasattr(b, "text")), "")
    verdict: dict = {}
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            verdict = parsed
    if cache is not None:
        cache.put(key, verdict)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.job_id), indent=2))


if __name__ == "__main__":
    main()
