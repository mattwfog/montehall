"""GAME STATE: the third state store — the closed world's cheap invariants.

Design decision (2026-08-25): a game carries many structural cues — one
team attacks one end for a full period and then the sides switch — and
those can be identified directly. Each structural
fact is identified ONCE from the footage and then constrains thousands
of frames — replacing per-event polling (the ~2-crop luminance votes
whose pooled cross-tab came back self-contradictory on the HS clip).

v1 identifies, from existing stages only (no new inference):

  possession   team-in-control over time: the ball spine's non-graze
               contact runs joined to the PERSON REGISTRY's team for
               each holder body, melted into team runs.
  direction    which team attacks which rim, PER PERIOD: every spine
               shot yields (rim end, possession team at release); a
               change-point split is searched so a side switch (halves/
               quarters) is DETECTED, never assumed away.

NOT in v1 (stated): live/dead-ball segmentation (needs a clock channel;
substitution-gated registry enrollment waits on it), lineup state.

Consumers: shot_attribution reads shooting-team from possession state
first, then period direction, then its own vote fallbacks.

Stage written: game_state/ — typed timeline rows (kind, span, team,
end, payload).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa

from montehall_cv.store.artifacts import (
    ArtifactWriter,
    read_stage,
    stage_complete,
)

GAME_STATE_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("kind", pa.string()),  # possession | direction
        pa.field("t0_ms", pa.int64()),
        pa.field("t1_ms", pa.int64(), nullable=True),
        pa.field("team", pa.int32(), nullable=True),   # registry cluster
        pa.field("end", pa.string(), nullable=True),   # left | right
        pa.field("payload_json", pa.string(), nullable=True),
    ]
)

POSS_MERGE_GAP_MS = 4000   # same-team contacts this close = one possession
                           # (passes and dribbles within a trip downcourt)
MIN_SHOTS_PER_SIDE = 2     # direction claims per (period, end) need this
SPLIT_MIN_GAIN = 2         # a detected side switch must explain at least
                           # this many extra shots vs the no-switch story


# ------------------------------------------------------------- possession --

def possession_runs(contacts: list[dict],
                    team_of_contact) -> list[dict]:
    """Melt ts-ordered non-graze contacts into team possession runs.
    team_of_contact(contact) -> registry team cluster or None; unknown-
    team contacts extend the current run without breaking it."""
    runs: list[dict] = []
    for c in contacts:
        team = team_of_contact(c)
        t0, t1 = int(c["ts_ms"]), int(c.get("ts_end_ms") or c["ts_ms"])
        cur = runs[-1] if runs else None
        if (cur is not None
                and (team is None or cur["team"] is None
                     or team == cur["team"])
                and t0 - cur["t1_ms"] <= POSS_MERGE_GAP_MS):
            cur["t1_ms"] = max(cur["t1_ms"], t1)
            if cur["team"] is None:
                cur["team"] = team
            cur["contacts"] += 1
        else:
            runs.append({"t0_ms": t0, "t1_ms": t1, "team": team,
                         "contacts": 1})
    return runs


def possession_at(runs: list[dict], ts_ms: int) -> int | None:
    """Team in possession at ts: the run containing ts, else the last
    run ending before it (a shot is released by the team that just had
    the ball)."""
    best = None
    for r in runs:
        if r["t0_ms"] <= ts_ms <= r["t1_ms"]:
            return r["team"]
        if r["t1_ms"] < ts_ms and (best is None
                                   or r["t1_ms"] > best["t1_ms"]):
            best = r
    return best["team"] if best else None


# -------------------------------------------------------------- direction --

def direction_periods(shot_facts: list[tuple[int, str, int]]
                      ) -> list[dict]:
    """(ts_ms, end, team) per shot -> period direction segments.

    Searches ONE change point (a side switch between periods): every
    boundary between consecutive shots is scored by how many shots the
    resulting two end->team maps explain; the split must beat the
    no-split story by SPLIT_MIN_GAIN and each side of it must be
    internally consistent (the two ends map to different teams).
    Multi-switch footage (full games) will call this per detected
    period once a clock channel exists — stated v1 bound."""
    facts = sorted(shot_facts)

    def best_map(chunk: list[tuple[int, str, int]]
                 ) -> tuple[dict, int]:
        by_end: dict[str, Counter] = defaultdict(Counter)
        for _ts, end, team in chunk:
            by_end[end][team] += 1
        out: dict[str, int] = {}
        explained = 0
        for end, c in by_end.items():
            team, n = c.most_common(1)[0]
            if n >= MIN_SHOTS_PER_SIDE:
                out[end] = team
                explained += n
        if len(out) == 2 and len(set(out.values())) == 1:
            return {}, 0  # both ends one team: not a valid map
        return out, explained

    whole_map, whole_n = best_map(facts)
    best_split = None
    for i in range(1, len(facts)):
        m1, n1 = best_map(facts[:i])
        m2, n2 = best_map(facts[i:])
        if not m1 or not m2 or m1 == m2:
            continue
        # a true side switch: some rim end is claimed by DIFFERENT teams
        # on the two sides of the cut — partitioned one-team data is not
        # a switch
        switched = any(end in m2 and m2[end] != team
                       for end, team in m1.items())
        if not switched:
            continue
        if n1 + n2 >= whole_n + SPLIT_MIN_GAIN and (
                best_split is None or n1 + n2 > best_split[0]):
            best_split = (n1 + n2, i, m1, m2)
    if best_split is not None:
        _score, i, m1, m2 = best_split
        cut = (facts[i - 1][0] + facts[i][0]) // 2
        lo, hi = facts[0][0], facts[-1][0]
        return [{"t0_ms": lo, "t1_ms": cut, "map": m1},
                {"t0_ms": cut, "t1_ms": hi, "map": m2}]
    if whole_map:
        return [{"t0_ms": facts[0][0], "t1_ms": facts[-1][0],
                 "map": whole_map}]
    return []


def direction_at(periods: list[dict], ts_ms: int,
                 end: str) -> int | None:
    """Attacking team for `end` at `ts` under the detected periods
    (nearest period when ts falls outside all spans)."""
    best = None
    for p in periods:
        d = 0 if p["t0_ms"] <= ts_ms <= p["t1_ms"] else min(
            abs(ts_ms - p["t0_ms"]), abs(ts_ms - p["t1_ms"]))
        if best is None or d < best[0]:
            best = (d, p)
    return best[1]["map"].get(end) if best else None


# ------------------------------------------------------------------ stage --

def _rim_end_of(x: float, frame_w: float) -> str:
    return "left" if x < frame_w / 2 else "right"


def run(out_root: Path, job_id: str) -> dict:
    job_dir = out_root / job_id
    if stage_complete(job_dir / "game_state"):
        return {"job_id": job_id, "skipped": True,
                "reason": "stage already complete"}
    for need in ("ball_events", "person_registry"):
        if not stage_complete(job_dir / need):
            return {"job_id": job_id, "skipped": True,
                    "reason": f"missing stage: {need}"}
    started = time.monotonic()

    from montehall_cv.pipeline.person_registry import (
        registry_person_index,
        registry_person_teams,
    )
    from montehall_cv.pipeline.shot_attribution import person_at

    pidx = registry_person_index(job_dir)
    person_team = registry_person_teams(job_dir)

    events = read_stage(job_dir / "ball_events").to_pylist()
    contacts = sorted(
        (e for e in events
         if e["kind"] == "contact" and e["track_id"] is not None),
        key=lambda e: e["ts_ms"])

    def team_of_contact(c) -> int | None:
        pid = person_at(pidx, int(c["track_id"]), int(c["ts_ms"]))
        return person_team.get(pid) if pid is not None else None

    runs = possession_runs(contacts, team_of_contact)

    # frame width from detections extent -> left/right rim ends
    det = read_stage(job_dir / "detections")
    frame_w = float(det.column("x2").to_numpy(zero_copy_only=False).max())

    shot_facts: list[tuple[int, str, int]] = []
    for e in events:
        if e["kind"] != "shot":
            continue
        rel = int(e.get("release_ts_ms") or e["ts_ms"])
        team = possession_at(runs, rel)
        end = _rim_end_of(float(e["x"]), frame_w) if e.get("x") else None
        if team is not None and end is not None:
            shot_facts.append((int(e["ts_ms"]), end, team))
    periods = direction_periods(shot_facts)

    writer = ArtifactWriter(job_dir / "game_state", GAME_STATE_SCHEMA)
    for r in runs:
        writer.add({"job_id": job_id, "kind": "possession",
                    "t0_ms": r["t0_ms"], "t1_ms": r["t1_ms"],
                    "team": r["team"], "end": None,
                    "payload_json": json.dumps(
                        {"contacts": r["contacts"]})})
    for p in periods:
        for end, team in p["map"].items():
            writer.add({"job_id": job_id, "kind": "direction",
                        "t0_ms": p["t0_ms"], "t1_ms": p["t1_ms"],
                        "team": team, "end": end, "payload_json": None})
    writer.close()

    live_ms = sum(r["t1_ms"] - r["t0_ms"] for r in runs)
    known = sum(r["t1_ms"] - r["t0_ms"] for r in runs
                if r["team"] is not None)
    return {
        "job_id": job_id,
        "possession_runs": len(runs),
        "possession_known_team": round(known / max(live_ms, 1), 3),
        "shot_facts": len(shot_facts),
        "direction_periods": [
            {"span_s": [p["t0_ms"] / 1000, p["t1_ms"] / 1000],
             "map": p["map"]} for p in periods],
        "side_switch_detected": len(periods) > 1,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


# -------------------------------------------------------------- consumers --

def load_state(job_dir: Path) -> dict | None:
    """{'runs': [...], 'periods': [...]} for consumers; None if absent."""
    if not stage_complete(job_dir / "game_state"):
        return None
    rows = read_stage(job_dir / "game_state").to_pylist()
    runs = [{"t0_ms": r["t0_ms"], "t1_ms": r["t1_ms"], "team": r["team"]}
            for r in rows if r["kind"] == "possession"]
    by_span: dict[tuple, dict] = {}
    for r in rows:
        if r["kind"] == "direction":
            p = by_span.setdefault((r["t0_ms"], r["t1_ms"]),
                                   {"t0_ms": r["t0_ms"],
                                    "t1_ms": r["t1_ms"], "map": {}})
            p["map"][r["end"]] = r["team"]
    return {"runs": runs,
            "periods": sorted(by_span.values(),
                              key=lambda p: p["t0_ms"])}


def shooting_team_cluster(state: dict, release_ts_ms: int | None,
                          anchor_ts_ms: int,
                          end: str | None) -> int | None:
    """The state's answer for who is shooting: possession at release
    first (the team that had the ball), then the period direction for
    the rim end being attacked."""
    ts = release_ts_ms if release_ts_ms is not None else anchor_ts_ms
    team = possession_at(state["runs"], ts)
    if team is not None:
        return team
    if end is not None:
        return direction_at(state["periods"], ts, end)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--job-id", required=True)
    args = ap.parse_args()
    print("GAME_STATE_DONE " + json.dumps(run(args.out, args.job_id)))


if __name__ == "__main__":
    main()
