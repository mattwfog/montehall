"""Results-JSON -> scorer stages: the inverse of runner/adapter.py.

runner/adapter.build_results projects our stage artifacts into the
results-JSON shape the app consumes (stats.team_totals/player_totals +
an `events` timeline). An external production pipeline emits that same
shape (inference_outputs_*.json: metadata.fps, stats, events[] with
frame_idx / event / team / shot_class / player_with_ball{track_id,
jersey}). This script goes the other way: results JSON -> the three
stage datasets score_e2e_vs_anchors reads (shot_events,
shot_attribution, box_events), so ANY pipeline that speaks the results
schema scores on the ONE scorecard against the same truth files — the
external-pipeline arm of the harness (2026-08-27).

Mapping (observed on an external pipeline's output, 2026-08-24):
  events[event=="fga"]         -> one shot_events window per attempt:
                                  ts_start = frame_idx/fps, ts_end = the
                                  paired outcome event's time (fgm /
                                  shot_miss by the same team+track within
                                  the next attempt), verdict_attempt=True,
                                  verdict_made = paired outcome == fgm
  player_with_ball.jersey      -> shot_attribution.number (their claim)
  player_with_ball.track_id    -> shot_attribution.person_id (their
                                  tracklet id — person consistency then
                                  measures their identity fragmentation)
  team_a/team_b                -> --team-map colours (the truth files
                                  speak white/dark; the map is evidence-
                                  derived and passed explicitly, never
                                  guessed)
  every event                  -> box_events row (FGA/FGM/FTA/FTM/REB/AST/
                                  PF); FGM rows carry ts_ms of the PAIRED
                                  attempt, which is what the scorer's
                                  made_of lookup keys on (our run_boxscore
                                  does the same).

Not produced: box_score (hand-truth files carry no box_truth), ball_track /
detections / ball_events (no primitives in a results JSON — the ball and
spine blocks are correctly absent for this arm).

Usage (inside cvbench):
    python scripts/results_json_to_stages.py \
        --results /work/external_hs_inference_outputs.json \
        --out /work/out-harvest/external_baseline_hs_20260824 \
        --job-id external_baseline_hs_20260824 --team-map team_a=white team_b=dark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa

from montehall_cv.store.artifacts import ArtifactWriter

TEAM_CLUSTER = {"team_a": 0, "team_b": 1}
ATTEMPT_EVENTS = frozenset({"fga"})
OUTCOME_EVENTS = {"fgm": True, "shot_miss": False}
BOX_EVENT_TYPES = {
    "fga": "FGA", "fgm": "FGM", "fta": "FTA", "ftm": "FTM",
    "rebound": "REB", "assist": "AST", "personal_foul": "PF", "turnover": "TOV",
}
DEFAULT_WINDOW_MS = 1000

SHOT_EVENTS_SCHEMA = pa.schema([
    ("job_id", pa.string()), ("event_id", pa.int32()),
    ("frame_start", pa.int32()), ("frame_end", pa.int32()),
    ("ts_start_ms", pa.int64()), ("ts_end_ms", pa.int64()),
    ("court_end", pa.string()), ("n_ball_obs", pa.int32()),
    ("verdict_attempt", pa.bool_()), ("verdict_made", pa.bool_()),
    ("confidence", pa.float32()), ("rationale", pa.string()),
])
SHOT_ATTRIBUTION_SCHEMA = pa.schema([
    ("job_id", pa.string()), ("event_id", pa.int32()),
    ("ts_anchor_ms", pa.int64()), ("track_id", pa.int64()),
    ("rank", pa.int32()), ("picked", pa.bool_()), ("number", pa.string()),
    ("read_confidence", pa.float32()), ("team", pa.string()),
    ("shooting_team", pa.string()), ("release_prox_px", pa.float32()),
    ("whole_prox_px", pa.float32()), ("launch_dist_ft", pa.float32()),
    ("team_match", pa.bool_()), ("person_id", pa.int64()),
    ("n_obs", pa.int32()), ("max_h_px", pa.float32()), ("method", pa.string()),
])
BOX_EVENTS_SCHEMA = pa.schema([
    ("job_id", pa.string()), ("frame_idx", pa.int32()), ("ts_ms", pa.int64()),
    ("team_cluster", pa.int8()), ("player_key", pa.string()),
    ("event_type", pa.string()), ("subtype", pa.string()),
])


def _parse_team_map(specs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in specs:
        key, _, colour = spec.partition("=")
        if key not in TEAM_CLUSTER or not colour:
            raise SystemExit(f"--team-map needs team_a=<colour> team_b=<colour>, got {spec!r}")
        out[key] = colour
    if set(out) != set(TEAM_CLUSTER):
        raise SystemExit("--team-map must name both team_a and team_b")
    return out


def _ts_ms(frame_idx: int, fps: float) -> int:
    return int(round(frame_idx / fps * 1000))


def _pair_outcomes(events: list[dict]) -> list[tuple[dict, dict | None]]:
    """Each attempt with its outcome: the first fgm/shot_miss by the same
    team and track_id before the next attempt; None when unpaired."""
    ordered = sorted(events, key=lambda e: e["frame_idx"])
    pairs: list[tuple[dict, dict | None]] = []
    for i, ev in enumerate(ordered):
        if ev["event"] not in ATTEMPT_EVENTS:
            continue
        shooter = (ev.get("player_with_ball") or {}).get("track_id")
        outcome = None
        for later in ordered[i + 1:]:
            if later["event"] in ATTEMPT_EVENTS:
                break
            same_side = later["team"] == ev["team"]
            same_body = (later.get("player_with_ball") or {}).get("track_id") == shooter
            if later["event"] in OUTCOME_EVENTS and same_side and same_body:
                outcome = later
                break
        pairs.append((ev, outcome))
    return pairs


def _shot_rows(job_id: str, pairs: list, fps: float) -> tuple[list[dict], list[dict]]:
    windows, picks = [], []
    for eid, (att, outcome) in enumerate(pairs):
        body = att.get("player_with_ball") or {}
        ts_start = _ts_ms(att["frame_idx"], fps)
        ts_end = _ts_ms(outcome["frame_idx"], fps) if outcome else ts_start + DEFAULT_WINDOW_MS
        made = OUTCOME_EVENTS[outcome["event"]] if outcome else False
        windows.append({
            "job_id": job_id, "event_id": eid,
            "frame_start": att["frame_idx"],
            "frame_end": outcome["frame_idx"] if outcome else att["frame_idx"],
            "ts_start_ms": ts_start, "ts_end_ms": ts_end,
            "court_end": None, "n_ball_obs": None,
            "verdict_attempt": True, "verdict_made": made,
            "confidence": None,
            "rationale": f"results-json {att['event']} {att.get('shot_class')} -> "
                         f"{outcome['event'] if outcome else 'UNPAIRED'}",
        })
        picks.append({
            "job_id": job_id, "event_id": eid, "ts_anchor_ms": ts_start,
            "track_id": body.get("track_id"), "rank": 0, "picked": True,
            "number": str(body["jersey"]) if body.get("jersey") is not None else None,
            "read_confidence": None, "team": att["_colour"],
            "shooting_team": att["_colour"], "release_prox_px": None,
            "whole_prox_px": None, "launch_dist_ft": None, "team_match": True,
            "person_id": body.get("track_id"), "n_obs": None, "max_h_px": None,
            "method": "results_json",
        })
    return windows, picks


def _box_rows(job_id: str, events: list[dict], pairs: list, fps: float) -> list[dict]:
    attempt_ts = {id(outcome): _ts_ms(att["frame_idx"], fps)
                  for att, outcome in pairs if outcome is not None}
    rows = []
    for ev in sorted(events, key=lambda e: e["frame_idx"]):
        etype = BOX_EVENT_TYPES.get(ev["event"])
        if etype is None:
            continue
        body = ev.get("player_with_ball") or {}
        ts = attempt_ts.get(id(ev), _ts_ms(ev["frame_idx"], fps))
        rows.append({
            "job_id": job_id, "frame_idx": ev["frame_idx"], "ts_ms": ts,
            "team_cluster": TEAM_CLUSTER[ev["team"]],
            "player_key": str(body["jersey"]) if body.get("jersey") is not None else "team",
            "event_type": etype, "subtype": ev.get("shot_class"),
        })
    return rows


def _write(stage_dir: Path, schema: pa.Schema, rows: list[dict]) -> None:
    writer = ArtifactWriter(stage_dir, schema)
    writer.add_many(rows)
    writer.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="game_dir to create")
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--team-map", nargs=2, required=True, metavar="team_x=colour")
    ap.add_argument("--fps", type=float, default=None,
                    help="frame_idx rate; default metadata.fps from the JSON")
    args = ap.parse_args()

    doc = json.loads(args.results.read_text())
    fps = args.fps or float((doc.get("metadata") or {}).get("fps") or 0)
    if not fps:
        raise SystemExit("no fps: pass --fps or a JSON with metadata.fps")
    colours = _parse_team_map(args.team_map)
    events = [dict(e, _colour=colours[e["team"]]) for e in doc["events"]]
    pairs = _pair_outcomes(events)
    windows, picks = _shot_rows(args.job_id, pairs, fps)
    box = _box_rows(args.job_id, events, pairs, fps)

    for stage in ("shot_events", "shot_attribution", "box_events"):
        if (args.out / stage / "_SUCCESS").exists():
            raise SystemExit(f"{args.out / stage} already complete — refusing to overwrite")
    _write(args.out / "shot_events", SHOT_EVENTS_SCHEMA, windows)
    _write(args.out / "shot_attribution", SHOT_ATTRIBUTION_SCHEMA, picks)
    _write(args.out / "box_events", BOX_EVENTS_SCHEMA, box)

    unpaired = [w["event_id"] for w in windows if w["rationale"].endswith("UNPAIRED")]
    print(json.dumps({
        "job_id": args.job_id, "fps": fps, "team_map": colours,
        "events_in": len(doc["events"]), "attempts": len(windows),
        "unpaired_attempts": unpaired, "box_events": len(box),
        "windows": [{k: w[k] for k in ("event_id", "ts_start_ms", "ts_end_ms", "verdict_made")}
                    | {"number": p["number"], "team": p["team"], "person_id": p["person_id"]}
                    for w, p in zip(windows, picks)],
    }, indent=2))


if __name__ == "__main__":
    main()
