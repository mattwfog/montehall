"""Box-score assembly: events x entities x jersey posteriors -> stat projection.

Entity-first keying (identity design-of-record, ratified 2026-07-10): a
stat attributed to a tracked entity ALWAYS takes that entity's stat line.
A trusted jersey names the line; without one the line stays an unnamed
entity row (player_key "e<entity_id>") for late binding — by the coach or
by accumulated evidence. Only stats attributable to no entity at all land
on the TEAM line (the NCAA team-rebound/turnover convention). Nothing is
ever fabricated.

Attribution:
- Shooter: entity controlling the ball closest before the shot window
  (nearest on-court player to a ball observation within a lookback).
- 2pt vs 3pt: shooter's court distance to the attacked rim vs the NCAA
  constant-radius arc (22.146 ft).
- Rebound: first ball control after a missed attempt within a timeout;
  offensive/defensive by team vs the shooter's team; no control -> team
  rebound for neither side (unattributed row).
- Free throws: run_freethrows' geometric verdicts reclassify attempts out
  of the field-goal columns into FTA/FTM at one point (missed-FT rebounds
  keep normal credit — the dead-ball-rebound distinction is a known v0
  simplification).
- Assists: control-transition projection (assists.py) on made field
  goals; a passer takes their entity line, named or not — never a
  fabricated jersey.
- Turnovers: harness possession verdicts (confidence-gated) at team grain.
- Roster filter: an impossible jersey read never NAMES a line; the entity
  keeps its stats as an unnamed row (demotion, not deletion).
- Position descriptor: confidence-gated guard/big label from the entity's
  court-position distribution (positions stage); abstains otherwise.

Output: <out>/<job_id>/box_score/ + <out>/<job_id>/box_events/
(per-event attribution rows the adapter projects into the app's clickable
frame-stamped timelines).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa

from montehall_cv.pipeline.aftermath import Aftermath
from montehall_cv.pipeline.assists import assist_candidate
from montehall_cv.pipeline.controls import entity_control_samples
from montehall_cv.pipeline.event_identity import TrackIndex, event_shooter_reads
from montehall_cv.pipeline.shot_attribution import attribution_picks
from montehall_cv.roster import (
    attribution_allowed,
    bind_roster_cluster,
    roster_entry,
    roster_number_set,
    top_posteriors,
)
from montehall_cv.store.artifacts import ArtifactWriter, read_stage, stage_complete
from montehall_cv.store.records import possession_offense_cluster
from montehall_cv.zones import RIM_X_FT, THREE_PT_RADIUS_FT, is_three, rim_distance_ft

BOX_SCORE_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("team_cluster", pa.int8()),  # -1 = unattributed to any team
        # jersey number, "e<entity_id>" (unnamed entity row), or "team"
        pa.field("player_key", pa.string()),
        pa.field("entity_id", pa.int32(), nullable=True),
        pa.field("position", pa.string(), nullable=True),  # guard|big, gated
        pa.field("fga", pa.int32()),
        pa.field("fgm", pa.int32()),
        pa.field("fga3", pa.int32()),
        pa.field("fgm3", pa.int32()),
        pa.field("fta", pa.int32()),
        pa.field("ftm", pa.int32()),
        pa.field("ast", pa.int32()),
        pa.field("oreb", pa.int32()),
        pa.field("dreb", pa.int32()),
        pa.field("points", pa.int32()),
        pa.field("confidence", pa.float32()),
    ]
)

BOX_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("frame_idx", pa.int32()),
        pa.field("ts_ms", pa.int64()),
        pa.field("team_cluster", pa.int8()),
        pa.field("player_key", pa.string()),  # jersey number, or "team"
        pa.field("event_type", pa.string()),  # FGA | FGM | FTA | FTM | AST | REB | TOV
        pa.field("subtype", pa.string(), nullable=True),  # FGA2/FGA3/FGM2/FGM3
    ]
)

SHOOTER_LOOKBACK_MS = 2500
REBOUND_LOOKAHEAD_MS = 5000
MIN_JERSEY_PROB = 0.6
MIN_TOV_CONFIDENCE = 0.5

GEOM_MATCH_PAD_MS = 2000     # spine verdict claims a window span-keyed
FLOW_VETO_FT = 5.0           # away-flow below this vetoes a non-geometric make

# Position descriptor: emitted only on clear evidence, else abstain (None).
POSITION_MIN_SAMPLES = 150
PAINT_DIST_FT = 10.0
GUARD_MIN_PERIMETER_SHARE = 0.45
GUARD_MAX_PAINT_SHARE = 0.20
BIG_MIN_PAINT_SHARE = 0.40
BIG_MAX_PERIMETER_SHARE = 0.20


def run(out_root: Path, job_id: str,
        roster_map: Path | None = None, roster_job_id: str | None = None,
        video: Path | None = None, ocr_weights: Path | None = None) -> dict:
    job_dir = out_root / job_id
    if stage_complete(job_dir / "box_score") and stage_complete(job_dir / "box_events"):
        return {"job_id": job_id, "skipped": True, "reason": "stage already complete"}
    started = time.monotonic()

    entity_of = _entity_map(job_dir)
    roster_info = _roster_binding(job_dir, entity_of, roster_map,
                                  roster_job_id or job_id)
    jersey_of, n_roster_filtered, n_exclusivity = _jersey_map(
        job_dir, entity_of, roster_info
    )
    controls = _ball_controls(job_dir, entity_of)
    windows = read_stage(job_dir / "shot_events").to_pylist()
    geom_summary = _apply_rim_geometry(job_dir, windows)
    move_summary = _movement_veto(job_dir, windows)
    events = [e for e in windows if e["verdict_attempt"]]
    ft_of = _ft_map(job_dir)

    # Event-time shooter reads (item-3 fix, 2026-07-12): the jersey read
    # around the attempt names the row; the cluster name is only a prior.
    shooter_of: dict[int, dict | None] = {
        e["event_id"]: _last_control_before(controls, e["ts_start_ms"], SHOOTER_LOOKBACK_MS)
        for e in events
    }
    entity_tracks: dict[int, list[int]] = defaultdict(list)
    for tid, ent in entity_of.items():
        entity_tracks[ent["entity_id"]].append(tid)
    wants = [
        (e["event_id"], s["entity_id"], e["ts_start_ms"])
        for e in events
        if (s := shooter_of[e["event_id"]]) is not None
    ]
    # Shot-attribution stage picks name the row when the stage ran
    # (structural fix A, 2026-08-25): the composed candidates->sheets->
    # team->launch chain supersedes the event_identity read path — the
    # legacy tracklet-posterior + event-crop reads run only on jobs
    # without a shot_attribution stage (no quark binding or no key).
    track_index = TrackIndex(job_dir, MIN_JERSEY_PROB)
    picks = attribution_picks(job_dir)
    if picks:
        # PERSON-ONLY (2026-08-25): the stage claims persons; the box
        # keys rows "p<person_id>" — stable, nameable once by a human.
        # The number hint rides along for display only and never keys
        # a row (attribution_allowed/smear machinery is a number-claim
        # defense the person key doesn't need).
        ev_reads: dict[int, str] = {
            eid: f"p{p['person_id']}" for eid, p in picks.items()
            if p["person_id"] is not None
        }
    else:
        ev_reads = {}
        for event_id, entity_id, ts_start_ms in wants:
            name = track_index.active_name(entity_tracks.get(entity_id, ()), ts_start_ms)
            if name is not None:
                ev_reads[event_id] = name
        if video is not None:
            crop_reads = event_shooter_reads(
                video, job_dir,
                [w for w in wants if w[0] not in ev_reads],
                dict(entity_tracks), ocr_weights,
            )
            ev_reads = {**crop_reads, **ev_reads}
    bound, roster = roster_info if roster_info else (None, set())
    n_event_named = n_event_conflicts = n_concurrency_vetoed = 0

    lines: dict[tuple[int, str], dict] = defaultdict(
        lambda: {"fga": 0, "fgm": 0, "fga3": 0, "fgm3": 0, "fta": 0, "ftm": 0,
                 "ast": 0, "oreb": 0, "dreb": 0,
                 "points": 0, "entity_id": None, "conf": []}
    )
    n_shooter_attributed = 0
    n_rebounds = 0
    n_ft = 0
    n_assists = 0
    event_rows: list[dict] = []
    for event in events:
        shooter = shooter_of[event["event_id"]]
        made = bool(event["verdict_made"])
        is_ft = ft_of.get(event["event_id"], False)
        key = _line_key(shooter, jersey_of)
        ev_num = ev_reads.get(event["event_id"])
        if shooter is not None and ev_num is not None \
                and ev_num.startswith("p"):
            # person key: names the row directly — the roster/smear
            # defenses exist for NUMBER claims; a person id is already
            # collision-free and self-consistent
            key = (key[0], ev_num)
            n_event_named += 1
        elif shooter is not None and ev_num is not None and attribution_allowed(
            ev_num, shooter["team_cluster"], bound, roster
        ):
            cluster_name = jersey_of.get(shooter["entity_id"])
            if cluster_name is not None and cluster_name != ev_num:
                # smear signal: the event's own evidence contradicts the
                # merged entity's name — never carry a possibly-wrong name
                key = (key[0], f"e{shooter['entity_id']}")
                n_event_conflicts += 1
            else:
                key = (key[0], ev_num)
                n_event_named += 1
        elif (
            shooter is not None
            and ev_num is None
            and key[1] not in ("team",)
            and not is_entity_key(key[1])
            and track_index.ready
            and (
                not track_index.plausibly_one_body(
                    entity_tracks.get(shooter["entity_id"], ())
                )
                or track_index.concurrent_reader_elsewhere(
                    key[1], entity_tracks.get(shooter["entity_id"], ()),
                    event["ts_start_ms"],
                )
            )
        ):
            # the entity provably holds two bodies, or the number is
            # visibly on ANOTHER body right now — the merged entity's
            # prior cannot name this row (Clemson 151.3s class)
            key = (key[0], f"e{shooter['entity_id']}")
            n_concurrency_vetoed += 1
        if shooter is not None:
            n_shooter_attributed += 1
        line = lines[key]
        line["entity_id"] = shooter["entity_id"] if shooter else None
        line["conf"].append(float(event["confidence"] or 0.5))
        if is_ft:
            # run_shots' adjudicator can't tell a FT from a field goal —
            # unclassified, this window would score two points as an FGA2
            n_ft += 1
            line["fta"] += 1
            line["ftm"] += int(made)
            line["points"] += int(made)
            event_rows.append(_event_row(job_id, event["frame_start"],
                                         event["ts_start_ms"], key, "FTA"))
            if made:
                event_rows.append(_event_row(job_id, event["frame_start"],
                                             event["ts_start_ms"], key, "FTM"))
        else:
            # 2 vs 3 is COURT GEOMETRY only (board ripped 2026-08-25)
            is3 = _is_three(shooter, event["court_end"]) if shooter else False
            line["fga"] += 1
            line["fgm"] += int(made)
            line["fga3"] += int(is3)
            line["fgm3"] += int(made and is3)
            line["points"] += (3 if is3 else 2) if made else 0
            event_rows.append(_event_row(job_id, event["frame_start"],
                                         event["ts_start_ms"], key, "FGA",
                                         "FGA3" if is3 else "FGA2"))
            if made:
                event_rows.append(_event_row(job_id, event["frame_start"],
                                             event["ts_start_ms"], key, "FGM",
                                             "FGM3" if is3 else "FGM2"))
                n_assists += _credit_assist(
                    controls, event, shooter, jersey_of,
                    lines, event_rows, job_id,
                )

        if not made:
            rebounder = _first_control_after(
                controls, event["ts_end_ms"], REBOUND_LOOKAHEAD_MS
            )
            shooter_team = key[0]
            if rebounder is not None:
                rb_key = _line_key(rebounder, jersey_of)
                offensive = rb_key[0] == shooter_team and shooter_team >= 0
                lines[rb_key]["oreb" if offensive else "dreb"] += 1
                lines[rb_key]["entity_id"] = rebounder["entity_id"]
                lines[rb_key]["conf"].append(0.5)
                n_rebounds += 1
                event_rows.append(_event_row(job_id, rebounder["frame_idx"],
                                             rebounder["ts_ms"], rb_key, "REB"))

    event_rows.extend(_turnover_rows(job_dir, job_id))
    position_of = _position_map(job_dir)
    _write(job_dir, job_id, lines, position_of)
    _write_events(job_dir, event_rows)
    missed = sum(1 for e in events if not e["verdict_made"])
    named = sum(1 for (_, pk) in lines if pk != "team" and not is_entity_key(pk))
    unnamed = sum(1 for (_, pk) in lines if is_entity_key(pk))
    return {
        "job_id": job_id,
        "attempts": len(events),
        "shooter_attributed": n_shooter_attributed,
        "free_throws": n_ft,
        "assists": n_assists,
        "missed_shots": missed,
        "rebounds_credited": n_rebounds,
        "balance_gap": missed - n_rebounds,  # NCAA invariant: should be 0 when complete
        "stat_lines": len(lines),
        "named_rows": named,
        "entity_rows": unnamed,  # unnamed entity rows awaiting a label
        "positions_labeled": sum(
            1 for line in lines.values()
            if line["entity_id"] is not None and line["entity_id"] in position_of
        ),
        "box_events": len(event_rows),
        "turnovers": sum(1 for r in event_rows if r["event_type"] == "TOV"),
        "roster_bound_cluster": roster_info[0] if roster_info else None,
        "roster_filtered_candidates": n_roster_filtered,
        "conflicted_entities_unnamed": n_exclusivity,
        "event_reads_trusted": len(ev_reads),
        "attribution_source": "shot_attribution" if picks else "event_identity",
        "event_named_rows": n_event_named,
        "event_conflicts_demoted": n_event_conflicts,
        "concurrency_vetoed": n_concurrency_vetoed,
        "movement": move_summary,
        "rim_geometry": geom_summary,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def _movement_veto(job_dir: Path, windows: list[dict]) -> dict:
    """CV-only make sanity (design decision 2026-08-25: the scoreboard is
    never part of the algorithm; made/miss = trajectory >
    VLM > this veto). The collective-movement veto (aftermath.py) demotes
    a frame-judged "made" whose aftermath shows no transition flow —
    vision judging vision. A geometric make is exempt: the ball going
    THROUGH the rim outranks a crowd-flow heuristic."""
    flow = Aftermath(job_dir)
    vetoes = 0
    for w in windows:
        if not (w["verdict_attempt"] and w["verdict_made"]):
            continue
        if w.get("_geom_made"):
            continue
        f = flow.away_flow_ft(w["ts_end_ms"], w["court_end"])
        if f is not None and f < FLOW_VETO_FT:
            w["verdict_made"] = False
            vetoes += 1
    return {"movement_vetoes": vetoes}


def _apply_rim_geometry(job_dir: Path, windows: list[dict]) -> dict:
    """Spine trajectory made/miss onto the VLM windows (2026-08-25).
    The ball spine's rim_pass_verdict — did the track
    go DOWN THROUGH the rim mouth — replaces the VLM's made guess
    wherever the track can judge. Matching is span-keyed (the
    shot_attribution lesson: a window's anchor can sit seconds before
    the shot it resolves on); the LATEST in-span spine verdict wins."""
    if not stage_complete(job_dir / "ball_events"):
        return {"geometry": False}
    verdicts = [
        (int(r["ts_ms"]), bool(r["made_geom"]))
        for r in read_stage(job_dir / "ball_events").to_pylist()
        if r["kind"] in ("shot", "putback") and r["made_geom"] is not None
    ]
    applied = flipped = 0
    for w in windows:
        if not w["verdict_attempt"]:
            continue
        in_span = [(ts, v) for ts, v in verdicts
                   if w["ts_start_ms"] - GEOM_MATCH_PAD_MS <= ts
                   <= w["ts_end_ms"] + GEOM_MATCH_PAD_MS]
        if not in_span:
            continue
        _ts, verdict = max(in_span)
        applied += 1
        if bool(w["verdict_made"]) != verdict:
            flipped += 1
        w["verdict_made"] = verdict
        w["_geom_made"] = verdict
    return {"geometry": True, "geom_applied": applied,
            "geom_flipped": flipped}


def _ft_map(job_dir: Path) -> dict[int, bool]:
    """shot event_id -> free-throw verdict (run_freethrows). Absent stage
    (older artifacts, harvest mode) -> every attempt stays a field goal."""
    if not stage_complete(job_dir / "ft_events"):
        return {}
    return {
        r["event_id"]: bool(r["is_ft"])
        for r in read_stage(job_dir / "ft_events").to_pylist()
    }


def _credit_assist(
    controls: list[dict], event: dict, shooter: dict | None,
    jersey_of: dict[int, str],
    lines: dict, event_rows: list[dict], job_id: str,
) -> int:
    """Assist for a made field goal, individual-only: the passer takes
    their entity line — named or unnamed — never a fabricated jersey."""
    if shooter is None:
        return 0
    passer = assist_candidate(controls, event["ts_start_ms"], shooter["entity_id"])
    if passer is None:
        return 0
    key = _line_key(passer, jersey_of)
    lines[key]["ast"] += 1
    lines[key]["entity_id"] = passer["entity_id"]
    lines[key]["conf"].append(0.5)
    event_rows.append(_event_row(job_id, event["frame_start"],
                                 event["ts_start_ms"], key, "AST"))
    return 1


def _event_row(job_id: str, frame_idx: int, ts_ms: int, key: tuple[int, str],
               event_type: str, subtype: str | None = None) -> dict:
    return {
        "job_id": job_id,
        "frame_idx": int(frame_idx),
        "ts_ms": int(ts_ms),
        "team_cluster": key[0],
        "player_key": key[1],
        "event_type": event_type,
        "subtype": subtype,
    }


def _turnover_rows(job_dir: Path, job_id: str) -> list[dict]:
    """Confidence-gated harness turnover verdicts -> team-grain TOV events.

    possession_offense_cluster reads the honest int field and falls back
    to the pre-2026-07-09 "left"/"right" costume on old artifacts. LLM
    stage optional -> absent = no rows."""
    if not stage_complete(job_dir / "possession_verdicts"):
        return []
    possessions = {
        p["possession_id"]: p for p in read_stage(job_dir / "possessions").to_pylist()
    }
    rows = []
    for verdict in read_stage(job_dir / "possession_verdicts").to_pylist():
        if verdict["outcome"] != "turnover":
            continue
        if (verdict["confidence"] or 0.0) < MIN_TOV_CONFIDENCE:
            continue
        possession = possessions.get(verdict["possession_id"])
        if possession is None:
            continue
        team = possession_offense_cluster(possession)
        if team is None:
            continue
        rows.append(_event_row(job_id, possession["frame_end"],
                               possession["ts_end_ms"], (team, "team"), "TOV"))
    return rows


def _write_events(job_dir: Path, event_rows: list[dict]) -> None:
    writer = ArtifactWriter(job_dir / "box_events", BOX_EVENTS_SCHEMA)
    writer.add_many(sorted(event_rows, key=lambda r: r["frame_idx"]))
    writer.close()


def _entity_map(job_dir: Path) -> dict[int, dict]:
    return {
        row["track_id"]: row for row in read_stage(job_dir / "entities").to_pylist()
    }


def _roster_binding(
    job_dir: Path, entity_of: dict[int, dict],
    roster_map: Path | None, roster_job_id: str,
) -> tuple[int | None, set[str]] | None:
    """(bound_cluster, roster number set), or None when no roster given.

    The roster covers the uploader's team only; binding by trusted-match
    margin decides which k-means cluster that is (fail-safe: none)."""
    if roster_map is None:
        return None
    entry = roster_entry(json.loads(roster_map.read_text()), roster_job_id)
    if entry is None:
        return None
    roster = roster_number_set(entry["players"])
    cluster_of = {tid: e["team_cluster"] for tid, e in entity_of.items()}
    reads = top_posteriors(
        read_stage(job_dir / "identity").to_pylist(), MIN_JERSEY_PROB
    )
    bound, _ = bind_roster_cluster(reads, cluster_of, roster)
    return (bound, roster)


def _jersey_map(
    job_dir: Path, entity_of: dict[int, dict],
    roster_info: tuple[int | None, set[str]] | None = None,
) -> tuple[dict[int, str], int, int]:
    """entity_id -> jersey where any member tracklet has a strong posterior.

    With a roster binding, candidates on the roster-bound cluster that are
    not legal roster numbers (or are truncation-ambiguous) never take a
    stat line — the entity degrades to the NCAA team line instead of a
    phantom jersey (attribution phase 1, removal-only)."""
    bound, roster = roster_info if roster_info else (None, set())
    candidates: dict[int, set[str]] = defaultdict(set)
    n_filtered = 0
    for row in read_stage(job_dir / "identity").to_pylist():
        entity = entity_of.get(row["track_id"])
        if entity is None or row["prob"] < MIN_JERSEY_PROB:
            continue
        if not attribution_allowed(row["candidate"], entity["team_cluster"],
                                   bound, roster):
            n_filtered += 1
            continue
        candidates[entity["entity_id"]].add(row["candidate"])

    # Internal consistency names; internal conflict never does. Same-team
    # duplicates are NOT rivals (2026-07-12: white #24 fragmented across 3
    # entities, all correctly reading 24) — they share the name, and
    # _line_key folds their stats into one row. Bad merges are caught per
    # event by the concurrency veto, not by demoting names here.
    named = {eid: next(iter(nums)) for eid, nums in candidates.items()
             if len(nums) == 1}
    n_conflicted = sum(1 for nums in candidates.values() if len(nums) > 1)
    return named, n_filtered, n_conflicted


def _position_map(job_dir: Path) -> dict[int, str]:
    """entity_id -> confidence-gated position descriptor (guard | big).

    Heuristic over the positions artifact: share of samples in paint
    proximity vs beyond the arc, measured to the NEARER rim (works without
    attacked-end knowledge — bigs live near rims at both ends). Thin or
    ambiguous evidence abstains; the descriptor decorates, never decides."""
    if not stage_complete(job_dir / "positions"):
        return {}
    samples: dict[int, list[float]] = defaultdict(list)
    for row in read_stage(job_dir / "positions").to_pylist():
        if row["entity_id"] is None or row["team_cluster"] not in (0, 1):
            continue
        samples[row["entity_id"]].append(min(
            rim_distance_ft(row["court_x"], row["court_y"], "left"),
            rim_distance_ft(row["court_x"], row["court_y"], "right"),
        ))
    out: dict[int, str] = {}
    for entity_id, dists in samples.items():
        if len(dists) < POSITION_MIN_SAMPLES:
            continue
        arr = np.asarray(dists)
        paint = float(np.mean(arr <= PAINT_DIST_FT))
        perimeter = float(np.mean(arr >= THREE_PT_RADIUS_FT))
        if perimeter >= GUARD_MIN_PERIMETER_SHARE and paint <= GUARD_MAX_PAINT_SHARE:
            out[entity_id] = "guard"
        elif paint >= BIG_MIN_PAINT_SHARE and perimeter <= BIG_MAX_PERIMETER_SHARE:
            out[entity_id] = "big"
    return out


def _ball_controls(job_dir: Path, entity_of: dict[int, dict]) -> list[dict]:
    """Shared computation (controls.py); read the persisted artifact when
    the events stage already ran, compute otherwise (old artifacts)."""
    if stage_complete(job_dir / "ball_controls"):
        return sorted(
            read_stage(job_dir / "ball_controls").to_pylist(),
            key=lambda r: r["ts_ms"],
        )
    return entity_control_samples(job_dir, entity_of)


def _last_control_before(controls: list[dict], ts_ms: int, window_ms: int) -> dict | None:
    eligible = [c for c in controls if ts_ms - window_ms <= c["ts_ms"] <= ts_ms]
    return eligible[-1] if eligible else None


def _first_control_after(controls: list[dict], ts_ms: int, window_ms: int) -> dict | None:
    eligible = [c for c in controls if ts_ms < c["ts_ms"] <= ts_ms + window_ms]
    return eligible[0] if eligible else None


def _is_three(shooter: dict, court_end: str | None) -> bool:
    """Corner-corrected 3pt test (zones.is_three) — the pure-radius test
    scored legitimate corner threes as twos."""
    if court_end not in RIM_X_FT:
        return False
    return is_three(shooter["court_x"], shooter["court_y"], court_end)


def _line_key(control: dict | None, jersey_of: dict[int, str]) -> tuple[int, str]:
    """Entity-first: an attributed entity always takes its own line — named
    by a trusted jersey, else the unnamed "e<entity_id>" row. Only a stat
    with no entity at all falls to the team grain."""
    if control is None:
        return (-1, "team")
    team = control["team_cluster"]
    jersey = jersey_of.get(control["entity_id"])
    return (team, jersey if jersey is not None else f"e{control['entity_id']}")


def is_entity_key(player_key: str) -> bool:
    """Unnamed entity row? (jersey candidates are numeric-only — see
    roster.normalize_number — so the "e" prefix cannot collide)."""
    return player_key.startswith("e")


def _write(job_dir: Path, job_id: str, lines: dict,
           position_of: dict[int, str] | None = None) -> None:
    position_of = position_of or {}
    writer = ArtifactWriter(job_dir / "box_score", BOX_SCORE_SCHEMA)
    for (team, player_key), line in sorted(lines.items()):
        entity_id = line["entity_id"] if player_key != "team" else None
        writer.add(
            {
                "job_id": job_id,
                "team_cluster": team,
                "player_key": player_key,
                "entity_id": entity_id,
                "position": position_of.get(entity_id) if entity_id is not None else None,
                "fga": line["fga"],
                "fgm": line["fgm"],
                "fga3": line["fga3"],
                "fgm3": line["fgm3"],
                "fta": line["fta"],
                "ftm": line["ftm"],
                "ast": line["ast"],
                "oreb": line["oreb"],
                "dreb": line["dreb"],
                "points": line["points"],
                "confidence": float(np.mean(line["conf"])) if line["conf"] else 0.0,
            }
        )
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--roster-map", type=Path, default=None,
                        help="roster JSON (prod export map or per-job "
                             "roster.json); enables the attribution filter")
    parser.add_argument("--roster-job-id", default=None,
                        help="key into a job_id-keyed roster map when it "
                             "differs from --job-id (e.g. short out-dir name)")
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.job_id, roster_map=args.roster_map,
                         roster_job_id=args.roster_job_id), indent=2))


if __name__ == "__main__":
    main()
