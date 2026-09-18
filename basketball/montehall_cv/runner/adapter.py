"""Artifact -> legacy results-JSON adapter.

Projects the primitive-store artifacts into the exact dict shape the app's
transform_output_data consumes (stats.team_totals/player_totals keyed
team_a/team_b, events with frame + type + subtype). Honest by
construction: a player row is either jersey-named or an UNNAMED ENTITY row
("e<id>" keys from the entity-first box score) that ships with
track_id=<entity_id>, no jersey, and a "Player A"-style display name — the
app's transformer marks those is_unknown (jersey is None) and the coach
names them via the identity-vote flow. An invented jersey never exists.

Timelines ride the box_events stage (run_boxscore): FGA/FGM as
{frame: subtype} dicts, REB/TOV as frame lists — the transformer's two
consumption shapes. Turnover totals accumulate from TOV events (team
grain). Made-basket analyses carry the scorer (jersey, or track_id+label
for unnamed entities) + shot_class from the matching FGM event (the legacy
pipeline's vocabulary: FGM2/FGM3).
"""

from __future__ import annotations

import json
import string
from pathlib import Path

from montehall_cv.pipeline.run_boxscore import is_entity_key
from montehall_cv.store.artifacts import read_stage, stage_complete

TEAM_KEYS = {0: "team_a", 1: "team_b"}


def build_results(job_dir: Path, fps: float | None) -> dict:
    box_rows = read_stage(job_dir / "box_score").to_pylist()
    shot_events = [
        e for e in read_stage(job_dir / "shot_events").to_pylist() if e["verdict_attempt"]
    ]
    box_events = (
        read_stage(job_dir / "box_events").to_pylist()
        if stage_complete(job_dir / "box_events")
        else []  # artifacts predating the box_events stage
    )
    plays = (
        [json.loads(r["payload_json"]) for r in read_stage(job_dir / "plays").to_pylist()]
        if stage_complete(job_dir / "plays")
        else []  # artifacts predating the plays stage
    )

    team_totals = {k: _zero_totals() for k in TEAM_KEYS.values()}
    player_totals: dict[str, dict] = {k: {} for k in TEAM_KEYS.values()}
    unattributed = _zero_totals()
    unnamed_rows: dict[str, list[dict]] = {k: [] for k in TEAM_KEYS.values()}
    for row in box_rows:
        team_key = TEAM_KEYS.get(row["team_cluster"])
        if team_key is None:
            # Not attributable to either team — must still exist somewhere or
            # game-level truth (total attempts) silently shrinks. The app's
            # transformer reads only team_a/team_b, so this sibling is inert
            # there but keeps the totals honest for any other reader.
            _accumulate(unattributed, row)
            continue
        _accumulate(team_totals[team_key], row)
        if is_entity_key(row["player_key"]):
            unnamed_rows[team_key].append(row)
        elif row["player_key"] != "team":
            player_totals[team_key][row["player_key"]] = _player_dict(row)

    labels = _label_unnamed(unnamed_rows)
    suggestions = _entity_suggestions(job_dir)
    for team_key, rows in unnamed_rows.items():
        for row in rows:
            player_totals[team_key][row["player_key"]] = _unnamed_player_dict(
                row,
                labels[(row["team_cluster"], row["player_key"])],
                suggestions.get(row["entity_id"]),
            )

    team_events, player_events = _project_events(box_events)
    for ev in box_events:  # turnovers exist only at event grain
        team_key = TEAM_KEYS.get(ev["team_cluster"])
        if ev["event_type"] == "TOV" and team_key is not None:
            team_totals[team_key]["turnover"] += 1

    return {
        "stats": {
            "team_totals": team_totals,
            "player_totals": player_totals,
            "unattributed_totals": unattributed,
            "team_events": team_events,
            "player_events": player_events,
        },
        "metadata": {"fps": fps or 30.0},
        "made_basket_play_analyses": _made_baskets(shot_events, box_events, labels),
        # Additive: per-possession play payloads (court-frame trajectories,
        # zone entries, atomic events) — the transformer ignores unknown
        # keys; new consumers read plays directly.
        "plays": plays,
    }


def _project_events(box_events: list[dict]) -> tuple[dict, dict]:
    """box_events rows -> the transformer's two consumption shapes:
    FGA/FGM as {frame: subtype} dicts, REB/TOV as frame lists. Team events
    carry everything (incl. team-line rows); player events only jersey-bound
    rows keyed by jersey."""
    team_events: dict = {k: {} for k in TEAM_KEYS.values()}
    player_events: dict = {k: {} for k in TEAM_KEYS.values()}
    for ev in box_events:
        team_key = TEAM_KEYS.get(ev["team_cluster"])
        if team_key is None:
            continue
        buckets = [team_events[team_key]]
        if ev["player_key"] != "team":
            buckets.append(
                player_events[team_key].setdefault(ev["player_key"], {})
            )
        for bucket in buckets:
            if ev["event_type"] in ("FGA", "FGM"):
                bucket.setdefault(ev["event_type"], {})[ev["frame_idx"]] = ev["subtype"]
            else:
                bucket.setdefault(ev["event_type"], []).append(ev["frame_idx"])
    return team_events, player_events


def _made_baskets(
    shot_events: list[dict], box_events: list[dict],
    labels: dict[tuple[int, str], str],
) -> list[dict]:
    fgm_by_frame = {
        ev["frame_idx"]: ev for ev in box_events if ev["event_type"] == "FGM"
    }
    out = []
    for e in shot_events:
        if not e["verdict_made"]:
            continue
        fgm = fgm_by_frame.get(e["frame_start"])
        out.append(
            {
                "team": TEAM_KEYS.get(fgm["team_cluster"]) if fgm else None,
                "scorer": _scorer(fgm, labels) if fgm else None,
                "shot_class": fgm["subtype"] if fgm else None,  # FGM2 | FGM3
                "goal_time_sec": round(e["ts_start_ms"] / 1000, 1),
                "goal_frame_idx": e["frame_start"],
            }
        )
    return out


def _scorer(fgm: dict, labels: dict[tuple[int, str], str]) -> dict | None:
    key = fgm["player_key"]
    if key == "team":
        return None
    if is_entity_key(key):
        return {
            "track_id": int(key[1:]),
            "label": labels.get((fgm["team_cluster"], key)),
        }
    return {"jersey": key}


def _label_unnamed(unnamed_rows: dict[str, list[dict]]) -> dict[tuple[int, str], str]:
    """(team_cluster, player_key) -> stable "Player A"-style display name,
    ordered per team by stat weight so the biggest unnamed line is always
    "Player A". A gated position descriptor decorates the name."""
    labels: dict[tuple[int, str], str] = {}
    for rows in unnamed_rows.values():
        ranked = sorted(
            rows,
            key=lambda r: (
                -(r["points"] + r["fga"] + r["oreb"] + r["dreb"]
                  + (r.get("ast") or 0)),
                r["entity_id"] or 0,
            ),
        )
        for i, row in enumerate(ranked):
            letter = string.ascii_uppercase[i] if i < 26 else str(i + 1)
            name = f"Player {letter}"
            if row.get("position"):
                name = f"{name} ({row['position']})"
            labels[(row["team_cluster"], row["player_key"])] = name
    return labels


def _unnamed_player_dict(
    row: dict, name: str, suggestion: tuple[str, float] | None = None
) -> dict:
    """Player dict for an unnamed entity row: track_id + display name, NO
    jersey key — the app's transformer marks it is_unknown and the coach
    binds it via the identity-vote flow. A contact-sheet suggestion rides
    as metadata for prompt pre-fill; it never names the row itself."""
    d = _player_dict(row)
    del d["jersey"]
    d["track_id"] = row["entity_id"]
    d["name"] = name
    if suggestion is not None:
        d["suggested_number"], d["suggested_confidence"] = suggestion
    return d


def _entity_suggestions(job_dir: Path) -> dict[int, tuple[str, float]]:
    """entity_id -> (number, confidence) from the contact-sheet VLM reads
    (C1). Tracklet verdicts roll up to the entity by highest-confidence
    agreeing read; conflicting numbers keep the strongest."""
    if not stage_complete(job_dir / "contact_reads"):
        return {}
    entity_of = {
        row["track_id"]: row["entity_id"]
        for row in read_stage(job_dir / "entities").to_pylist()
    }
    best: dict[int, tuple[str, float]] = {}
    for row in read_stage(job_dir / "contact_reads").to_pylist():
        if row["number"] is None:
            continue
        entity_id = entity_of.get(row["track_id"])
        if entity_id is None:
            continue
        if entity_id not in best or row["confidence"] > best[entity_id][1]:
            best[entity_id] = (row["number"], float(row["confidence"]))
    return best


def _zero_totals() -> dict:
    return {
        "fga": 0, "fgm": 0, "fga2": 0, "fgm2": 0, "fga3": 0, "fgm3": 0,
        "fta": 0, "ftm": 0, "rebound": 0, "assist": 0, "turnover": 0,
    }


def _accumulate(totals: dict, row: dict) -> None:
    totals["fga"] += row["fga"]
    totals["fgm"] += row["fgm"]
    totals["fga3"] += row["fga3"]
    totals["fgm3"] += row["fgm3"]
    totals["fga2"] += row["fga"] - row["fga3"]
    totals["fgm2"] += row["fgm"] - row["fgm3"]
    # `or 0`: artifacts written before the FT/AST columns miss the key OR
    # carry a schema-filled NULL — both must read as zero
    totals["fta"] += row.get("fta") or 0
    totals["ftm"] += row.get("ftm") or 0
    totals["assist"] += row.get("ast") or 0
    totals["rebound"] += row["oreb"] + row["dreb"]


def _player_dict(row: dict) -> dict:
    return {
        "jersey": row["player_key"],  # meta: keeps the app's is_unknown False
        "fga": row["fga"],
        "fgm": row["fgm"],
        "fga2": row["fga"] - row["fga3"],
        "fgm2": row["fgm"] - row["fgm3"],
        "fga3": row["fga3"],
        "fgm3": row["fgm3"],
        "fta": row.get("fta") or 0,
        "ftm": row.get("ftm") or 0,
        "rebound": row["oreb"] + row["dreb"],
        "assist": row.get("ast") or 0,
        "turnover": 0,
    }


