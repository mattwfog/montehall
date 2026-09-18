"""Evidence-linked briefs and bounded geometric scenarios; no unsupported managerial verdicts."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from soccerviz.core.analysis import FEATURES, snapshot_features
from soccerviz.core.data import load_match, write_json


def lineup_assignment(profiles, roles):
    """Assign available, explicitly scored players to roles; abstain on missing context.

    Profile: {player_id, available: bool, role_scores: {role: [0,1]}}.
    Scores must come from a reviewed scouting/model source. They are not invented.
    """
    if len({p["player_id"] for p in profiles}) != len(profiles):
        raise ValueError("Player IDs must be unique")
    if any(p.get("available") is not None and type(p["available"]) is not bool for p in profiles):
        raise ValueError("Availability must be true, false, or null")
    if any(p.get("available") is None for p in profiles):
        return {"status": "abstain", "reason": "Player availability is missing", "lineup": []}
    eligible = [p for p in profiles if p["available"]]
    if len(eligible) < len(roles):
        return {"status": "abstain", "reason": "Insufficient available players", "lineup": []}
    costs = np.full((len(eligible), len(roles)), 1e6)
    for i, profile in enumerate(eligible):
        for j, role in enumerate(roles):
            value = profile.get("role_scores", {}).get(role)
            if value is not None:
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Role scores must lie in [0,1]")
                costs[i, j] = -value
    ii, jj = linear_sum_assignment(costs)
    if (costs[ii, jj] >= 1e6).any():
        return {
            "status": "abstain",
            "reason": "Missing role evidence for a complete lineup",
            "lineup": [],
        }
    return {
        "status": "assignment_from_supplied_scores",
        "lineup": [
            {
                "player_id": eligible[i]["player_id"],
                "role": roles[j],
                "supplied_score": float(-costs[i, j]),
            }
            for i, j in zip(ii, jj, strict=True)
        ],
        "total_supplied_score": float(-costs[ii, jj].sum()),
        "interpretation": "Optimization of provided scores, not estimated causal match benefit",
    }


def positioning_scenario(match, index, team, player, dx, dy):
    if player not in match.players:
        raise ValueError("Unknown provider player ID")
    j = match.players.index(player)
    if match.teams[j] != team or not np.isfinite(match.xy[index, j]).all():
        raise ValueError("Select an observed player on the possession team")
    if not np.isfinite([dx, dy]).all() or np.hypot(dx, dy) > 10:
        raise ValueError("Scenario moves are bounded to ten metres")
    changed = copy.copy(match)
    changed.xy = match.xy.copy()
    changed.xy[index, j] += [dx, dy]
    if not ((changed.xy[index, j] >= [0, 0]).all() and (changed.xy[index, j] <= [105, 68]).all()):
        raise ValueError("Scenario moves must stay on the pitch")
    before, after = snapshot_features(match, index, team), snapshot_features(changed, index, team)
    if before is None or after is None:
        raise ValueError("Insufficient observations for this scenario")
    return pd.DataFrame(
        {
            "feature": FEATURES,
            "observed": [before[f] for f in FEATURES],
            "scenario": [after[f] for f in FEATURES],
            "change": [after[f] - before[f] for f in FEATURES],
        }
    )


def build_brief(data: Path, artifacts: Path, sample_id: str):
    folder = artifacts / "tactics"
    snapshots = pd.read_parquet(folder / "snapshots.parquet")
    selected = snapshots[snapshots.sample_id == sample_id]
    if selected.empty:
        raise ValueError("Unknown tactical sample")
    row = selected.iloc[0]
    neighbors = pd.read_parquet(folder / "retrieval.parquet")
    neighbors = neighbors[neighbors.query_sample == sample_id]
    options = pd.read_parquet(folder / "pass-options.parquet")
    options = options[options.sample_id == sample_id]
    lines = [
        f"# Analyst brief · {sample_id}",
        "",
        f"Match {int(row.game)} · {'Home' if row.team == 0 else 'Away'} · {row.time_s:.2f}s · possession {row.possession_id}.",
        f"Observed support: {int(row.observed_players)} players. Width {row.width_m:.1f}m; depth {row.depth_m:.1f}m.",
        f"Ball-zone phase: {row.phase_proxy}. Nearest template: {row.shape_template} (descriptive fit only).",
        f"Nearest-opponent pressure proxy: {'within 3m' if row.pressure_proxy else 'more than 3m'}.",
        "",
        "## Pass options to inspect",
        "",
        "Scores imitate completed-pass recipients. Open-lane fractions come from a toy ground-pass simulation; neither establishes the best decision.",
        "",
    ]
    for option in options.itertuples():
        lines.append(
            f"- {option.receiver}: {option.distance_m:.1f}m pass, {option.forward_gain_m:+.1f}m forward; "
            f"choice score {option.choice_score:.3f}, simulated open lane {option.simulated_open_lane_fraction:.0%}."
        )
    lines.extend(["", "## Comparable training-match sequences", ""])
    for neighbor in neighbors.itertuples():
        lines.append(
            f"- Match 1, {neighbor.neighbor_time_s:.2f}s, {neighbor.neighbor_possession}, {neighbor.neighbor_sample}; "
            f"standardized feature distance {neighbor.standardized_distance:.2f}."
        )
    lines.extend(
        [
            "",
            "## Managerial context",
            "",
            (
                "Player names, availability, bench options, fitness, score context, opponent plan, and intended roles are not supplied. "
                "No substitution or lineup recommendation is issued. The lineup solver requires explicit availability and reviewed role scores."
            ),
            "",
            (
                "Evidence: tactics/snapshots.parquet, pass-options.parquet, retrieval.parquet. "
                "A static positioning scenario measures geometric sensitivity; it does not predict opponent adaptation or causal outcome improvement."
            ),
        ]
    )
    return "\n".join(lines)


def run_manager(data: Path, artifacts: Path):
    out = artifacts / "manager"
    out.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(data / "processed/game_2/samples.parquet")
    selected = samples.iloc[len(samples) // 2]
    brief = build_brief(data, artifacts, selected.sample_id)
    (out / "example-brief.md").write_text(brief + "\n")
    match = load_match(data, 2)
    team, i = int(selected.team), int(selected["index"])
    candidates = np.flatnonzero((match.teams == team) & np.isfinite(match.xy[i]).all(axis=1))
    # Reproducible geometric sensitivity example: move one observed teammate 3m toward pitch center.
    j = int(candidates[0])
    dx = 3 if match.xy[i, j, 0] < 52.5 else -3
    scenario = positioning_scenario(match, i, team, match.players[j], dx, 0)
    scenario.to_parquet(out / "example-scenario.parquet", index=False)
    roster = [
        {"player_id": player, "available": None, "role_scores": {}}
        for player, side in zip(match.players, match.teams, strict=True)
        if side == team
    ]
    assignment = lineup_assignment(roster, [f"slot-{j}" for j in range(11)])
    write_json(
        out / "roster-context-template.json",
        {
            "source": "anonymous provider IDs; fill reviewed context before solving",
            "team": "Home" if team == 0 else "Away",
            "roles": ["GK", "LB", "LCB", "RCB", "RB", "LCM", "DM", "RCM", "LW", "CF", "RW"],
            "profiles": roster,
        },
    )
    write_json(
        out / "report.json",
        {
            "brief_sample": selected.sample_id,
            "brief_generated": True,
            "scenario_player": match.players[j],
            "scenario_dx_m": dx,
            "scenario_dy_m": 0,
            "scenario_changed_features": int((scenario.change.abs() > 1e-8).sum()),
            "lineup": assignment,
            "causal_managerial_validation": "unmeasured",
        },
    )
    return {"brief_sample": selected.sample_id, "lineup_status": assignment["status"]}
