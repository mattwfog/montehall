"""Side-by-side box-score comparison between two pipeline runs.

The stack-upgrade referee: given two run roots holding the same job's
artifacts (e.g. the v0 baseline vs a v2-stack re-run), reports per-team
totals, jersey-bound player rows, and the attribution rate — the share of
attempts that landed on a team (and on a player) instead of the
unattributed bucket. Pure CPU / parquet; safe next to a training run.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from montehall_cv.store.artifacts import read_stage


def summarize(job_dir: Path) -> dict:
    rows = read_stage(job_dir / "box_score").to_pylist()
    teams: dict[str, dict] = defaultdict(lambda: {"fga": 0, "fgm": 0, "points": 0})
    players = []
    unattributed = {"fga": 0, "fgm": 0, "points": 0}
    total = {"fga": 0, "fgm": 0, "points": 0}
    player_attributed_fga = 0
    for row in rows:
        for k in ("fga", "fgm", "points"):
            total[k] += row[k]
        if row["team_cluster"] not in (0, 1):
            for k in ("fga", "fgm", "points"):
                unattributed[k] += row[k]
            continue
        bucket = teams[f"team_{row['team_cluster']}"]
        for k in ("fga", "fgm", "points"):
            bucket[k] += row[k]
        if row["player_key"] != "team":
            players.append(
                {
                    "team": row["team_cluster"],
                    "jersey": row["player_key"],
                    "fga": row["fga"],
                    "fgm": row["fgm"],
                    "points": row["points"],
                    "reb": row["oreb"] + row["dreb"],
                }
            )
            player_attributed_fga += row["fga"]
    return {
        "total": total,
        "teams": dict(teams),
        "unattributed": unattributed,
        "team_attribution_rate": round(
            (total["fga"] - unattributed["fga"]) / total["fga"], 3
        )
        if total["fga"]
        else None,
        "player_attribution_rate": round(player_attributed_fga / total["fga"], 3)
        if total["fga"]
        else None,
        "jersey_bound_players": sorted(
            players, key=lambda p: (p["team"], -p["fga"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="job dir of the baseline run (…/<out_root>/<job_id>)")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="job dir of the new-stack run")
    args = parser.parse_args()
    report = {
        "baseline": summarize(args.baseline),
        "candidate": summarize(args.candidate),
    }
    base, cand = report["baseline"], report["candidate"]
    report["delta"] = {
        "team_attribution_rate": (
            round(cand["team_attribution_rate"] - base["team_attribution_rate"], 3)
            if None not in (cand["team_attribution_rate"], base["team_attribution_rate"])
            else None
        ),
        "player_attribution_rate": (
            round(cand["player_attribution_rate"] - base["player_attribution_rate"], 3)
            if None not in (cand["player_attribution_rate"], base["player_attribution_rate"])
            else None
        ),
        "jersey_bound_players": (
            len(cand["jersey_bound_players"]) - len(base["jersey_bound_players"])
        ),
        "total_fga": cand["total"]["fga"] - base["total"]["fga"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
