"""Roster verification primitives (design: roster as identity ground truth).

Shared by the training-label harvest (build_jersey_real) and box-score
attribution (run_boxscore). Rosters come from the montehall `players`
table — either the job_id-keyed prod export or the per-job roster.json
the api writes to input/<job_id>/roster.json — and cover the UPLOADER'S
team only, so every check is scoped to the roster-bound team cluster.

Core rules, applied identically in both consumers:
- cluster binding by trusted-match margin (fail-safe: thin or tied
  evidence binds nothing);
- exact number-set membership;
- truncation-ambiguity rejection: a single-digit read is unusable when
  the digit appears inside any two-digit roster number ("52" read as
  "5" is exactly that collision).
"""

from __future__ import annotations

from collections import defaultdict

MIN_BIND_MATCHES = 2
TEAM_CLUSTERS = (0, 1)  # run_team_assoc: 0/1 = teams, 2 = other/ref, -1 = unknown


def normalize_number(number: str) -> str | None:
    """Roster/read numbers to the pipeline's canonical form ("05" -> "5")."""
    text = number.strip()
    if not text.isdigit() or int(text) > 99:
        return None
    return str(int(text))


def roster_number_set(players: list[dict]) -> set[str]:
    numbers = (normalize_number(p["number"]) for p in players)
    return {n for n in numbers if n is not None}


def roster_entry(raw: dict, job_id: str) -> dict | None:
    """Accept either a job_id-keyed map (prod export) or a bare per-job
    roster.json (what the app writes to input/<job_id>/roster.json)."""
    if "players" in raw:
        return raw
    return raw.get(job_id)


def is_truncation_ambiguous(candidate: str, roster: set[str]) -> bool:
    """Single-digit reads are unusable when the digit appears inside any
    two-digit roster number — truncation makes them indistinguishable."""
    if len(candidate) != 1:
        return False
    return any(len(n) == 2 and candidate in n for n in roster)


def top_posteriors(identity_rows: list[dict], min_posterior: float) -> dict[int, str]:
    """track_id -> top candidate, for tracks whose best candidate clears the bar."""
    best: dict[int, tuple[str, float]] = {}
    for row in identity_rows:
        tid, prob = row["track_id"], row["prob"]
        if tid not in best or prob > best[tid][1]:
            best[tid] = (row["candidate"], prob)
    return {tid: cand for tid, (cand, prob) in best.items() if prob >= min_posterior}


def bind_roster_cluster(
    reads: dict[int, str],
    cluster_of: dict[int, int],
    roster: set[str],
    min_matches: int = MIN_BIND_MATCHES,
) -> tuple[int | None, dict[int, int]]:
    """Pick the team cluster the roster belongs to by trusted-match count.

    Requires a strict margin over the other cluster; ties or thin evidence
    bind nothing (fail-safe: no verification beats a wrong one)."""
    matches: dict[int, int] = defaultdict(int)
    for tid, candidate in reads.items():
        cluster = cluster_of.get(tid)
        if cluster in TEAM_CLUSTERS and candidate in roster \
                and not is_truncation_ambiguous(candidate, roster):
            matches[cluster] += 1
    counts = {c: matches.get(c, 0) for c in TEAM_CLUSTERS}
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    (best_cluster, best_n), (_, other_n) = ranked
    if best_n < min_matches or best_n == other_n:
        return None, counts
    return best_cluster, counts


def verified_tracks(
    reads: dict[int, str],
    cluster_of: dict[int, int],
    roster: set[str],
    bound_cluster: int,
) -> dict[int, str]:
    """track_id -> verified jersey label (bound cluster, in-roster, unambiguous)."""
    return {
        tid: candidate
        for tid, candidate in reads.items()
        if cluster_of.get(tid) == bound_cluster
        and candidate in roster
        and not is_truncation_ambiguous(candidate, roster)
    }


def attribution_allowed(
    candidate: str,
    team_cluster: int,
    bound_cluster: int | None,
    roster: set[str],
) -> bool:
    """May this jersey candidate take a stat line? (attribution phase 1)

    Hard filter, removal-only: for entities in the roster-bound cluster, a
    candidate must be a roster number and truncation-unambiguous — an
    impossible number never takes a stat line (it degrades to the NCAA
    team line instead). The opponent cluster has no roster, so its
    candidates pass through unchanged; with no binding, everything does."""
    if bound_cluster is None or team_cluster != bound_cluster:
        return True
    return candidate in roster and not is_truncation_ambiguous(candidate, roster)
