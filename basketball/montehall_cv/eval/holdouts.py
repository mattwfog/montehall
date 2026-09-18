"""Sealed holdout registry — the single source every mining manifest excludes.

Ground-truth principle 3: holdout games are never
mined, never trained on, never used for threshold tuning. Anything that
builds a mining or training manifest MUST call `is_sealed(event_id)` (or
subtract SEALED_EVENT_IDS) before including a game; scorers use the same
registry to know what they may be scored on.

Sealing is append-only: a game once sealed stays sealed even if its film is
re-downloaded or re-paired. The 2026 NCAA-tournament expansion slots (Ask 1,
different broadcast package) are named when the March Madness grab stage
lands — add entries here, never a parallel list elsewhere.
"""

from __future__ import annotations

SEALED: dict[str, dict] = {
    # ACC tournament 2026 — the original experiment pair (sealed 2026-07-12,
    # registry formalized 2026-07-16, Ask 1)
    "401851175": {
        "name": "Florida State vs California (ACC tournament 2026)",
        "date": "2026-03-11",
        "package": "acc_tournament_2026",
        "sealed": "2026-07-12",
    },
    "401851182": {
        "name": "Duke vs Clemson (ACC tournament 2026)",
        "date": "2026-03-14",
        "package": "acc_tournament_2026",
        "sealed": "2026-07-12",
    },
    # 2026 NCAA-tournament slots: named after the MM grab stage lands
    # (2-4 games, different broadcast package) — Ask 1 completion.
}

SEALED_EVENT_IDS: frozenset[str] = frozenset(SEALED)


def is_sealed(event_id: str) -> bool:
    return str(event_id) in SEALED_EVENT_IDS
