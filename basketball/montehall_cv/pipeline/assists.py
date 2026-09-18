"""Assist attribution: the pass that led directly to a made basket.

Pure projection over ball-control samples — no new perception. The
shooter's final control run is walked back to its first sample; the
control immediately before that run is the pass source. NCAA judgment
("leads directly to the basket") becomes two bounds: the handoff gap
(same as the pass_candidate semantics in run_events) and how long the
shooter held the ball before releasing.

Assists are individual-only in the NCAA box score — an assist whose
passer can't take a stat line is dropped, never team-lined or faked.
"""

from __future__ import annotations

MAX_HANDOFF_GAP_MS = 2000  # run_events pass_candidate bound
MAX_SHOOTER_HOLD_MS = 4000  # catch-and-shoot or one move; longer = solo play


def assist_candidate(
    controls: list[dict], shot_ts_ms: int, shooter_entity: int
) -> dict | None:
    """The control sample of the assisting passer, or None.

    controls must be ts-ordered (the ball_controls artifact contract).
    Conditions: the pass arrived within MAX_HANDOFF_GAP_MS of the
    shooter's first control of the final run, from a teammate, and the
    shooter released within MAX_SHOOTER_HOLD_MS of receiving.
    """
    before = [c for c in controls if c["ts_ms"] <= shot_ts_ms]
    if not before or before[-1]["entity_id"] != shooter_entity:
        return None

    # walk back through the shooter's final control run
    run_start = len(before) - 1
    while run_start > 0 and before[run_start - 1]["entity_id"] == shooter_entity:
        run_start -= 1
    received = before[run_start]
    if shot_ts_ms - received["ts_ms"] > MAX_SHOOTER_HOLD_MS:
        return None
    if run_start == 0:
        return None

    passer = before[run_start - 1]
    if passer["entity_id"] == shooter_entity:
        return None
    if passer["team_cluster"] != received["team_cluster"]:
        return None
    if received["ts_ms"] - passer["ts_ms"] > MAX_HANDOFF_GAP_MS:
        return None
    return passer
