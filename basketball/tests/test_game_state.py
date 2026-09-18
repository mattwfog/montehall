"""Game-state store: possession melting, direction change-point, lookups."""

from __future__ import annotations

from montehall_cv.pipeline.game_state import (
    POSS_MERGE_GAP_MS,
    direction_at,
    direction_periods,
    possession_at,
    possession_runs,
)


def _contacts(rows):
    return [{"ts_ms": t0, "ts_end_ms": t1, "track_id": tid}
            for t0, t1, tid in rows]


def test_possession_melts_same_team_and_breaks_on_steal():
    team_of = {1: 0, 2: 0, 3: 1}.get
    contacts = _contacts([(0, 500, 1), (1200, 1500, 2),   # team 0 trip
                          (2000, 2600, 3),                 # steal: team 1
                          (2900, 3100, 3)])
    runs = possession_runs(
        contacts, lambda c: team_of(c["track_id"]))
    assert [(r["team"], r["t0_ms"], r["t1_ms"]) for r in runs] == [
        (0, 0, 1500), (1, 2000, 3100)]
    # unknown-team contact extends, never breaks
    contacts = _contacts([(0, 500, 1), (800, 900, 99), (1200, 1500, 2)])
    runs = possession_runs(
        contacts, lambda c: team_of(c["track_id"]))
    assert len(runs) == 1 and runs[0]["team"] == 0
    # same team beyond the merge gap = a new possession
    far = 1500 + POSS_MERGE_GAP_MS + 1
    contacts = _contacts([(0, 1500, 1), (far, far + 400, 2)])
    runs = possession_runs(
        contacts, lambda c: team_of(c["track_id"]))
    assert len(runs) == 2


def test_possession_at_uses_containing_then_last_run():
    runs = [{"t0_ms": 0, "t1_ms": 1000, "team": 0},
            {"t0_ms": 3000, "t1_ms": 4000, "team": 1}]
    assert possession_at(runs, 500) == 0
    assert possession_at(runs, 2000) == 0    # shot released after team 0's run
    assert possession_at(runs, 4500) == 1


def test_direction_detects_the_quarter_switch():
    # The motivating case: team 0 attacks right for the first stretch, then the
    # sides SWITCH. Change-point must find it, not average it away.
    first = [(t, "right", 0) for t in (1000, 5000, 9000)] + \
            [(t, "left", 1) for t in (3000, 7000)]
    second = [(t, "left", 0) for t in (20000, 24000)] + \
             [(t, "right", 1) for t in (22000, 26000)]
    periods = direction_periods(first + second)
    assert len(periods) == 2
    assert periods[0]["map"] == {"right": 0, "left": 1}
    assert periods[1]["map"] == {"left": 0, "right": 1}
    assert direction_at(periods, 2000, "right") == 0
    assert direction_at(periods, 25000, "right") == 1
    # no switch in the data -> one period
    assert len(direction_periods(first)) == 1
    # both ends one team = invalid map, refused
    assert direction_periods([(t, e, 0) for t, e in
                              ((1, "left"), (2, "left"), (3, "right"),
                               (4, "right"))]) == []
