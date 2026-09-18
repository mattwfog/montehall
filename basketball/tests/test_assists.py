"""Assist attribution over ball-control sequences."""

from __future__ import annotations

from montehall_cv.pipeline.assists import assist_candidate


def _c(ts: int, entity: int, team: int) -> dict:
    return {"ts_ms": ts, "entity_id": entity, "team_cluster": team,
            "court_x": 10.0, "court_y": 25.0, "frame_idx": ts // 33}


def test_catch_and_shoot_credits_passer() -> None:
    controls = [_c(0, 1, 0), _c(900, 2, 0), _c(1400, 2, 0)]
    passer = assist_candidate(controls, shot_ts_ms=2000, shooter_entity=2)
    assert passer is not None and passer["entity_id"] == 1


def test_opponent_pass_is_not_an_assist() -> None:
    controls = [_c(0, 1, 1), _c(900, 2, 0)]
    assert assist_candidate(controls, 1500, shooter_entity=2) is None


def test_long_hold_breaks_the_assist() -> None:
    # shooter received at 900 but shot at 6000 — held past the bound
    controls = [_c(0, 1, 0), _c(900, 2, 0), _c(3000, 2, 0), _c(5500, 2, 0)]
    assert assist_candidate(controls, 6000, shooter_entity=2) is None


def test_slow_handoff_is_not_a_pass() -> None:
    # 3s gap between the passer's last touch and the shooter's first
    controls = [_c(0, 1, 0), _c(3200, 2, 0)]
    assert assist_candidate(controls, 3800, shooter_entity=2) is None


def test_unassisted_first_control_run() -> None:
    controls = [_c(0, 2, 0), _c(700, 2, 0)]
    assert assist_candidate(controls, 1200, shooter_entity=2) is None


def test_last_control_must_belong_to_shooter() -> None:
    # someone else touched it after the shooter — attribution mismatch
    controls = [_c(0, 1, 0), _c(500, 2, 0), _c(900, 3, 0)]
    assert assist_candidate(controls, 1200, shooter_entity=2) is None


def test_walkback_spans_shooter_dribbles() -> None:
    # pass at 400 -> shooter controls repeatedly -> shot at 3000: the run
    # start (500) is what the hold bound measures, not the last sample
    controls = [_c(400, 1, 0), _c(500, 2, 0), _c(1500, 2, 0), _c(2800, 2, 0)]
    passer = assist_candidate(controls, 3000, shooter_entity=2)
    assert passer is not None and passer["entity_id"] == 1
