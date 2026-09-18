"""Sim state-trace generator: determinism, invariants, degradation."""

from __future__ import annotations

import json

from montehall_cv.brain.sim_traces import generate_game
from montehall_cv.brain.tokens import (
    CH_CLOCK,
    CH_PBP_ANCHOR,
    CH_PLAYER,
    CH_SCORE_DELTA,
)
from montehall_cv.store.artifacts import read_stage


def _gen(tmp_path, name="simtest", seed=42, duration_s=240.0):
    meta = generate_game(tmp_path, name, seed, duration_s=duration_s)
    game_dir = tmp_path / name
    return (
        meta,
        read_stage(game_dir / "sim_states").to_pylist(),
        read_stage(game_dir / "sim_positions").to_pylist(),
        read_stage(game_dir / "tokens").to_pylist(),
    )


class TestSimTraces:
    def test_deterministic_per_seed(self, tmp_path) -> None:
        m1 = generate_game(tmp_path / "a", "g", 123, duration_s=120.0)
        m2 = generate_game(tmp_path / "b", "g", 123, duration_s=120.0)
        assert m1["total"] == m2["total"]
        assert m1["final_score"] == m2["final_score"]
        assert m1["makes"] == m2["makes"]

    def test_state_stream_dense_and_clock_lawful(self, tmp_path) -> None:
        meta, states, _, _ = _gen(tmp_path)
        assert len(states) == meta["ticks"]
        for a, b in zip(states, states[1:]):
            if b["period"] == a["period"]:
                assert b["clock_s"] <= a["clock_s"] + 1e-6
                if b["clock_s"] < a["clock_s"] - 1e-6:
                    assert not a["dead_ball"]  # clock only runs live

    def test_modes_and_holder_consistent(self, tmp_path) -> None:
        _, states, _, _ = _gen(tmp_path)
        for s in states:
            assert s["ball_mode"] in ("held", "dribbled", "ballistic", "dead")
            if s["ball_mode"] in ("held", "dribbled"):
                assert s["holder_entity"] is not None
            if s["ball_mode"] in ("ballistic", "dead"):
                assert s["holder_entity"] is None

    def test_scores_reconcile_across_layers(self, tmp_path) -> None:
        meta, states, _, tokens = _gen(tmp_path)
        final = states[-1]
        assert [final["score_home"], final["score_guest"]] == meta["final_score"]
        made_anchor_pts = sum(
            int(t["value"]) for t in tokens
            if t["channel"] == CH_PBP_ANCHOR
            and json.loads(t["payload_json"])["scoring_play"]
        )
        assert made_anchor_pts == sum(meta["final_score"])
        delta_pts = sum(int(t["value"]) for t in tokens
                        if t["channel"] == CH_SCORE_DELTA)
        # board deltas lag 1-5s; at most the trailing makes are unposted
        assert 0 <= made_anchor_pts - delta_pts <= 6

    def test_observations_are_degraded_truth(self, tmp_path) -> None:
        meta, states, positions, tokens = _gen(tmp_path)
        n_ticks = meta["ticks"]
        # truth is dense: 11 rows per tick (10 players + ball)
        assert len(positions) == n_ticks * 11
        player_tokens = [t for t in tokens if t["channel"] == CH_PLAYER]
        # observations are strictly sparser than truth (drops + occlusion)
        assert 0 < len(player_tokens) < n_ticks * 10
        clock_tokens = [t for t in tokens if t["channel"] == CH_CLOCK]
        assert len(clock_tokens) >= n_ticks * 0.2 / (2.0 / 0.2) / 2

    def test_token_stream_ordered_and_indexed(self, tmp_path) -> None:
        _, _, _, tokens = _gen(tmp_path)
        assert [t["token_idx"] for t in tokens] == list(range(len(tokens)))
        assert all(a["t_ms"] <= b["t_ms"] for a, b in zip(tokens, tokens[1:]))

    def test_resume_skips(self, tmp_path) -> None:
        generate_game(tmp_path, "g2", 9, duration_s=120.0)
        again = generate_game(tmp_path, "g2", 9, duration_s=120.0)
        assert again.get("skipped") is True
