"""sim_identity stage: the persisted truth permutation is exact.

The slot-estimator's supervision (contract §1, thesis §10) is the
true_entity -> observed_id mapping. These tests pin the event-sourced
persistence to the sim's actual emission: replaying sim_identity rows
must reproduce the mapping every PLAYER token was degraded with — the
test fails if recording is dropped, mis-ticked, or wired to the wrong
entities.
"""

from __future__ import annotations

import math

from montehall_cv.brain.sim_traces import TICK_MS, generate_game
from montehall_cv.brain.tokens import CH_PLAYER
from montehall_cv.store.artifacts import read_stage


def _gen(tmp_path, name="simid", seed=42, duration_s=240.0):
    meta = generate_game(tmp_path, name, seed, duration_s=duration_s)
    game_dir = tmp_path / name
    return (
        meta,
        read_stage(game_dir / "sim_identity").to_pylist(),
        read_stage(game_dir / "sim_positions").to_pylist(),
        read_stage(game_dir / "tokens").to_pylist(),
    )


def _mapping_timeline(rows: list[dict]) -> list[tuple[int, dict[int, int]]]:
    """Replay rows in tick order -> [(tick, full mapping after that tick)]."""
    timeline: list[tuple[int, dict[int, int]]] = []
    mapping: dict[int, int] = {}
    for r in sorted(rows, key=lambda r: (r["tick"], r["true_entity"])):
        if timeline and timeline[-1][0] == r["tick"]:
            mapping = timeline.pop()[1]
        mapping = {**mapping, r["true_entity"]: r["observed_id"]}
        timeline.append((r["tick"], mapping))
    return timeline


def _mapping_at(timeline, tick: int) -> dict[int, int]:
    current = timeline[0][1]
    for t, mapping in timeline:
        if t > tick:
            break
        current = mapping
    return current


class TestSimIdentity:
    def test_initial_mapping_is_identity(self, tmp_path) -> None:
        _meta, rows, _pos, _tok = _gen(tmp_path)
        initial = [r for r in rows if r["tick"] == 0]
        assert len(initial) == 10
        assert all(r["observed_id"] == r["true_entity"] for r in initial)
        assert all(r["t_ms"] == 0 for r in initial)

    def test_swaps_recorded_and_final_mapping_is_permutation(
            self, tmp_path) -> None:
        meta, rows, _pos, _tok = _gen(tmp_path)
        assert meta["identity_swaps"] > 0, (
            "seed/duration produced no crossover swaps; the replay test "
            "would not exercise permutation reconstruction")
        assert len(rows) == 10 + 2 * meta["identity_swaps"]
        final = _mapping_timeline(rows)[-1][1]
        assert sorted(final) == list(range(10))
        assert sorted(final.values()) == list(range(10))

    def test_replay_matches_emitted_player_tokens(self, tmp_path) -> None:
        """The recorded mapping is the one the tokens were degraded with:
        for each PLAYER token, the true entity nearest the (noised) token
        position must map to exactly the token's observed entity_id."""
        _meta, rows, pos, tok = _gen(tmp_path)
        timeline = _mapping_timeline(rows)
        true_at_tick: dict[int, dict[int, tuple[float, float]]] = {}
        for p in pos:
            if p["cls"] == 0:
                true_at_tick.setdefault(p["frame_idx"], {})[p["entity_id"]] = (
                    p["court_x"], p["court_y"])
        player_tokens = [t for t in tok if t["channel"] == CH_PLAYER]
        assert player_tokens
        matched = 0
        for t in player_tokens:
            tick = t["t_ms"] // TICK_MS
            nearest = min(
                true_at_tick[tick],
                key=lambda e: math.hypot(
                    true_at_tick[tick][e][0] - t["court_x"],
                    true_at_tick[tick][e][1] - t["court_y"]))
            if _mapping_at(timeline, tick)[nearest] == t["entity_id"]:
                matched += 1
        # token positions carry gauss(0, 0.7) noise; non-occluded entities
        # are >= 3 ft apart, so nearest-true attribution is right for the
        # overwhelming majority — anything below this bound means the
        # recorded permutation is not the one emission used
        assert matched / len(player_tokens) > 0.97

    def test_deterministic_per_seed(self, tmp_path) -> None:
        _m1, r1, _p1, _t1 = _gen(tmp_path / "a", seed=123)
        _m2, r2, _p2, _t2 = _gen(tmp_path / "b", seed=123)
        assert r1 == r2
