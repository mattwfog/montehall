"""Slot estimator layer: dataset binding, baseline, model smoke.

Pins the property v0 lacked by construction: identity evidence stays
bound to the observed track it arrived on, and the training labels are
the sim's exact truth permutation per bin.
"""

from __future__ import annotations

import numpy as np
import pytest

from montehall_cv.brain.dataset_slots import (
    F_PRESENT,
    F_READ,
    F_READ_VALUE,
    LABEL_IGNORE,
    MAX_JERSEY,
    N_TRACK_FEATURES,
    load_slot_game,
    load_track_features,
    parse_jersey,
)
from montehall_cv.brain.eval_slots import sticky_baseline
from montehall_cv.brain.sim_traces import generate_game
from montehall_cv.store.artifacts import read_stage


@pytest.fixture(scope="module")
def sim_game(tmp_path_factory):
    root = tmp_path_factory.mktemp("slots")
    generate_game(root, "g", 42, duration_s=240.0)
    game = load_slot_game(root / "g")
    assert game is not None
    rows = read_stage(root / "g" / "sim_identity").to_pylist()
    return game, rows


class TestDatasetSlots:
    def test_shapes_and_alignment(self, sim_game) -> None:
        game, _rows = sim_game
        k, t, f = game["x_tracks"].shape
        assert k == 10 and f == N_TRACK_FEATURES
        assert game["x_global"].shape[0] == t
        assert game["labels"].shape == (k, t)
        assert game["roster"].shape == (10,)
        assert (game["labels"] != LABEL_IGNORE).all(), (
            "sim supervises every track at every bin")

    def test_labels_start_as_identity_and_flip_at_swaps(self, sim_game) -> None:
        game, rows = sim_game
        tracks = game["tracks"]
        for k, obs in enumerate(tracks):
            assert game["labels"][k, 0] == obs, "tick-0 mapping is identity"
        swaps = [r for r in rows if r["tick"] > 0]
        assert swaps, "test game must contain crossover swaps"
        last = swaps[-1]
        k = tracks.index(last["observed_id"])
        assert game["labels"][k, -1] == last["true_entity"], (
            "label after the final swap must be that swap's new truth")

    def test_jersey_reads_bind_to_their_track(self, sim_game) -> None:
        game, _rows = sim_game
        read_bins = game["x_tracks"][:, :, F_READ] > 0
        assert read_bins.any(), "sim emits jersey reads"
        values = np.round(
            game["x_tracks"][:, :, F_READ_VALUE][read_bins] * MAX_JERSEY)
        roster = set(int(j) for j in game["roster"])
        assert all(int(v) in roster for v in values), (
            "every read value on a track is a roster jersey")

    def test_carry_forward_persists_and_ages(self, sim_game) -> None:
        from montehall_cv.brain.dataset_slots import (
            F_LAST_READ_VALUE,
            F_READ_AGE,
        )
        game, _rows = sim_game
        x = game["x_tracks"]
        k, bins = np.argwhere(x[:, :, F_READ] > 0)[0]
        assert x[k, bins + 1, F_LAST_READ_VALUE] == x[k, bins, F_READ_VALUE]
        assert x[k, bins, F_READ_AGE] == 0.0
        assert x[k, bins + 2, F_READ_AGE] > x[k, bins + 1, F_READ_AGE] or (
            x[k, bins + 2, F_READ] > 0)
        never_read = [kk for kk in range(x.shape[0])
                      if not (x[kk, :, F_READ] > 0).any()]
        for kk in never_read:
            assert (x[kk, :, F_READ_AGE] == 1.0).all()

    def test_load_track_features_works_without_sim_identity(
            self, sim_game, tmp_path) -> None:
        generate_game(tmp_path, "r", 7, duration_s=120.0)
        (tmp_path / "r" / "sim_identity" / "_SUCCESS").unlink()
        real_like = load_track_features(tmp_path / "r")
        assert real_like is not None
        assert real_like["x_tracks"].shape[0] == 10
        assert load_slot_game(tmp_path / "r") is None

    def test_parse_jersey(self) -> None:
        assert parse_jersey("24") == 24
        assert parse_jersey("#5") == 5
        assert parse_jersey(None) is None
        assert parse_jersey("abc") is None
        assert parse_jersey("123") is None


class TestStickyBaseline:
    def test_holds_last_read(self) -> None:
        x = np.zeros((1, 4, N_TRACK_FEATURES), dtype=np.float32)
        x[0, :, F_PRESENT] = 1.0
        x[0, 1, F_READ] = 1.0
        x[0, 1, F_READ_VALUE] = 23 / MAX_JERSEY
        roster = np.array([10, 23], dtype=np.int64)
        pred = sticky_baseline(x, roster)
        assert pred.tolist() == [[-1, 1, 1, 1]]


class TestSlotModelSmoke:
    def test_forward_and_one_step(self, sim_game) -> None:
        torch = pytest.importorskip("torch")
        from montehall_cv.brain.train_slots import build_model

        game, _rows = sim_game
        model = build_model()
        t = 200  # short slice keeps the smoke fast
        logits = model(
            torch.from_numpy(game["x_tracks"][:, :t]),
            torch.from_numpy(game["x_global"][:t]),
            torch.from_numpy(game["roster"]))
        assert logits.shape == (10, t, 10)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 10),
            torch.from_numpy(game["labels"][:, :t]).reshape(-1),
            ignore_index=LABEL_IGNORE)
        loss.backward()
        assert float(loss) > 0
