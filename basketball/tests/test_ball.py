"""Ball estimator layer: dataset binding, baselines, model smoke.

Pins the properties the ball slice hinges on: truth arrays stay
bin-aligned with the observation features, the holder label lives in
OBSERVED-track space (mapped through the identity permutation), and the
carry-forward observation block behaves like the v1.1 sticky-evidence
contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from montehall_cv.brain.dataset import MODE_IGNORE, MODES
from montehall_cv.brain.dataset_ball import (
    B_AGE,
    B_LAST_X,
    B_PRESENT,
    B_X,
    B_Y,
    HOLDER_IGNORE,
    N_BALL_FEATURES,
    featurize_ball,
    holder_track_labels,
    load_ball_features,
    load_ball_game,
)
from montehall_cv.brain.eval_ball import (
    interp_ball_baseline,
    sticky_ball_baseline,
)
from montehall_cv.brain.sim_traces import generate_game


@pytest.fixture(scope="module")
def ball_game(tmp_path_factory):
    root = tmp_path_factory.mktemp("ball")
    generate_game(root, "g", 42, duration_s=240.0)
    game = load_ball_game(root / "g")
    assert game is not None
    return game


class TestDatasetBall:
    def test_shapes_and_alignment(self, ball_game) -> None:
        t = ball_game["x_global"].shape[0]
        assert ball_game["x_ball"].shape == (t, N_BALL_FEATURES)
        assert ball_game["ball_xy"].shape == (t, 2)
        assert ball_game["mode"].shape == (t,)
        assert ball_game["holder"].shape == (t,)
        assert ball_game["x_tracks"].shape[1] == t

    def test_truth_covers_and_modes_appear(self, ball_game) -> None:
        assert ball_game["pos_mask"].mean() > 0.95, (
            "sim ticks are denser than bins — near-total truth coverage")
        seen = {MODES[m] for m in ball_game["mode"][ball_game["mode"]
                                                    != MODE_IGNORE]}
        assert {"held", "ballistic", "dead"} <= seen

    def test_observed_bins_near_truth(self, ball_game) -> None:
        sel = ball_game["obs_mask"] & ball_game["pos_mask"]
        assert sel.any(), "sim emits ball observations"
        obs = ball_game["x_ball"][sel][:, (B_X, B_Y)]
        truth = ball_game["ball_xy"][sel]
        err_ft = np.hypot((obs[:, 0] - truth[:, 0]) * 94.0,
                          (obs[:, 1] - truth[:, 1]) * 50.0)
        assert np.median(err_ft) < 4.0, (
            "observation noise is ~0.5ft gauss; within-bin drift stays small")

    def test_holder_labels_follow_mode(self, ball_game) -> None:
        k_none = len(ball_game["tracks"])
        held = ball_game["mode"] == MODES.index("held")
        labeled = held & (ball_game["holder"] != HOLDER_IGNORE)
        assert labeled.any()
        assert (ball_game["holder"][labeled] < k_none).all(), (
            "held bins name a real track, never the none slot")
        dead = ball_game["mode"] == MODES.index("dead")
        dead_labeled = dead & (ball_game["holder"] != HOLDER_IGNORE)
        assert (ball_game["holder"][dead_labeled] == k_none).all()

    def test_carry_forward_ages(self, ball_game) -> None:
        x = ball_game["x_ball"]
        first = int(np.flatnonzero(x[:, B_PRESENT] > 0)[0])
        assert (x[:first, B_AGE] == 1.0).all(), "never-observed = age 1.0"
        assert x[first, B_AGE] == 0.0
        assert x[first + 1, B_LAST_X] == x[first, B_X] or (
            x[first + 1, B_PRESENT] > 0)

    def test_load_features_without_truth(self, tmp_path) -> None:
        generate_game(tmp_path, "r", 7, duration_s=120.0)
        (tmp_path / "r" / "sim_states" / "_SUCCESS").unlink()
        feats = load_ball_features(tmp_path / "r")
        assert feats is not None and feats["x_ball"].shape[0] > 1
        assert load_ball_game(tmp_path / "r") is None

    def test_ball_dropout_thins_and_recomputes(self, tmp_path) -> None:
        generate_game(tmp_path, "d", 11, duration_s=240.0)
        full = load_ball_game(tmp_path / "d")
        dropped = load_ball_game(tmp_path / "d", ball_dropout=0.85, seed=1)
        assert full is not None and dropped is not None
        assert dropped["obs_mask"].mean() < full["obs_mask"].mean() * 0.5
        # carry-forward rebuilt from the thinned stream: age 1.0 before
        # the first KEPT observation, and targets untouched by the drop
        x = dropped["x_ball"]
        first = int(np.flatnonzero(x[:, B_PRESENT] > 0)[0])
        assert (x[:first, B_AGE] == 1.0).all()
        np.testing.assert_array_equal(
            np.nan_to_num(dropped["ball_xy"]), np.nan_to_num(full["ball_xy"]))


class TestHolderLabels:
    def test_labels_live_in_observed_space(self) -> None:
        # 2 tracks; truth: entity 1 holds throughout. After a swap the
        # OBSERVED id carrying entity 1 changes — the label must follow.
        labels = np.array([[0, 0, 1, 1],   # track 0: entity 0 then 1
                           [1, 1, 0, 0]])  # track 1: entity 1 then 0
        holder_entity = np.array([1, 1, 1, -1])
        out = holder_track_labels(holder_entity, labels, k_none=2)
        assert out.tolist() == [1, 1, 0, 2]

    def test_ignore_propagates(self) -> None:
        labels = np.array([[0, 0]])
        out = holder_track_labels(
            np.array([HOLDER_IGNORE, 5]), labels, k_none=1)
        assert out.tolist() == [HOLDER_IGNORE, HOLDER_IGNORE]


class TestBallBaselines:
    def test_sticky_and_interp(self) -> None:
        x = np.zeros((5, N_BALL_FEATURES), dtype=np.float32)
        for b, (px, py) in ((1, (0.2, 0.4)), (4, (0.6, 0.4))):
            x[b, B_PRESENT] = 1.0
            x[b, B_X], x[b, B_Y] = px, py
        x[:, B_AGE] = 1.0
        x[1:, B_AGE] = [0.0, 0.025, 0.05, 0.0]
        x[1:, B_LAST_X] = [0.2, 0.2, 0.2, 0.6]
        sticky, s_valid = sticky_ball_baseline(x)
        assert s_valid.tolist() == [False, True, True, True, True]
        assert sticky[2, 0] == pytest.approx(0.2)
        interp, i_valid = interp_ball_baseline(x)
        assert i_valid.tolist() == [False, True, True, True, True]
        assert interp[2, 0] == pytest.approx(0.2 + (0.6 - 0.2) / 3)

    def test_featurize_ball_empty(self) -> None:
        x = featurize_ball([])
        assert x.shape == (1, N_BALL_FEATURES)
        assert x[0, B_AGE] == 1.0


class TestBallModelSmoke:
    def test_forward_and_one_step(self, ball_game) -> None:
        torch = pytest.importorskip("torch")
        from montehall_cv.brain.train_ball import build_model

        model = build_model()
        t = 200
        mode_logits, pos, holder_logits = model(
            torch.from_numpy(ball_game["x_tracks"][:, :t]),
            torch.from_numpy(ball_game["x_global"][:t]),
            torch.from_numpy(ball_game["x_ball"][:t]))
        k = ball_game["x_tracks"].shape[0]
        assert mode_logits.shape == (t, 4)
        assert pos.shape == (t, 2)
        assert holder_logits.shape == (t, k + 1)
        loss = (
            torch.nn.functional.cross_entropy(
                mode_logits,
                torch.from_numpy(ball_game["mode"][:t]),
                ignore_index=MODE_IGNORE)
            + torch.nn.functional.cross_entropy(
                holder_logits,
                torch.from_numpy(ball_game["holder"][:t]),
                ignore_index=HOLDER_IGNORE)
            + torch.nn.functional.mse_loss(
                pos[ball_game["pos_mask"][:t]],
                torch.from_numpy(np.nan_to_num(
                    ball_game["ball_xy"][:t]))[ball_game["pos_mask"][:t]]))
        loss.backward()
        assert float(loss) > 0
