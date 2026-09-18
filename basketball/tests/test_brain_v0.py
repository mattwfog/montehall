"""Brain v0: featurizer leakage guard, targets, matching, train smoke."""

from __future__ import annotations

import json

import numpy as np
import pytest

from montehall_cv.brain.dataset import (
    ALIGN_CHANNELS,
    BIN_MS,
    MODE_IGNORE,
    N_FEATURES,
    PERCEPTION_CHANNELS,
    event_targets,
    featurize,
    load_game,
    mode_targets,
)
from montehall_cv.brain.eval_v0 import extract_events, greedy_match
from montehall_cv.brain.sim_traces import generate_game
from montehall_cv.brain.tokens import CH_BALL, CH_CLOCK, CH_PBP_ANCHOR, make_token


def _anchor(t_ms: int, scoring: bool) -> dict:
    return make_token("g", t_ms, CH_PBP_ANCHOR, text="JumpShot",
                      payload_json=json.dumps(
                          {"shooting_play": True, "scoring_play": scoring}))


class TestDataset:
    def test_anchors_never_reach_features(self) -> None:
        tokens = [
            make_token("g", 1000, CH_CLOCK, value=1100.0, conf=0.9, period=1),
            _anchor(1000, True),
        ]
        x = featurize(tokens)
        no_anchor = featurize(tokens[:1])
        assert np.array_equal(x, no_anchor)

    def test_event_targets_smeared(self) -> None:
        tokens = [_anchor(5000, True)]
        shot, made = event_targets(tokens, 20)
        b = 5000 // BIN_MS
        assert shot[b - 2 : b + 3].tolist() == [1.0] * 5
        assert shot[b - 3] == 0.0 and shot[b + 3] == 0.0
        assert made[b] == 1.0

    def test_featurizer_shapes_and_ball(self) -> None:
        tokens = [make_token("g", 700, CH_BALL, court_x=47.0, court_y=25.0,
                             conf=0.8)]
        x = featurize(tokens)
        assert x.shape == (2, N_FEATURES)
        assert x[1, 8] > 0 and abs(x[1, 9] - 0.5) < 0.01

    def test_sim_game_loads_with_modes(self, tmp_path) -> None:
        generate_game(tmp_path, "simg", 5, duration_s=120.0)
        game = load_game(tmp_path / "simg")
        assert game is not None
        assert game["x"].shape[1] == N_FEATURES
        assert (game["mode"] != MODE_IGNORE).sum() > 0
        assert game["shot"].sum() > 0
        assert len(game["mode"]) == len(game["x"])

    def test_mode_targets_ignore_without_sim_states(self, tmp_path) -> None:
        (tmp_path / "empty").mkdir()
        out = mode_targets(tmp_path / "empty", 10)
        assert (out == MODE_IGNORE).all()


def _mask_tokens() -> list[dict]:
    return [
        make_token("g", 600, CH_CLOCK, value=1100.0, conf=0.9, period=1),
        make_token("g", 1200, CH_BALL, court_x=47.0, court_y=25.0, conf=0.8),
        make_token("g", 5200, CH_BALL, court_x=40.0, court_y=25.0, conf=0.8),
    ]


class TestChannelMasking:
    def test_masks_drop_features_keep_length(self) -> None:
        tokens = _mask_tokens()
        full = featurize(tokens)
        align = featurize(tokens, ALIGN_CHANNELS)
        perc = featurize(tokens, PERCEPTION_CHANNELS)
        # T spans ALL observation channels in every variant
        assert align.shape == full.shape == perc.shape
        assert align[:, 8:12].sum() == 0          # ball features gone
        assert align[1, 0] == 1.0                 # clock features stay
        assert align[2, 34] == 0.0                # rate counts kept tokens only
        assert perc[:, 0:5].sum() == 0            # clock+cut features gone
        assert perc[2, 8] > 0                     # ball features stay
        assert perc[1, 34] == 0.0

    def test_union_mask_is_identity(self) -> None:
        tokens = _mask_tokens()
        both = ALIGN_CHANNELS | PERCEPTION_CHANNELS
        assert np.array_equal(featurize(tokens), featurize(tokens, both))

    def test_masked_load_game_keeps_targets(self, tmp_path) -> None:
        generate_game(tmp_path, "simg", 5, duration_s=120.0)
        full = load_game(tmp_path / "simg")
        perc = load_game(tmp_path / "simg", PERCEPTION_CHANNELS)
        align = load_game(tmp_path / "simg", ALIGN_CHANNELS)
        assert full is not None and perc is not None and align is not None
        assert np.array_equal(full["shot"], perc["shot"])
        assert np.array_equal(full["shot"], align["shot"])
        assert len(perc["x"]) == len(full["x"]) == len(align["x"])


class TestEvalMachinery:
    def test_extract_events_peaks_and_separation(self) -> None:
        probs = np.zeros(100)
        probs[10] = 0.9
        probs[12] = 0.8   # within min separation of the stronger peak
        probs[40] = 0.7
        assert extract_events(probs, 0.5) == [10, 40]

    def test_greedy_match_one_to_one(self) -> None:
        pred = [1000, 5000, 9000]
        truth = [1200, 5100]
        assert greedy_match(pred, truth, 2.0) == 2
        assert greedy_match(pred, truth, 0.05) == 0
        # two preds cannot claim one truth
        assert greedy_match([1000, 1100], [1050], 2.0) == 1


class TestLadderKnobs:
    def test_build_pool_oversamples_matching_paths(self) -> None:
        from pathlib import Path

        from montehall_cv.brain.train_v0 import _build_pool

        dirs = [Path("/a/sim/g1"), Path("/a/rich/g2"), Path("/a/align/g3")]
        pool = _build_pool(dirs, {"/a/sim": 3, "/a/rich": 2})
        assert pool.count(Path("/a/sim/g1")) == 3
        assert pool.count(Path("/a/rich/g2")) == 2
        assert pool.count(Path("/a/align/g3")) == 1
        assert len(pool) == 6

    def test_dropout_channel_selection(self) -> None:
        import random
        from pathlib import Path

        from montehall_cv.brain.train_v0 import _dropout_channels

        rng = random.Random(0)
        roots = [Path("/a/sim")]
        args = (1.0, roots, rng)
        # p=1 under a root: always flips to perception
        assert _dropout_channels(Path("/a/sim/g1"), "all", *args) == "perception"
        # not under a root: untouched
        assert _dropout_channels(Path("/a/align/g3"), "all", *args) == "all"
        # sibling-prefix dir is NOT under the root
        assert _dropout_channels(Path("/a/simX/g"), "all", *args) == "all"
        # p=0 or a non-"all" base: untouched
        assert _dropout_channels(Path("/a/sim/g1"), "all", 0.0, roots, rng) == "all"
        assert _dropout_channels(
            Path("/a/sim/g1"), "perception", *args) == "perception"


class TestTrainSmoke:
    def test_two_epoch_loss_decreases(self, tmp_path) -> None:
        pytest.importorskip("torch")
        from montehall_cv.brain.train_v0 import train

        for i in range(3):
            generate_game(tmp_path / "sim", f"s{i}", 11 + i, duration_s=90.0)
        out = tmp_path / "model"
        train([tmp_path / "sim"], out, epochs=3, exclude_keys=[],
              device="cpu")
        rows = [json.loads(line) for line in
                (out / "train_log.jsonl").read_text().splitlines()]
        assert len(rows) == 3
        assert rows[-1]["shot_loss"] < rows[0]["shot_loss"]
        assert (out / "last.pt").exists()
