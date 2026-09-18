import copy

import numpy as np
import pandas as pd
import pytest

from soccerviz.core.analysis import (
    FEATURES,
    control_grid,
    forward_target,
    possessions,
    snapshot_features,
)
from soccerviz.core.data import Match
from soccerviz.core.model import explain, train_baseline
from soccerviz.core.reviews import read_reviews, save_review


@pytest.fixture
def match():
    times = np.arange(101) / 5
    ours = np.column_stack([np.linspace(10, 55, 11), np.linspace(5, 63, 11)])
    theirs = ours + [30, 0]
    xy = np.broadcast_to(np.concatenate([ours, theirs]), (101, 22, 2)).copy()
    return Match(
        1,
        np.arange(101),
        times,
        np.ones(101, dtype=int),
        xy,
        np.column_stack([30 + times * 3, np.full(101, 34)]),
        np.array([0] * 11 + [1] * 11),
        [str(i) for i in range(22)],
        np.tile([1, -1], (101, 1)),
        pd.DataFrame(),
    )


def test_features_are_causal_and_direction_invariant(match):
    original = snapshot_features(match, 20, 0)
    future = copy.deepcopy(match)
    future.xy[21:] = 0
    future.ball[21:] = [0, 0]
    assert snapshot_features(future, 20, 0) == original
    assert forward_target(match, 20, 15, 0) == 1
    assert forward_target(future, 20, 15, 0) == 0
    reflected = copy.deepcopy(match)
    reflected.xy[..., 0] = 105 - reflected.xy[..., 0]
    reflected.ball[..., 0] = 105 - reflected.ball[..., 0]
    reflected.direction *= -1
    assert snapshot_features(reflected, 20, 0) == pytest.approx(original)


def test_missing_observations_and_possession_boundary(match):
    assert forward_target(match, 20, 5, 0) == 0  # Later advancement belongs outside interval.
    match.ball[21:40] = np.nan
    assert forward_target(match, 20, 15, 0) is None
    match.xy[20, :6] = np.nan
    assert snapshot_features(match, 20, 0) is None
    assert np.isnan(control_grid(np.empty((0, 2)), np.array([[10, 10]]))[2]).all()


def test_target_includes_five_second_endpoint_but_excludes_possession_end(match):
    match.ball[:, 0] = 30
    match.ball[45, 0] = 40  # Exactly five seconds after index 20.
    assert forward_target(match, 20, 15, 0) == 1
    assert forward_target(match, 20, match.times[45], 0) == 0
    match.ball[20] = np.nan
    assert forward_target(match, 20, 15, 0) is None


def test_possessions_stop_at_loss_and_restart(match):
    match.events = pd.DataFrame(
        [
            ["Home", 1, "PASS", 1],
            ["Home", 1, "BALL LOST", 5],
            ["Away", 1, "RECOVERY", 6],
            ["Away", 1, "SET PIECE", 9],
            ["Away", 1, "SHOT", 12],
        ],
        columns=["Team", "Period", "Type", "Start Time [s]"],
    )
    segments = possessions(match)
    assert list(segments.start_s) == [1, 6, 9]
    assert list(segments.end_s) == [5, 9, 12]
    assert np.all(segments.start_s.to_numpy()[1:] >= segments.end_s.to_numpy()[:-1])


def test_same_match_training_rejected(tmp_path):
    with pytest.raises(ValueError, match="different matches"):
        train_baseline(tmp_path, tmp_path, 1, 1)


def test_probability_shap_reconstructs_predictions():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(8)
    x = pd.DataFrame(rng.normal(size=(100, len(FEATURES))), columns=FEATURES)
    y = (x.iloc[:, 0] + x.iloc[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=8, max_depth=3, random_state=1).fit(x, y)
    values, base = explain(model, x.iloc[:30], x.iloc[70:])
    np.testing.assert_allclose(
        base + values.sum(axis=1), model.predict_proba(x.iloc[70:])[:, 1], atol=1e-5
    )


def test_reviews_persist_without_overwriting(tmp_path):
    path = tmp_path / "reviews.sqlite"
    save_review(path, "g1-f10", "Incorrect", "A note with 'quotes'; DROP TABLE reviews;")
    save_review(path, "g1-f10", "Correct", "Second look")
    rows = read_reviews(path)
    assert len(rows) == 2
    assert rows.iloc[0].verdict == "Correct"
    assert rows.iloc[1].note.endswith("DROP TABLE reviews;")
