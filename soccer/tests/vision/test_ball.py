import numpy as np
import pytest

from soccerviz.modeling.manager import lineup_assignment
from soccerviz.vision.ball import actor_hypothesis, ball_windows


def test_actor_abstains_for_missing_distant_or_contested_ball():
    positions = np.array([[10, 10], [12, 10]], float)
    assert actor_hypothesis(positions, np.array([10, 10]))[0] == 0
    assert actor_hypothesis(positions, np.array([11, 10]))[0] is None
    assert actor_hypothesis(positions, np.array([30, 30]))[0] is None
    assert actor_hypothesis(positions, np.array([np.nan, 10]))[0] is None


def test_ball_window_features_are_independent_of_future_positions(tmp_path):
    times = np.arange(40) / 5
    ball = np.column_stack([20 + times, np.full(40, 34)])
    xy = np.broadcast_to(np.array([[10, 10], [70, 60]]), (40, 2, 2)).copy()
    path = tmp_path / "tracking.npz"

    def save():
        np.savez(path, ball=ball, xy=xy, teams=[0, 1], periods=np.ones(40, int), times=times)

    save()
    (inputs, labels, _), origins = ball_windows(path)
    ball[6:11, 0] += 3
    xy[6:11] += 10
    save()
    (altered_inputs, altered_labels, _), altered_origins = ball_windows(path)
    assert origins[0] == altered_origins[0] == 5
    np.testing.assert_array_equal(inputs[0], altered_inputs[0])
    assert not np.array_equal(labels[0], altered_labels[0])


def test_lineup_rejects_duplicate_players_and_string_availability():
    with pytest.raises(ValueError, match="unique"):
        lineup_assignment([{"player_id": "A", "available": True}] * 2, ["GK", "CF"])
    with pytest.raises(ValueError, match="Availability"):
        lineup_assignment([{"player_id": "A", "available": "false"}], ["GK"])
