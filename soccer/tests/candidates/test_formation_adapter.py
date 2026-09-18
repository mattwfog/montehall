import numpy as np
import pandas as pd
import pytest

from soccerviz.candidates.formation import (
    eligible_frames,
    goalkeeper_proxies,
    summarize_shapes,
    to_unravel,
    validate_observations,
)


@pytest.fixture
def observations():
    rows = []
    for frame in range(3):
        for team in ("home", "away"):
            for player in range(11):
                x = 2.0 if player == 0 else 25 + player * 3.0
                if team == "away":
                    x = 105 - x
                rows.append(
                    {
                        "frame_id": frame * 25 + 1,
                        "timestamp_s": frame + 0.04,
                        "period": 1,
                        "provider_entity_id": f"{team}_{player}",
                        "team_id": team,
                        "entity": "player",
                        "x_m": x,
                        "y_m": 34 if player == 0 else 6 + (player % 5) * 12,
                        "status": "observed",
                        "coordinate_system": "metres_top_left_105x68",
                    }
                )
        rows.append(
            {
                "frame_id": frame * 25 + 1,
                "timestamp_s": frame + 0.04,
                "period": 1,
                "provider_entity_id": "ball",
                "team_id": None,
                "entity": "ball",
                "x_m": 52.5,
                "y_m": 34,
                "status": "observed",
                "coordinate_system": "metres_top_left_105x68",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def possessions():
    return pd.DataFrame([{"period": 1, "start_s": 0.0, "end_s": 3.0, "team": 0}])


def test_missing_outfield_player_is_omitted_not_filled(observations, possessions):
    proxies = goalkeeper_proxies(observations)
    missing = (observations.frame_id == 26) & (observations.provider_entity_id == "away_5")
    observations.loc[missing, ["x_m", "y_m"]] = np.nan
    observations.loc[missing, "status"] = "unavailable"
    validate_observations(observations)
    audit = eligible_frames(observations, proxies, possessions)
    assert audit.eligible.tolist() == [True, False, True]
    assert audit.loc[1, "away_observed_players"] == 10
    assert "away_incomplete" in audit.loc[1, "omitted_reasons"]


def test_cross_team_identity_contamination_fails(observations):
    observations.loc[
        (observations.frame_id == 26) & (observations.provider_entity_id == "home_1"), "team_id"
    ] = "away"
    with pytest.raises(ValueError, match="multiple teams"):
        validate_observations(observations)


def test_wrong_coordinate_units_fail(observations):
    observations["coordinate_system"] = "normalized_0_1"
    with pytest.raises(ValueError, match="Coordinates"):
        validate_observations(observations)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -2.0, 110.0])
def test_invalid_player_coordinates_are_not_clamped(observations, possessions, value):
    proxies = goalkeeper_proxies(observations)
    observations.loc[
        (observations.frame_id == 26) & (observations.provider_entity_id == "home_1"), "x_m"
    ] = value
    assert eligible_frames(observations, proxies, possessions).eligible.tolist() == [
        True,
        False,
        True,
    ]


def test_missing_ball_and_possession_do_not_become_predictions(observations, possessions):
    proxies = goalkeeper_proxies(observations)
    observations.loc[(observations.frame_id == 26) & (observations.entity == "ball"), "status"] = (
        "unavailable"
    )
    possessions.loc[0, "end_s"] = 1.5
    audit = eligible_frames(observations, proxies, possessions)
    assert audit.eligible.tolist() == [True, False, False]
    assert pd.isna(audit.loc[2, "ball_owning_team_id"])


def test_stability_does_not_bridge_omitted_frames_or_possession_changes():
    assignments = pd.DataFrame(
        {
            "frame_id": [1, 26, 76, 101],
            "team_id": ["home"] * 4,
            "formation": ["442", "442", "433", "433"],
        }
    )
    audit = pd.DataFrame(
        {
            "frame_id": [1, 26, 76, 101],
            "timestamp_s": [0.04, 1.04, 3.04, 4.04],
            "ball_owning_team_id": ["home", "home", "home", "away"],
        }
    )
    report = summarize_shapes(assignments, audit)["home"]
    assert report["comparable_adjacent_pairs_same_phase"] == 1
    assert report["same_phase_adjacent_stability"] == 1.0


def test_actual_unravel_preserves_clocks_coordinates_and_team_assignments(
    observations, possessions
):
    pytest.importorskip("unravel")
    pytest.importorskip("mplsoccer")
    from unravel.soccer import EFPI

    proxies = goalkeeper_proxies(observations)
    audit = eligible_frames(observations, proxies, possessions)
    dataset, report = to_unravel(observations, audit, proxies, game=1)
    assert report["rows_verified"] == 69
    assert report["max_coordinate_error_m"] < 1e-8
    assert report["max_timestamp_error_s"] < 1e-8
    model = EFPI(dataset=dataset).fit(every="frame", formations=["442", "433"])
    result = model.output.to_pandas()
    players = result[result.team_id != "ball"]
    assert len(players) == 66
    assert players.formation.notna().all()
    assert players.groupby(["frame_id", "team_id"]).size().eq(11).all()
    assert (
        (players.id.str.startswith("home_") & (players.team_id == "home"))
        | (players.id.str.startswith("away_") & (players.team_id == "away"))
    ).all()
