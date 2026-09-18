import numpy as np
import pandas as pd
import pytest

from soccerviz.providers.action_value import run_baseline, vaep_values, validate_actions


def actions(game):
    rows = []
    for i in range(30):
        shot = i % 3 == 2
        rows.append(
            {
                "game_id": game,
                "original_event_id": str(i),
                "action_id": i,
                "period_id": 1,
                "time_seconds": float(i * 2),
                "team_id": 1,
                "player_id": 10,
                "start_x": float(20 + i % 3 * 30),
                "start_y": 34.0,
                "end_x": float(50 + i % 3 * 25),
                "end_y": 34.0,
                "bodypart_id": 0,
                "type_id": 11 if shot else 0,
                "result_id": int(not shot or i % 2 == 0),
            }
        )
    return pd.DataFrame(rows)


def test_real_library_xt_and_vaep_are_explicit_about_training(tmp_path):
    pytest.importorskip("socceraction")
    train, evaluate = tmp_path / "train.parquet", tmp_path / "eval.parquet"
    actions(1).to_parquet(train)
    actions(2).to_parquet(evaluate)
    report = run_baseline(
        train,
        evaluate,
        tmp_path / "out",
        orientation="attacking_left_to_right",
        data_kind="synthetic",
        grid=(4, 3),
    )
    assert report["vaep_status"] == "untrained"
    assert report["real_world_value_accuracy"] is None
    assert report["rated_successful_moves"] == 20
    values = pd.read_parquet(tmp_path / "out/action-values.parquet")
    assert np.isfinite(values.xT.dropna()).all()
    with pytest.raises(ValueError, match="disjoint"):
        run_baseline(train, train, tmp_path / "leak", orientation="attacking_left_to_right")
    with pytest.raises(ValueError, match="orientation|left_to_right"):
        run_baseline(train, evaluate, tmp_path / "wrong", orientation="source")


def test_vaep_probabilities_require_matching_keys_and_model_provenance():
    pytest.importorskip("socceraction")
    sample = validate_actions(actions(2))
    probability = sample[["game_id", "action_id"]].assign(scores=0.15, concedes=0.08)
    model = {"model_sha256": "a" * 64, "training_game_ids": [1]}
    values = vaep_values(sample, probability, model)
    assert len(values) == len(sample)
    assert np.isfinite(values.vaep_value).all()
    with pytest.raises(ValueError, match="overlap"):
        vaep_values(sample, probability, {**model, "training_game_ids": [2]})
    with pytest.raises(ValueError, match="finite"):
        vaep_values(sample, probability.iloc[1:], model)
