import json

import numpy as np
import pandas as pd
import pytest

from soccerviz.core.assets import sha256
from soccerviz.core.data import write_json
from soccerviz.modeling.action_training import (
    load_splits,
    predict_probabilities,
    probability_metrics,
    state_tables,
    train_baselines,
)


def actions(game):
    rows = []
    for i in range(160):
        shot = i % 10 == 9
        rows.append(
            {
                "game_id": game,
                "original_event_id": str(i),
                "action_id": i,
                "period_id": 1 if i < 80 else 2,
                "time_seconds": float(i % 80 * 2),
                "team_id": 1 if i % 10 < 5 else 2,
                "player_id": 10,
                "start_x": float(20 + i % 3 * 30),
                "start_y": 34.0,
                "end_x": float(50 + i % 3 * 25),
                "end_y": 34.0,
                "bodypart_id": 0,
                "type_id": 11 if shot else 0,
                "result_id": int(not shot or i % 40 == 9),
            }
        )
    return pd.DataFrame(rows)


def dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    splits = {"train": [1, 2], "dev": [3], "test": [4]}
    write_json(root / "splits.json", {"splits": splits})
    outputs = {}
    for split, ids in splits.items():
        path = root / f"{split}.parquet"
        pd.concat([actions(game) for game in ids], ignore_index=True).to_parquet(path, index=False)
        outputs[split] = {"path": path.name, "sha256": sha256(path), "game_ids": ids}
    write_json(
        root / "manifest.json",
        {
            "orientation": "attacking_left_to_right",
            "splits": splits,
            "split_sha256": sha256(root / "splits.json"),
            "outputs": outputs,
            "repository": "synthetic-test",
            "configuration": {"commit": "0" * 40},
        },
    )
    return root


def test_state_features_exclude_future_and_labels_stop_at_period_boundary():
    pytest.importorskip("socceraction")
    sample = actions(1)
    original_x, original_y = state_tables(sample)
    changed = sample.copy()
    changed.loc[80:, "type_id"] = 11
    changed.loc[80:, "result_id"] = 1
    changed_x, changed_y = state_tables(changed)
    pd.testing.assert_frame_equal(original_x.iloc[:80], changed_x.iloc[:80])
    pd.testing.assert_frame_equal(original_y.iloc[:80], changed_y.iloc[:80])
    assert not {"game_id", "player_id", "team_id", "scores", "concedes"} & set(original_x)


def test_probabilities_have_explicit_baseline_metrics_and_validation():
    pytest.importorskip("sklearn")
    result = probability_metrics([0, 1], [0.25, 0.75])
    assert result["brier"] == 0.0625
    with pytest.raises(ValueError, match="within"):
        probability_metrics([0, 1], [-1, 1])


def test_training_freezes_model_and_emits_disjoint_values(tmp_path):
    pytest.importorskip("socceraction")
    root = dataset(tmp_path)
    result = train_baselines(root, tmp_path / "trained")
    assert result["game_counts"] == {"train": 2, "dev": 1, "test": 1}
    assert result["vaep_model"]["training_game_ids"] == [1, 2]
    values = pd.read_parquet(tmp_path / "trained/test-vaep-values.parquet")
    assert set(values.game_id) == {4}
    assert np.isfinite(values.vaep_value).all()
    replayed = predict_probabilities(
        tmp_path / "trained", actions(4), orientation="attacking_left_to_right"
    )
    original = pd.read_parquet(tmp_path / "trained/test-probabilities.parquet")
    np.testing.assert_allclose(replayed[["scores", "concedes"]], original[["scores", "concedes"]])
    with pytest.raises(ValueError, match="training or development"):
        predict_probabilities(
            tmp_path / "trained", actions(3), orientation="attacking_left_to_right"
        )
    assert result["test_metrics"]["scores"]["train_prevalence_baseline"]["actions"] == 160
    with pytest.raises(ValueError, match="already exists"):
        train_baselines(root, tmp_path / "trained")


def test_dataset_integrity_rejects_changed_actions_and_overlapping_splits(tmp_path):
    pytest.importorskip("socceraction")
    root = dataset(tmp_path)
    with (root / "test.parquet").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="content hash"):
        load_splits(root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["splits"]["test"] = [1]
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="predeclared split"):
        load_splits(root)
