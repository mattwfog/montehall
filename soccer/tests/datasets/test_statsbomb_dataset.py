import json

import pytest

from soccerviz.datasets.statsbomb_dataset import build_dataset, split_matches


def test_split_is_reproducible_order_independent_and_disjoint():
    matches = [{"match_id": value} for value in range(51)]
    result = split_matches(matches)
    assert result == split_matches(list(reversed(matches)))
    assert {key: len(value) for key, value in result.items()} == {"train": 20, "dev": 5, "test": 5}
    assert len({game for values in result.values() for game in values}) == 30
    with pytest.raises(ValueError, match="unique"):
        split_matches(matches + [matches[0]])
    with pytest.raises(ValueError, match="ten"):
        split_matches(matches, 5)


def test_source_must_be_pinned_before_network_or_conversion(tmp_path):
    pytest.importorskip("socceraction")
    with pytest.raises(ValueError, match="full Git commit"):
        build_dataset(tmp_path / "data", commit="master")
    root = tmp_path / "existing"
    root.mkdir()
    (root / "configuration.json").write_text(json.dumps({"max_matches": 999}))
    with pytest.raises(ValueError, match="configuration differs"):
        build_dataset(root)
