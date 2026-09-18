"""Oversampling contract for train_parseq: repeats expand the train side
only, never the val split — a repeated crop must not appear in both."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from montehall_cv.training.train_parseq import expand_repeats, load_rows


def _write_dataset(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    dataset_dir = tmp_path / name
    dataset_dir.mkdir()
    with open(dataset_dir / "labels.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return dataset_dir


def test_load_rows_tags_repeat_factor_per_dataset(tmp_path: Path) -> None:
    synth = _write_dataset(
        tmp_path, "synth",
        [{"file": "a.jpg", "legible": True, "label": "24"},
         {"file": "b.jpg", "legible": False, "label": None}],
    )
    real = _write_dataset(
        tmp_path, "real", [{"file": "c.jpg", "legible": True, "label": "5"}]
    )
    legible, illegible = load_rows([synth, real], repeats=[1, 100])
    assert [r["_repeat"] for r in legible] == [1, 100]
    assert [r["_repeat"] for r in illegible] == [1]


def test_load_rows_rejects_mismatched_repeat_count(tmp_path: Path) -> None:
    synth = _write_dataset(
        tmp_path, "synth", [{"file": "a.jpg", "legible": True, "label": "24"}]
    )
    with pytest.raises(SystemExit, match="one factor per dataset"):
        load_rows([synth], repeats=[1, 2])


def test_load_rows_rejects_zero_factor(tmp_path: Path) -> None:
    synth = _write_dataset(
        tmp_path, "synth", [{"file": "a.jpg", "legible": True, "label": "24"}]
    )
    with pytest.raises(SystemExit, match=">=1"):
        load_rows([synth], repeats=[0])


def test_expand_repeats_multiplies_rows() -> None:
    rows = [{"path": "a", "_repeat": 1}, {"path": "b", "_repeat": 3}]
    expanded = expand_repeats(rows)
    assert [r["path"] for r in expanded] == ["a", "b", "b", "b"]


def test_expand_repeats_defaults_to_one_without_tag() -> None:
    assert len(expand_repeats([{"path": "a"}])) == 1


def test_repeat_expansion_never_duplicates_into_val() -> None:
    """The train() split contract: val is drawn from unique rows, expansion
    happens after — reproduced here on the same primitives train() uses."""
    legible = [{"path": f"r{i}", "_repeat": 50} for i in range(10)]
    n_val = max(int(len(legible) * 0.1), 1)
    val_rows, train_rows = legible[:n_val], expand_repeats(legible[n_val:])
    val_paths = {r["path"] for r in val_rows}
    assert val_paths.isdisjoint({r["path"] for r in train_rows})
    assert len(train_rows) == 9 * 50
