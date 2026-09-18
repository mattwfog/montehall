"""Pinned val split for merged-pool ReID training — the leak guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from montehall_cv.training.train_reid_v2 import pinned_val_split


def _dirs(names: list[str]) -> list[Path]:
    return [Path(f"/data/{n}") for n in names]


def test_pinned_names_go_to_val_everything_else_trains() -> None:
    identities = _dirs(["a", "b", "ours_1", "c", "ours_2"])
    val, train = pinned_val_split(identities, {"a", "c"})
    assert [d.name for d in val] == ["a", "c"]
    assert [d.name for d in train] == ["b", "ours_1", "ours_2"]


def test_missing_pinned_id_is_an_error_not_a_silent_shrink() -> None:
    with pytest.raises(ValueError, match="absent from dataset"):
        pinned_val_split(_dirs(["a", "b"]), {"a", "ghost"})


def test_disjoint_by_construction() -> None:
    identities = _dirs([f"id{i}" for i in range(50)])
    pinned = {f"id{i}" for i in range(0, 50, 7)}
    val, train = pinned_val_split(identities, pinned)
    assert {d.name for d in val}.isdisjoint({d.name for d in train})
    assert len(val) + len(train) == 50
