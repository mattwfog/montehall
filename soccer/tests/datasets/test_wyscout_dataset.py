"""Pinned file table and loader layout of the Wyscout open-data corpus; no network."""

from pathlib import Path

import pytest

from soccerviz.datasets import wyscout_dataset as w


def test_every_pinned_file_has_a_figshare_id_and_md5():
    assert set(w.FILES) >= {
        "events.zip",
        "matches.zip",
        "players.json",
        "teams.json",
        "competitions.json",
    }
    for name, (file_id, md5, size) in w.FILES.items():
        assert file_id > 0 and len(md5) == 32 and int(md5, 16) >= 0 and size > 0, name
        assert w.download_url(file_id).endswith(str(file_id))


def test_loader_layout_covers_seven_competitions():
    names = [name for _, name in w.LOADER_FILES]
    assert names.count("competitions.json") == 1
    assert sum(n.startswith("events_") for n in names) == 7
    assert sum(n.startswith("matches_") for n in names) == 7


def test_loader_root_refuses_a_missing_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        w.loader_root(tmp_path)


def test_downloaded_corpus_matches_the_pinned_hashes():
    source = Path("artifacts/public-data/wyscout/source.json")
    if not source.exists():
        pytest.skip("Wyscout corpus not downloaded")
    import json

    record = json.loads(source.read_text())
    assert record["licence"] == "CC BY 4.0"
    for name, (_, md5, size) in w.FILES.items():
        assert record["files"][name]["md5"] == md5 and record["files"][name]["bytes"] == size, name
