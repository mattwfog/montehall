import json

import pandas as pd
import pytest

from soccerviz.providers.skillcorner import import_match, load_tracking, run, split_manifest


def fixture_files(tmp_path):
    meta = {
        "id": 3,
        "pitch_length": 104,
        "pitch_width": 68,
        "home_team_side": ["right_to_left", "left_to_right"],
        "players": [
            {
                "id": 11,
                "team_id": 1,
                "trackable_object": 111,
                "playing_time": {"by_period": [{"start_frame": 10, "end_frame": 20}]},
            },
            {
                "id": 12,
                "team_id": 2,
                "trackable_object": 112,
                "playing_time": {"by_period": [{"start_frame": 10, "end_frame": 20}]},
            },
            {"id": 13, "team_id": 1, "playing_time": {"by_period": []}},
        ],
        "ball": {"trackable_object": 55},
    }
    frames = [
        {"frame": 0, "period": None, "timestamp": None, "player_data": [], "ball_data": {}},
        {
            "frame": 10,
            "period": 1,
            "timestamp": "00:00:00.0",
            "player_data": [
                {"player_id": 11, "x": 23.0, "y": -4.0, "is_detected": True},
                {"player_id": 12, "x": -11.0, "y": 2.0, "is_detected": False},
            ],
            "ball_data": {"x": 0.2, "y": 1.0, "z": 0.4, "is_detected": None},
        },
        {
            "frame": 11,
            "period": 1,
            "timestamp": "00:00:00.1",
            "player_data": [],
            "ball_data": {"x": None, "y": None, "is_detected": False},
        },
    ]
    matches = [{"id": i, "date_time": f"2025-01-{i:02d}T12:00:00Z"} for i in range(1, 11)]
    paths = {
        k: tmp_path / (k + ext)
        for k, ext in [
            ("metadata", ".json"),
            ("tracking", ".jsonl"),
            ("events", ".csv"),
            ("phases", ".csv"),
            ("matches", ".json"),
        ]
    }
    paths["metadata"].write_text(json.dumps(meta))
    paths["tracking"].write_text("".join(json.dumps(f) + "\n" for f in frames))
    paths["matches"].write_text(json.dumps(matches))
    paths["events"].write_text(
        "event_id,match_id,frame_start,frame_end,period,duration,event_type,event_subtype,x_start\n6_0,3,10,11,1,0.1,off_ball_run,behind,-23\n7_0,3,100,110,1,1,passing_option,,\n"
    )
    paths["phases"].write_text(
        "index,match_id,frame_start,frame_end,period,duration,team_in_possession_phase_type\n0,3,10,11,1,0.1,build_up\n"
    )
    return meta, frames, paths


def test_source_clocks_axes_flags_and_missingness_are_preserved(tmp_path):
    meta, _, paths = fixture_files(tmp_path)
    obs, frames = load_tracking(meta, paths["tracking"])
    assert pd.isna(frames.iloc[0].timestamp_s)
    assert frames.iloc[1].timestamp_s == 0
    assert frames.iloc[1].source_frame_time_s == 1
    detected = obs[(obs.frame_id == 10) & (obs.provider_entity_id == "11")].iloc[0]
    assert (detected.x_m, detected.y_m, detected.status) == (23, -4, "detected")
    assert (
        obs[(obs.frame_id == 10) & (obs.provider_entity_id == "12")].iloc[0].status
        == "extrapolated"
    )
    assert (
        obs[(obs.frame_id == 10) & (obs.entity == "ball")].iloc[0].status
        == "unknown_detection_status"
    )
    absent = obs[(obs.frame_id == 11) & (obs.provider_entity_id == "11")].iloc[0]
    assert absent.status == "unavailable" and not absent.present_in_source
    assert "13" not in set(
        obs.provider_entity_id
    )  # An unused substitute is not a missing detection.


def test_real_export_contract_scope_and_descriptive_summaries(tmp_path):
    _, _, paths = fixture_files(tmp_path)
    report = import_match(**paths, out=tmp_path / "out")
    assert report["human_ground_truth"] is False
    assert report["pitch_length_m"] == 104
    assert report["events"] == 2 and report["events_with_complete_tracking_interval"] == 1
    assert report["descriptive_summaries"]["off_ball_run_subtype_counts"] == {"behind": 1}
    events = pd.read_parquet(tmp_path / "out/events.parquet")
    assert events.event_uid.tolist() == ["3:event:6_0", "3:event:7_0"]
    assert events.x_start.iloc[0] == "-23"  # Never mixed with tracking's +23 metre coordinate.
    assert events.tracking_interval_fully_available.tolist() == [True, False]
    assert (tmp_path / "out/split-manifest.json").exists()
    with pytest.raises(ValueError, match="already exists"):
        import_match(**paths, out=tmp_path / "out")


def test_splits_are_chronological_and_game_disjoint():
    matches = [{"id": i, "date_time": f"2025-01-{i:02d}"} for i in range(1, 11)]
    first = split_manifest(matches)
    assert first == split_manifest(list(reversed(matches)))
    assert [m["split"] for m in first["matches"]] == ["train"] * 6 + ["validation"] * 2 + [
        "test"
    ] * 2
    assert not first["official_split"]


@pytest.mark.parametrize(
    "corruption", ["duplicate_frame", "unknown_player", "invalid_detection_flag"]
)
def test_invalid_tracking_is_rejected(tmp_path, corruption):
    meta, frames, paths = fixture_files(tmp_path)
    if corruption == "duplicate_frame":
        frames[-1]["frame"] = 10
    elif corruption == "unknown_player":
        frames[1]["player_data"][0]["player_id"] = 999
    else:
        frames[1]["player_data"][0]["is_detected"] = "true"
    paths["tracking"].write_text("".join(json.dumps(f) + "\n" for f in frames))
    with pytest.raises(ValueError):
        load_tracking(meta, paths["tracking"])


def test_source_manifest_tamper_rejected(tmp_path):
    _, _, paths = fixture_files(tmp_path)
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps({"sources": {k: {"sha256": "bad"} for k in paths}}))
    with pytest.raises(ValueError, match="hash mismatch"):
        import_match(**paths, out=tmp_path / "out", source_manifest=manifest)
    assert not (tmp_path / "out").exists()


def test_cross_match_events_rejected(tmp_path):
    _, _, paths = fixture_files(tmp_path)
    paths["events"].write_text(paths["events"].read_text().replace("6_0,3,", "6_0,99,"))
    with pytest.raises(ValueError, match="match IDs"):
        import_match(**paths, out=tmp_path / "out")


def test_unknown_worker_operation_rejected():
    with pytest.raises(ValueError, match="Unknown SkillCorner"):
        run({"operation": "unknown"})


def test_download_is_bounded_and_cached_bytes_are_verified(tmp_path, monkeypatch):
    import io

    from soccerviz.providers.skillcorner import _fetch

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: io.BytesIO(b'{"frame":0}\n{"frame":1}\n')
    )
    path = tmp_path / "prefix.jsonl"
    info = _fetch("https://example.test/pinned", path, max_bytes=100, max_lines=1)
    assert path.read_bytes() == b'{"frame":0}\n'
    assert not info["complete_source"] and info["complete_json_lines"] == 1
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="provenance mismatch"):
        _fetch("https://example.test/pinned", path, max_bytes=100, max_lines=1)


def test_oversized_source_never_publishes_partial_download(tmp_path, monkeypatch):
    import io

    from soccerviz.providers.skillcorner import _fetch

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"x" * 100))
    path = tmp_path / "bounded.csv"
    with pytest.raises(ValueError, match="download bound"):
        _fetch("https://example.test/pinned", path, max_bytes=50)
    assert not path.exists() and not list(tmp_path.glob("*.partial"))
