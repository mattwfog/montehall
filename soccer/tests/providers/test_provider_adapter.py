from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soccerviz.providers.provider import load_metrica


def write_csv(path, team, number, offset=0):
    path.write_text(
        f",,,{team},,,\n,,,1,,,\nPeriod,Frame,Time [s],Player{number},,Ball,\n"
        f"1,1,{9.04 + offset},0.1,0.2,0.4,0.6\n"
        f"1,2,{9.09 + offset},NaN,NaN,NaN,NaN\n"
        f"1,3,{9.15 + offset},0.2,0.3,0.5,0.7\n"
    )


def test_kloppy_preserves_raw_clock_and_missing_positions(tmp_path):
    pytest.importorskip("kloppy")
    home, away = tmp_path / "home.csv", tmp_path / "away.csv"
    write_csv(home, "Home", 1)
    write_csv(away, "Away", 2)
    table, report = load_metrica(home, away, sample_rate=1, limit=3)
    assert table.frame_id.nunique() == 3
    np.testing.assert_allclose(table.groupby("frame_id").timestamp_s.first(), [9.04, 9.09, 9.15])
    assert table[table.frame_id == 2].status.eq("unavailable").all()
    assert set(table.provider_entity_id) == {"home_1", "away_1", "ball"}
    first = table[(table.frame_id == 1) & (table.provider_entity_id == "home_1")].iloc[0]
    assert first.x_m == pytest.approx(10.5)
    assert first.y_m == pytest.approx(13.6)
    assert report["missing_observations"] == 3
    write_csv(away, "Away", 2, offset=0.01)
    with pytest.raises(ValueError, match="clocks disagree"):
        load_metrica(home, away)


def test_local_metrica_first_frame_matches_existing_adapter():
    pytest.importorskip("kloppy")
    raw = Path(__file__).resolve().parents[2] / "data/raw/game_1"
    home = raw / "Sample_Game_1_RawTrackingData_Home_Team.csv"
    if not home.exists():
        pytest.skip("Local Metrica sample absent")
    table, report = load_metrica(home, raw / "Sample_Game_1_RawTrackingData_Away_Team.csv", limit=5)
    source = pd.read_csv(home, skiprows=3, header=None).iloc[0]
    first = table[(table.frame_id == 1) & (table.provider_entity_id == "home_11")].iloc[0]
    assert first.timestamp_s == source.iloc[2]
    np.testing.assert_allclose([first.x_m, first.y_m], source.iloc[3:5].to_numpy(float) * [105, 68])
    assert report["registered_players"] == 28
    assert report["frames"] == 5
