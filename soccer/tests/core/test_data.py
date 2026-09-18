import numpy as np

from soccerviz.core.data import read_tracking


def test_ingestion_preserves_ids_source_timestamps_and_missing_positions(tmp_path):
    source = tmp_path / "tracking.csv"
    source.write_text(
        ",,,Home,,Home,,,\n"
        ",,,11,,1,,,\n"
        "Period,Frame,Time [s],Player11,,Player1,,Ball,\n"
        "1,1,0.04,0.1,0.2,NaN,NaN,0.5,0.5\n"
        "1,2,0.08,0.2,0.2,NaN,NaN,0.5,0.5\n"
        "1,5,0.20,0.3,0.2,0.6,0.7,NaN,NaN\n"
    )
    table, players, xy = read_tracking(source, 0)
    assert players == ["Home:Player11", "Home:Player1"]
    assert list(table.iloc[:, 2]) == [0.04, 0.20]
    np.testing.assert_allclose(xy[0, 0], [10.5, 13.6])
    assert np.isnan(xy[0, 1]).all()
