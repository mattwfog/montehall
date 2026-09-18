import numpy as np

from soccerviz.modeling.forecast import displacement_metrics, windows


def test_windows_do_not_cross_periods_or_fill_missing_players(tmp_path):
    t = np.arange(100) / 5
    xy = np.zeros((100, 16, 2), dtype=np.float32)
    xy[:, :, 0] = t[:, None]
    xy[:, :, 1] = np.arange(16)[None, :]
    xy[:, 0] = np.nan
    path = tmp_path / "tracking.npz"
    np.savez(
        path,
        xy=xy,
        ball=np.full((100, 2), np.nan),
        teams=np.array([0] * 8 + [1] * 8),
        periods=np.where(t < 10, 1, 2),
        times=t,
    )
    (inputs, labels, velocity), origins = windows(path)
    assert np.isfinite(inputs).all()
    assert (origins[:, 1] != 0).all()
    assert not ((origins[:, 0] >= 35) & (origins[:, 0] < 60)).any()
    assert np.all(inputs[:, :, 6] == 0)  # Ball observation mask.
    assert np.allclose(velocity[:, 0], 1)
    assert np.allclose(labels[:, -1, 0], 3)
    assert displacement_metrics(labels, labels) == {"ade_m": 0.0, "fde_3s_m": 0.0}
