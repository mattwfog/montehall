"""FT classification: the geometric signature and its failure modes."""

from __future__ import annotations

from montehall_cv.pipeline.run_freethrows import classify_window, ft_point

FT_X_LEFT, FT_Y = ft_point("left")  # (19.0, 25.0)


def _shot(court_end: str = "left") -> dict:
    return {"event_id": 0, "ts_start_ms": 10_000, "court_end": court_end}


def _shooter(x: float, y: float, entity: int = 7) -> dict:
    return {"entity_id": entity, "court_x": x, "court_y": y, "team_cluster": 0}


def _stationary_track(x: float, y: float) -> list[tuple[int, float, float]]:
    return [(8_000 + i * 500, x + 0.2 * i, y) for i in range(5)]


def _lane_lineup(court_end: str = "left", n: int = 4) -> list[tuple[float, float]]:
    # alternating lane-edge spots between baseline and FT line
    xs = [7.0, 10.0, 13.0, 16.0, 8.5, 11.5]
    out = []
    for i in range(n):
        x = xs[i] if court_end == "left" else 94.0 - xs[i]
        y = 25.0 + (7.5 if i % 2 == 0 else -7.5)
        out.append((x, y))
    return out


def test_true_free_throw_signature() -> None:
    verdict = classify_window(
        _shot(), _shooter(FT_X_LEFT + 0.5, FT_Y),
        _stationary_track(FT_X_LEFT + 0.5, FT_Y), _lane_lineup(),
    )
    assert verdict["is_ft"] is True
    assert verdict["confidence"] == 1.0
    assert verdict["lane_lineup_count"] >= 3


def test_driving_layup_is_not_a_ft() -> None:
    # shooter ends near the rim after a fast approach across the court
    track = [(8_000 + i * 500, 40.0 - i * 8.0, 25.0) for i in range(5)]
    verdict = classify_window(_shot(), _shooter(6.0, 25.0), track, [])
    assert verdict["is_ft"] is False


def test_jumper_from_the_wing_is_not_a_ft() -> None:
    # stationary catch-and-shoot, but 18ft out on the wing with live play
    verdict = classify_window(
        _shot(), _shooter(15.0, 38.0), _stationary_track(15.0, 38.0), []
    )
    assert verdict["is_ft"] is False


def test_stationary_at_line_without_lineup_stays_field_goal() -> None:
    # elbow jumper: right spot, no rebounding lineup -> not a FT
    verdict = classify_window(
        _shot(), _shooter(FT_X_LEFT, FT_Y), _stationary_track(FT_X_LEFT, FT_Y), []
    )
    assert verdict["is_ft"] is False
    assert verdict["confidence"] < 1.0


def test_right_end_mirrors() -> None:
    fx, fy = ft_point("right")
    assert fx == 75.0
    verdict = classify_window(
        _shot("right"), _shooter(fx, fy),
        _stationary_track(fx, fy), _lane_lineup("right"),
    )
    assert verdict["is_ft"] is True


def test_unattributed_shooter_never_classifies_ft() -> None:
    verdict = classify_window(_shot(), None, [], _lane_lineup())
    assert verdict["is_ft"] is False
    assert verdict["shooter_entity"] is None
