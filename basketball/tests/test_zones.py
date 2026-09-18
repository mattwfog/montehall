"""Zone geometry: NCAA constants, the corner-3 correction, attacked-end
derivation from shot evidence."""

from __future__ import annotations

import pytest

from montehall_cv.zones import (
    CORNER_LINE_FROM_SIDELINE_FT,
    RIM_X_FT,
    THREE_PT_RADIUS_FT,
    derive_attacked_ends,
    is_three,
    rim_distance_ft,
    zone_of,
)


class TestIsThree:
    def test_beyond_arc_is_three(self) -> None:
        # straight-away, 24 ft from the left rim
        assert is_three(5.25 + 24.0, 25.0, "left")

    def test_inside_arc_is_two(self) -> None:
        assert not is_three(5.25 + 18.0, 25.0, "left")

    def test_corner_inside_radius_is_still_three(self) -> None:
        # The pure-radius test scores this a 2: corner shooter at y=3.0
        # (inside the 3.344 ft strip), rim distance 22.07 < 22.146.
        x, y = 7.0, 3.0
        assert rim_distance_ft(x, y, "left") < THREE_PT_RADIUS_FT
        assert is_three(x, y, "left")

    def test_corner_strip_boundary(self) -> None:
        inside_line_y = CORNER_LINE_FROM_SIDELINE_FT + 0.5  # 2pt side
        assert not is_three(6.0, inside_line_y, "left")

    def test_right_end_mirrors(self) -> None:
        assert is_three(RIM_X_FT["right"] - 24.0, 25.0, "right")
        assert not is_three(RIM_X_FT["right"] - 10.0, 25.0, "right")


class TestZoneOf:
    def test_restricted_area(self) -> None:
        assert zone_of(6.0, 25.0, "left") == "restricted_area"

    def test_paint(self) -> None:
        assert zone_of(15.0, 25.0, "left") == "paint"

    def test_paint_respects_lane_width(self) -> None:
        assert zone_of(15.0, 33.0, "left") == "midrange"

    def test_midrange(self) -> None:
        assert zone_of(5.25 + 18.0, 25.0, "left") == "midrange"

    def test_above_break_three(self) -> None:
        assert zone_of(5.25 + 24.0, 25.0, "left") == "above_break3"

    def test_corner_three(self) -> None:
        assert zone_of(7.0, 2.0, "left") == "corner3"

    def test_backcourt(self) -> None:
        assert zone_of(60.0, 25.0, "left") == "backcourt"
        assert zone_of(30.0, 25.0, "right") == "backcourt"

    def test_out_of_bounds(self) -> None:
        assert zone_of(-5.0, 25.0, "left") == "out_of_bounds"
        assert zone_of(47.0, 60.0, "left") == "out_of_bounds"

    def test_right_end_paint_mirrors(self) -> None:
        assert zone_of(94.0 - 15.0, 25.0, "right") == "paint"

    def test_rejects_bad_end(self) -> None:
        with pytest.raises(ValueError, match="left|right"):
            zone_of(10.0, 25.0, "up")


class TestDeriveAttackedEnds:
    def test_majority_binds_each_team(self) -> None:
        ends = derive_attacked_ends(
            {0: ["left", "left", "right"], 1: ["right", "right"]}
        )
        assert ends == {0: "left", 1: "right"}

    def test_unlocated_shots_ignored(self) -> None:
        ends = derive_attacked_ends({0: ["left", None, "left"], 1: []})
        assert ends[0] == "left"

    def test_silent_team_gets_opposite_end(self) -> None:
        ends = derive_attacked_ends({0: ["left", "left"], 1: []})
        assert ends == {0: "left", 1: "right"}

    def test_contradictory_same_end_drops_weaker(self) -> None:
        ends = derive_attacked_ends(
            {0: ["left", "left", "left"], 1: ["left"]}
        )
        assert ends[0] == "left"
        assert ends[1] is None

    def test_no_evidence_no_binding(self) -> None:
        assert derive_attacked_ends({0: [], 1: []}) == {0: None, 1: None}
