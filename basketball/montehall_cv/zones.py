"""Court zone semantics over the canonical frame (court.py, feet).

The single source for court geometry meaning: rim positions, the 3-point
boundary (arc + the corner straight lines the pure-radius test misses),
NCAA zone taxonomy, and attacked-end derivation from shot evidence.

Zone names (attacked-basket relative):
    backcourt | restricted_area | paint | midrange | corner3 | above_break3
`out_of_bounds` covers positions outside the court + margin — projection
noise, bench, or a bad homography sample.

NCAA geometry (men/women share these since 2021-22):
- basket center 5.25 ft from the baseline, on the width midline
- 3-point arc radius 22.146 ft (22' 1.75"), meeting straight lines
  40.125" (3.344 ft) from each sideline in the corners
- lane ("paint") 12 ft wide, baseline to the free-throw line at 19 ft
  (15 ft from the backboard face, which sits 4 ft in)
- restricted-area arc 4 ft from the basket center
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from montehall_cv.court import NCAA_LENGTH_FT, NCAA_WIDTH_FT

RIM_X_FT: dict[str, float] = {"left": 5.25, "right": NCAA_LENGTH_FT - 5.25}
RIM_Y_FT = NCAA_WIDTH_FT / 2  # 25.0

THREE_PT_RADIUS_FT = 22.146
CORNER_LINE_FROM_SIDELINE_FT = 3.344  # 40.125"
RESTRICTED_RADIUS_FT = 4.0
LANE_HALF_WIDTH_FT = 6.0
FT_LINE_FROM_BASELINE_FT = 19.0
# Where the corner straight line meets the arc, as x-distance from the
# baseline: 5.25 + sqrt(r^2 - (25 - 3.344)^2) ~= 9.87 ft. Charting
# convention: "corner 3" extends to a fixed cut a bit past the break.
CORNER3_MAX_X_FT = 14.0

ZONES = (
    "backcourt",
    "restricted_area",
    "paint",
    "midrange",
    "corner3",
    "above_break3",
    "out_of_bounds",
)


def rim_xy(end: str) -> tuple[float, float]:
    """Basket-center court position for an end ("left" | "right")."""
    return (RIM_X_FT[end], RIM_Y_FT)


def rim_distance_ft(x: float, y: float, end: str) -> float:
    rx, ry = rim_xy(end)
    return float(np.hypot(x - rx, y - ry))


def is_three(x: float, y: float, end: str) -> bool:
    """True when (x, y) is behind the 3-point line attacking `end`.

    The line is an arc of 22.146 ft EXCEPT in the corners, where straight
    lines 3.344 ft from each sideline govern — a corner shooter can be a
    legitimate 3 at under 22 ft of rim distance. The strip test alone is
    safe to OR with the radius test: any inside-the-arc position in the
    strip is behind the straight line by construction.
    """
    if rim_distance_ft(x, y, end) > THREE_PT_RADIUS_FT:
        return True
    return abs(y - RIM_Y_FT) > (RIM_Y_FT - CORNER_LINE_FROM_SIDELINE_FT)


def _baseline_relative_x(x: float, end: str) -> float:
    """Distance from the attacked baseline along the length axis."""
    return x if end == "left" else NCAA_LENGTH_FT - x


def zone_of(x: float, y: float, end: str, margin_ft: float = 3.0) -> str:
    """Zone name for a court position attacking `end`."""
    if end not in RIM_X_FT:
        raise ValueError(f"end must be left|right, got {end!r}")
    if not (
        -margin_ft <= x <= NCAA_LENGTH_FT + margin_ft
        and -margin_ft <= y <= NCAA_WIDTH_FT + margin_ft
    ):
        return "out_of_bounds"
    if _baseline_relative_x(x, end) > NCAA_LENGTH_FT / 2:
        return "backcourt"
    if rim_distance_ft(x, y, end) <= RESTRICTED_RADIUS_FT:
        return "restricted_area"
    in_lane_width = abs(y - RIM_Y_FT) <= LANE_HALF_WIDTH_FT
    if in_lane_width and _baseline_relative_x(x, end) <= FT_LINE_FROM_BASELINE_FT:
        return "paint"
    if is_three(x, y, end):
        if _baseline_relative_x(x, end) <= CORNER3_MAX_X_FT:
            return "corner3"
        return "above_break3"
    return "midrange"


def derive_attacked_ends(
    shot_ends_by_team: dict[int, list[str]],
) -> dict[int, str | None]:
    """team_cluster -> attacked end, by majority of its shots' court_end.

    Evidence comes from shot events (the VLM locates each attempt at an
    end); a team with no located shots gets None. When both teams bind to
    the SAME end the evidence is contradictory (a rim mislocation or a
    one-sided clip) — the weaker team's binding is dropped rather than
    inventing an opposite.
    """
    binding: dict[int, str | None] = {}
    counts: dict[int, Counter] = {}
    for team, ends in shot_ends_by_team.items():
        located = [e for e in ends if e in RIM_X_FT]
        counts[team] = Counter(located)
        binding[team] = counts[team].most_common(1)[0][0] if located else None

    bound = {t: e for t, e in binding.items() if e is not None}
    if len(bound) == 2:
        (t_a, end_a), (t_b, end_b) = bound.items()
        if end_a == end_b:
            weaker = min((t_a, t_b), key=lambda t: counts[t][binding[t]])
            binding[weaker] = None
    # One team bound, the other silent -> the silent team attacks the
    # other end (two-team game on one court).
    if len(bound) == 1 and len(binding) == 2:
        (bound_team, bound_end), = bound.items()
        other = next(t for t in binding if t != bound_team)
        binding[other] = "right" if bound_end == "left" else "left"
    return binding
