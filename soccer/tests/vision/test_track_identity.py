"""Lineup restriction changes the answer; aggregation turns weak frames into a decision; illegible frames do not vote."""

import numpy as np
import pytest

from soccerviz.vision import track_identity as ti


def dist(number=None, p=0.6, illegible=0.1, spread=None):
    """101-way distribution peaked on `number` with the rest spread over `spread` or all numbers."""
    d = np.zeros(ti.NUM_NUMBERS + 1)
    d[ti.ILLEGIBLE] = illegible
    rest = 1.0 - illegible - (p if number is not None else 0.0)
    others = spread if spread is not None else [n for n in range(ti.NUM_NUMBERS) if n != number]
    d[others] += rest / len(others)
    if number is not None:
        d[number] += p
    return d


def test_restrict_to_lineup_keeps_illegible_mass_and_zeroes_outsiders():
    d = dist(7, p=0.5, illegible=0.2)
    r = ti.restrict_to_lineup(d, [7, 10, 23])
    assert r[ti.ILLEGIBLE] == pytest.approx(0.2)
    assert r.sum() == pytest.approx(1.0)
    assert r[5] == 0.0 and r[7] > r[10] == r[23] > 0
    assert np.array_equal(ti.restrict_to_lineup(d, None), d)


def test_lineup_flips_a_reading_the_open_classifier_gets_wrong():
    """The crop looks like an 8 (not on the pitch); the lineup makes it the 3 it is."""
    d = dist(8, p=0.45, illegible=0.1, spread=[3, 6, 9])
    assert ti.aggregate_track([d] * 3)["top_number"] == 8
    result = ti.aggregate_track([d] * 3, lineup=[3, 10, 17])
    assert result["number"] == 3 and result["status"] == "identified"


def test_aggregation_decides_where_single_frames_abstain():
    frames = [dist(19, p=0.35, illegible=0.5) for _ in range(6)]  # each frame alone is weak
    single = ti.aggregate_track(frames[:1], lineup=[10, 13, 17, 19])
    assert single["number"] is None and single["status"] == "insufficient_evidence"
    result = ti.aggregate_track(frames, lineup=[10, 13, 17, 19])
    assert result["number"] == 19 and result["evidence"] == pytest.approx(6 * 0.5**2)


def test_illegible_frames_do_not_vote():
    legible = [dist(13, p=0.7, illegible=0.05)] * 2
    noise = [dist(17, p=0.05, illegible=0.94)] * 40  # forty unreadable frames leaning elsewhere
    result = ti.aggregate_track(legible + noise, lineup=[13, 17])
    assert result["number"] == 13


def test_conflicting_evidence_stays_undecided():
    frames = [dist(13, p=0.7, illegible=0.05)] * 3 + [dist(17, p=0.7, illegible=0.05)] * 3
    result = ti.aggregate_track(frames, lineup=[13, 17])
    assert result["number"] is None and result["status"] == "undecided"


def test_identify_tracks_maps_lineups_by_key():
    frames = {("s", 1): [dist(7, p=0.8, illegible=0.05)] * 3}
    out = ti.identify_tracks(frames, {("s", 1): [7, 8]})
    assert out[("s", 1)]["number"] == 7


def test_a_few_certain_frames_cannot_outrun_many_moderate_ones():
    """Two frames at probability 1.0 on 17 versus six frames at 0.8 on 13: the six win."""
    certain = [dist(17, p=0.95, illegible=0.02, spread=[13])] * 2
    moderate = [dist(13, p=0.75, illegible=0.05, spread=[17])] * 6
    result = ti.aggregate_track(certain + moderate, lineup=[13, 17])
    assert result["top_number"] == 13
