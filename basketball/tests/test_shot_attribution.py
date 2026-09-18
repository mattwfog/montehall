"""Contract tests for the shot_attribution stage (structural fix A/B)."""

from __future__ import annotations

import numpy as np

from montehall_cv.pipeline.shot_attribution import (
    OCR_ACCEPT_PROB,
    SHOT_ATTRIBUTION_SCHEMA,
    CLUSTER_COLOR_MIN_VOTES,
    SEGMENT_SLACK_MS,
    _learned_score,
    _match_spine_shot,
    _ocr_number_for,
    _team_alternate,
    cluster_colors,
    end_team_map,
    person_at,
    person_index,
    person_pool,
    attribution_picks,
    ball_proximity,
    candidate_features,
    rank_candidates,
    window_candidates,
)
from montehall_cv.store.artifacts import ArtifactWriter


def _rows(boxes: list[tuple[int, tuple, int]]) -> list[tuple]:
    """[(height, frame_idx, bbox, ts_ms)] from (frame, bbox, ts)."""
    return [(bbox[3] - bbox[1], fi, bbox, ts) for fi, bbox, ts in boxes]


def test_heuristic_rank_team_mismatch_last_launch_first():
    feats = {
        1: {"release_prox_px": 5.0, "whole_prox_px": 5.0,
            "launch_dist_ft": None, "team_match": True,
            "n_obs": 10, "max_h_px": 100.0},
        2: {"release_prox_px": 500.0, "whole_prox_px": 500.0,
            "launch_dist_ft": 2.0, "team_match": True,
            "n_obs": 10, "max_h_px": 100.0},
        3: {"release_prox_px": 0.0, "whole_prox_px": 0.0,
            "launch_dist_ft": 1.0, "team_match": False,
            "n_obs": 10, "max_h_px": 100.0},
    }
    order = rank_candidates(feats, model=None)
    # launch-fit candidate outranks release-only; team mismatch dead last
    # even with the best geometry
    assert order == [2, 1, 3]


def test_learned_rank_overrides_heuristic():
    feats = {
        1: {"release_prox_px": 5.0, "whole_prox_px": 5.0,
            "launch_dist_ft": None, "team_match": None,
            "n_obs": 10, "max_h_px": 100.0},
        2: {"release_prox_px": 500.0, "whole_prox_px": 500.0,
            "launch_dist_ft": None, "team_match": None,
            "n_obs": 40, "max_h_px": 100.0},
    }
    model = {"weights": {"n_obs": 1.0}, "bias": 0.0,
             "mean": {"n_obs": 0.0}, "std": {"n_obs": 1.0}, "impute": {}}
    assert rank_candidates(feats, model) == [2, 1]
    assert _learned_score(feats[2], model) > _learned_score(feats[1], model)


def test_learned_score_imputes_missing_and_inf():
    model = {"weights": {"release_prox_px": -1.0}, "bias": 0.0,
             "mean": {"release_prox_px": 0.0},
             "std": {"release_prox_px": 1.0},
             "impute": {"release_prox_px": 3000.0}}
    near = _learned_score({"release_prox_px": 1.0}, model)
    far = _learned_score({"release_prox_px": float("inf")}, model)
    missing = _learned_score({"release_prox_px": None}, model)
    assert near > far
    assert far == missing  # inf and absent both take the impute value


def test_candidate_features_release_window_and_proximity():
    a_ms = 10_000
    # candidate 1: near the ball only INSIDE the release window
    cands = {
        1: _rows([(100, (0, 0, 10, 100), a_ms - 2000),
                     (200, (0, 0, 10, 100), a_ms + 500)]),
        2: _rows([(100, (500, 0, 510, 100), a_ms - 2000)]),
    }
    balls = {100: [(5.0, 50.0)], 200: [(600.0, 50.0)]}
    feats = candidate_features(cands, balls, a_ms, fit=None, det=None)
    assert feats[1]["release_prox_px"] == 0.0       # inside bbox
    assert feats[2]["release_prox_px"] > 400.0
    assert feats[1]["n_obs"] == 2
    assert ball_proximity(cands[2], balls) < float("inf")


def test_window_candidates_carries_ts():
    b = {
        "frame_idx": np.array([1, 2]),
        "ts_ms": np.array([1000, 5000]),
        "track_id": np.array([7, 7]),
        "x1": np.array([0.0, 0.0]), "y1": np.array([0.0, 0.0]),
        "x2": np.array([10.0, 10.0]), "y2": np.array([90.0, 90.0]),
    }
    out = window_candidates(b, 0, 2000)
    assert list(out) == [7]
    (_h, frame_idx, _bbox, ts_ms), = out[7]
    assert (frame_idx, ts_ms) == (1, 1000)


def test_ocr_bridge_names_decisive_posterior_only():
    # spine v3, 2nd identity channel: quark rows IoU-bridge to ByteTrack
    # ids whose PARSeq posteriors pool weighted by match count
    rows = _rows([(10, (100.0, 100.0, 140.0, 260.0), 1000),
                  (11, (102.0, 100.0, 142.0, 260.0), 1033)])
    loc_idx = {
        10: [(77, (101.0, 100.0, 141.0, 260.0))],   # IoU ~.95 with row 1
        11: [(77, (103.0, 100.0, 143.0, 260.0))],
    }
    decisive = {77: {"21": 0.9, "12": 0.1}}
    assert _ocr_number_for(rows, loc_idx, decisive, set()) == ("21", 0.9)
    # sub-bar posterior: no name (never a guess)
    weak = {77: {"21": OCR_ACCEPT_PROB - 0.1, "12": 0.4}}
    assert _ocr_number_for(rows, loc_idx, weak, set()) is None
    # roster filter kills an off-roster number
    assert _ocr_number_for(rows, loc_idx, decisive, {"12"}) is None
    # no spatial overlap: no bridge at all
    far = {10: [(77, (900.0, 900.0, 940.0, 1060.0))]}
    assert _ocr_number_for(rows, far, decisive, set()) is None


def test_person_lookup_slack_and_pool_roster_constraint():
    # relink segments: body 5 belongs to person 1 [0,1000], person 2
    # [4000,5000]. Lookup inside a span, in slack range, and beyond.
    pidx = person_index([[5, 0, 1000, 1], [5, 4000, 5000, 2]])
    assert person_at(pidx, 5, 500) == 1
    assert person_at(pidx, 5, 1500) == 1          # in slack, span 2 too far
    assert person_at(pidx, 5, 2900) == 2          # nearer to person 2's span
    assert SEGMENT_SLACK_MS >= 1000               # the slack the above uses
    assert person_at(pidx, 9, 500) is None        # unknown body
    # pooling: off-roster numbers dropped, share = winner mass / total
    post = person_pool(
        sheet_rows=[(1, "23", 0.9), (1, "23", 0.9), (1, "5", 0.6)],
        contact_rows=[(1, "13", 0.9)],            # off-roster: dropped
        ocr_rows=[],
        roster={"23", "5"})
    num, share = post[1]
    assert num == "23" and abs(share - 1.8 / 2.4) < 1e-6
    # cluster->color map needs CLUSTER_COLOR_MIN_VOTES agreeing pairs
    pairs = [(0, "white")] * CLUSTER_COLOR_MIN_VOTES + [(1, "dark")] * 3
    cc = cluster_colors(pairs)
    assert cc == {0: "white"}


def test_end_team_map_offense_direction_prior():
    # 4 right-rim events vote dark with one white outlier; 3 left-rim
    # events vote white -> each end maps despite per-event noise
    votes = ([("right", "dark")] * 4 + [("right", "white")]
             + [("left", "white")] * 3)
    assert end_team_map(votes) == {"right": "dark", "left": "white"}
    # under quorum: no mapping for that end
    assert end_team_map([("right", "dark")] * 2) == {}
    # both ends resolving to ONE color contradicts the structure -> {}
    both = [("right", "dark")] * 3 + [("left", "dark")] * 3
    assert end_team_map(both) == {}


def test_spine_match_is_span_keyed_not_anchor_keyed():
    # the HS-clip w14 class: window [124866, 130500], spine shot at 130400 —
    # inside the span but 5.5s from the ts_start anchor. Anchor-keyed ±2s
    # never engaged; span-keyed must.
    spine = {130400: 676}
    feats = {676: {}}
    assert _match_spine_shot(spine, 124866, 130500, feats) == (130400, 676)
    # scramble tie-break: two in-span shots -> the LATEST claims the window
    spine = {127333: 682, 130400: 676}
    feats = {682: {}, 676: {}}
    assert _match_spine_shot(spine, 124866, 130500, feats) == (130400, 676)
    # outside span + pad: no claim
    assert _match_spine_shot({120000: 5}, 124866, 130500, {5: {}}) == (None,
                                                                       None)
    # within pad of the span edge: claims
    assert _match_spine_shot({123500: 5}, 124866, 130500, {5: {}}) == (123500,
                                                                       5)
    # spine body absent from the candidate features: never matched
    assert _match_spine_shot({130400: 676}, 124866, 130500, {9: {}}) == (None,
                                                                        None)


def test_team_alternate_repoints_to_shooting_team_contact():
    # ev-9 class: spine bound the white defender; the latest dark contact
    # in the same flight window is the body it should have bound
    arrival = 10_000
    contacts = [
        {"track_id": 3, "ts_ms": 8_000, "ts_end_ms": 8_600},   # dark, earlier
        {"track_id": 5, "ts_ms": 8_800, "ts_end_ms": 9_100},   # dark, latest
        {"track_id": 9, "ts_ms": 9_000, "ts_end_ms": 9_400},   # white (spine's)
        {"track_id": 6, "ts_ms": 9_800, "ts_end_ms": 9_900},   # dark, too late
                                                               # (sub-MIN_FLIGHT)
    ]
    feats = {3: {}, 5: {}, 9: {}, 6: {}}
    color_of = {3: "dark", 5: "dark", 9: "white", 6: "dark"}
    assert _team_alternate(contacts, arrival, feats, color_of, "dark") == 5
    # no candidate of the shooting team in the flight window -> None
    assert _team_alternate(contacts, arrival, feats,
                           {3: "white", 5: "white", 9: "white", 6: "white"},
                           "dark") is None


def test_attribution_picks_roundtrip(tmp_path):
    stage = tmp_path / "shot_attribution"
    writer = ArtifactWriter(stage, SHOT_ATTRIBUTION_SCHEMA)
    base = dict(job_id="j", ts_anchor_ms=0, team=None, shooting_team=None,
                person_id=None,
                release_prox_px=None, whole_prox_px=None, launch_dist_ft=None,
                team_match=None, n_obs=1, max_h_px=10.0, method="release")
    writer.add({**base, "event_id": 0, "track_id": 1, "rank": 0,
                "person_id": 7, "picked": True, "number": "23",
                "read_confidence": 0.9})
    writer.add({**base, "event_id": 0, "track_id": 2, "rank": 1,
                "picked": False, "number": "4", "read_confidence": 0.8})
    writer.add({**base, "event_id": 1, "track_id": 3, "rank": 0,
                "person_id": 9, "picked": True, "number": None,
                "read_confidence": None})
    writer.close()
    picks = attribution_picks(tmp_path)
    # person-only era: every picked row is a person claim; number is a
    # HINT that may be absent (event 1)
    assert sorted(picks) == [0, 1]
    assert picks[0]["person_id"] == 7
    assert picks[0]["number_hint"] == "23"
    assert abs(picks[0]["hint_conf"] - 0.9) < 1e-6
    assert picks[1]["person_id"] == 9
    assert picks[1]["number_hint"] is None
