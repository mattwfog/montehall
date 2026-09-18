"""Quark engine contract tests (pipeline/quark.py).

Each test drives synthetic detection streams through the public API and
asserts the property the architecture depends on. The severing test is
the anti-ByteTrack test: the one failure mode the quark layer exists to
kill is a segment silently bridging two humans through a pile.
"""

from __future__ import annotations

import numpy as np
import pytest

from montehall_cv.pipeline.quark import run_bidirectional, run_pass

FPS_MS = 33.0
W, H = 40.0, 100.0


def det(cx: float, cy: float, conf: float = 0.9) -> list[float]:
    return [cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2, conf]


Track = list[tuple[float, float] | None]


def frames_from_tracks(tracks: list[Track]):
    """tracks[k][t] = (cx, cy) of body k at frame t, None = not detected."""
    n_frames = len(tracks[0])
    frames = []
    for t in range(n_frames):
        dets = [det(*pos) for trk in tracks if (pos := trk[t]) is not None]
        frames.append((t, t * FPS_MS, np.array(dets, dtype=np.float64)
                       if dets else np.zeros((0, 5))))
    return frames


def linear(x0: float, x1: float, y: float, n: int) -> "Track":
    return [(x0 + (x1 - x0) * t / max(n - 1, 1), y) for t in range(n)]


def by_track(rows: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in rows:
        out.setdefault(r["track_id"], []).append(r)
    return out


def test_exclusive_claims_and_full_span() -> None:
    frames = frames_from_tracks([
        linear(100, 400, 100, 40),
        linear(100, 400, 400, 40),
    ])
    rows, stats = run_bidirectional(frames)
    claimed = [(r["frame_idx"], r["det_idx"]) for r in rows
               if r["det_idx"] >= 0]
    assert len(claimed) == len(set(claimed)), "a detection was double-claimed"
    tracks = by_track(rows)
    assert len(tracks) == 2
    for t_rows in tracks.values():
        assert t_rows[0]["frame_idx"] == 0
        assert t_rows[-1]["frame_idx"] == 39
    assert stats["rows_disputed"] == 0  # clean scene: fwd == bwd


def test_separable_crossing_keeps_identity() -> None:
    # cross in x, 150px apart in y: the wrong detection is always
    # outside the claim gate, so identity must hold through the cross
    frames = frames_from_tracks([
        linear(100, 500, 100, 40),
        linear(500, 100, 250, 40),
    ])
    rows, _stats = run_bidirectional(frames)
    tracks = by_track(rows)
    assert len(tracks) == 2
    for t_rows in tracks.values():
        xs = [(r["x1"] + r["x2"]) / 2 for r in t_rows]
        deltas = np.diff(xs)
        assert (deltas > 0).all() or (deltas < 0).all(), \
            "a segment reversed direction: identity swapped at the cross"


def test_coalescence_severs_both_segments() -> None:
    # converge to overlapping boxes (5px apart), hold, then separate:
    # identity through the pile is unknowable from motion — no output
    # segment may bridge pre-pile to post-pile
    a = (linear(100, 298, 100, 20) + [(298, 100)] * 5
         + linear(298, 500, 100, 20))
    b = (linear(500, 303, 100, 20) + [(303, 100)] * 5
         + linear(303, 100, 100, 20))
    frames = frames_from_tracks([a, b])
    fwd = run_pass(frames)
    assert fwd.n_coalescences >= 1, "coalescence never detected"
    pile_t0, pile_t1 = 20 * FPS_MS, 25 * FPS_MS
    for sid, seg_rows in fwd.segments:
        ts = [r.ts_ms for r in seg_rows]
        assert not (min(ts) < pile_t0 - FPS_MS and max(ts) > pile_t1 + FPS_MS), \
            f"segment {sid} silently bridged the pile"


def test_coast_bridges_gap_with_interpolation() -> None:
    track = linear(100, 500, 100, 40)
    for t in range(18, 28):  # 10 missing frames = 330ms < horizon
        track[t] = None
    frames = frames_from_tracks([track])
    rows, _stats = run_bidirectional(frames)
    tracks = by_track(rows)
    assert len(tracks) == 1, "gap within horizon must not split the segment"
    seg = next(iter(tracks.values()))
    coasted = [r for r in seg if r["coasted"]]
    assert len(coasted) == 10
    # rewritten as interpolation between observed endpoints, not the
    # blind prediction: mid-gap x must sit between the endpoints' x
    mid = coasted[5]
    cx = (mid["x1"] + mid["x2"]) / 2
    x_before = 100 + 400 * 17 / 39
    x_after = 100 + 400 * 28 / 39
    assert x_before < cx < x_after


def test_coast_horizon_ends_segment_honestly() -> None:
    track = linear(100, 200, 100, 20) + [None] * 90 \
        + [(400, 100)] * 20  # 90 frames ≈ 3s > 2s horizon
    frames = frames_from_tracks([track])
    rows, _stats = run_bidirectional(frames)
    tracks = by_track(rows)
    assert len(tracks) == 2, "beyond-horizon gap must split the segment"
    assert not any(r["coasted"] for r in rows), \
        "coast tail past segment death leaked into output"


def test_single_frame_fp_never_confirms() -> None:
    steady = linear(100, 400, 100, 30)
    fp: Track = [None] * 30
    fp[10] = (700, 400)
    frames = frames_from_tracks([steady, fp])
    rows, _stats = run_bidirectional(frames)
    assert len(by_track(rows)) == 1, "a one-frame detection earned a segment"


def test_contested_flag_on_near_ambiguous_claims() -> None:
    # two bodies running parallel 14px apart: below the 0.15-height
    # margin (contested) but below the coalescence IoU (no sever) — the
    # ambiguity must be reported even though identities hold
    frames = frames_from_tracks([
        linear(100, 400, 100, 30),
        linear(114, 414, 100, 30),
    ])
    rows, _stats = run_bidirectional(frames)
    assert len(by_track(rows)) == 2
    assert any(r["contested"] for r in rows)


def test_impossible_switch_vetoed_by_physics() -> None:
    # two bodies cross at the SAME y at speed: at the crossing instant
    # distance alone near-ties, but a swap implies instant velocity
    # reversal for both — the physics-normalized cost must hold each
    # identity straight through (overlap is too brief to arm a sever)
    frames = frames_from_tracks([
        linear(100, 500, 100, 40),
        linear(500, 100, 100, 40),
    ])
    rows, _stats = run_bidirectional(frames)
    tracks = by_track(rows)
    assert len(tracks) == 2, "crossing severed or spawned segments"
    for t_rows in tracks.values():
        xs = [(r["x1"] + r["x2"]) / 2 for r in t_rows if not r["coasted"]]
        deltas = np.diff(xs)
        assert (deltas > 0).all() or (deltas < 0).all(), \
            "identity switched at the cross despite physics"


def test_fast_pan_survives_ego_compensation() -> None:
    # six stationary bodies under a 25px/frame pan ≈ 7.6 h/s screen
    # motion — physically impossible for a body, routine for a camera.
    # The det-cloud camera estimate must absorb it; every segment
    # survives the whole pan un-split
    n, pan = 40, 25.0
    tracks: list[Track] = [
        [(200.0 + k * 120 + pan * t, 300.0) for t in range(n)]
        for k in range(6)
    ]
    rows, _stats = run_bidirectional(frames_from_tracks(tracks))
    tracks_out = by_track(rows)
    assert len(tracks_out) == 6, \
        f"pan shattered segments: {len(tracks_out)} != 6"
    for t_rows in tracks_out.values():
        assert t_rows[0]["frame_idx"] <= 2
        assert t_rows[-1]["frame_idx"] == n - 1


def test_zoom_survives_similarity_warp() -> None:
    # six static bodies under a 4%/frame zoom-in around (640, 360):
    # boxes drift radially up to ~20px/frame and heights rescale —
    # translation-only ego-motion coasts everyone (the 08-05 v2 first
    # run coasted half of Cal); the similarity warp must absorb it
    n, zoom_c = 30, np.array([640.0, 360.0])
    base = [np.array([200.0 + k * 180, 300.0 + (k % 2) * 250])
            for k in range(6)]
    tracks: list[Track] = [
        [tuple(zoom_c + (p - zoom_c) * (1.04 ** t)) for t in range(n)]
        for p in base
    ]
    rows, _stats = run_bidirectional(frames_from_tracks(tracks))
    tracks_out = by_track(rows)
    assert len(tracks_out) == 6, \
        f"zoom shattered segments: {len(tracks_out)} != 6"
    for t_rows in tracks_out.values():
        assert t_rows[-1]["frame_idx"] == n - 1
        assert not any(r["coasted"] for r in t_rows), \
            "zoom pushed claims outside the physics gate (coasting)"


def test_coast_through_body_severs() -> None:
    # A walks left-to-right; its detections vanish exactly while its
    # path passes THROUGH stationary B. The coast bridges the gap
    # physically, but identity across an occlusion BY A BODY is
    # unknowable — the segment must sever at coast start (the v2.x
    # swap channel: claimed-box coalescence never sees coasting
    # seekers). Coasting through EMPTY space must still bridge
    # (test_coast_bridges_gap_with_interpolation above).
    # A crosses behind B at constant 4px/frame — slow enough that the
    # interpolated coast path deeply overlaps B for >=132ms (sustained
    # pass-through, not a single-frame brush), constant enough that the
    # re-anchor lands inside the gate (a body that STOPS behind a
    # screen overshoots its own prediction and dies at horizon instead
    # — honest fragmentation, different mechanism)
    a: Track = (linear(230, 290, 100, 16) + [None] * 10
                + linear(330, 382, 100, 14))
    b: Track = [(300.0, 100.0)] * 40
    frames = frames_from_tracks([a, b])
    eng = run_pass(frames)
    assert eng.n_occlusion_severs >= 1, \
        "sustained coast through another body did not sever"
    assert len(eng.segments) == 3, \
        f"expected A-pre, A-post, B = 3 segments, got {len(eng.segments)}"
    # no MOVING segment may span the occlusion gap (stationary B does,
    # legitimately — it was observed throughout)
    for _sid, seg_rows in eng.segments:
        frames_in = [r.frame_idx for r in seg_rows]
        xs = [(r.x1 + r.x2) / 2 for r in seg_rows]
        spans_gap = min(frames_in) < 16 and max(frames_in) >= 26
        assert not (spans_gap and max(xs) - min(xs) > 50), \
            "a moving segment silently bridged the body occlusion"


def test_limbs_prevent_unnecessary_sever() -> None:
    # the coalescence geometry (converge, hold overlapped 5 frames,
    # separate): WITHOUT pose the boxes merge, margins collapse, the
    # interval arms and both segments sever (test_coalescence above).
    # WITH distinct limb configurations the assignment margins stay
    # high through the overlap — identity was never ambiguous, so the
    # cut must not happen (boxes merge; limbs don't).
    a: Track = (linear(100, 298, 100, 20) + [(298.0, 100.0)] * 5
                + linear(298, 100, 100, 20))
    b: Track = (linear(500, 303, 100, 20) + [(303.0, 100.0)] * 5
                + linear(303, 500, 100, 20))
    frames = frames_from_tracks([a, b])
    pos = {t: [a[t], b[t]] for t in range(45)}

    def kp_for(cx: float, cy: float, flavor: int) -> np.ndarray:
        kp = np.zeros((17, 3))
        sign = 1.0 if flavor == 0 else -1.0
        kp[0] = (cx, cy - 45, 0.9)                 # nose
        kp[9] = (cx - 50 * sign, cy - 30 * sign, 0.9)   # wrists
        kp[10] = (cx + 50 * sign, cy + 30 * sign, 0.9)
        kp[15] = (cx - 20 * sign, cy + 48, 0.9)    # ankles
        kp[16] = (cx + 20 * sign, cy + 48, 0.9)
        return kp

    def pose_lookup(ts_ms: float):
        t = min(max(int(round(ts_ms / FPS_MS)), 0), 44)
        pts = [p for p in pos[t] if p is not None]
        boxes = np.array([[p[0] - W / 2, p[1] - H / 2,
                           p[0] + W / 2, p[1] + H / 2] for p in pts])
        kps = np.stack([kp_for(pts[0][0], pts[0][1], 0),
                        kp_for(pts[1][0], pts[1][1], 1)])
        return boxes, kps

    fwd_plain = run_pass(frames)
    fwd_pose = run_pass(frames, pose_lookup)
    assert fwd_plain.n_coalescences >= 1, \
        "plain run no longer severs — scenario lost its ambiguity"
    assert fwd_pose.n_coalescences == 0, \
        "limb evidence did not prevent the unnecessary sever"
    assert len(fwd_pose.segments) == 2
    # and the identities did not swap: each body returns to its origin
    for _sid, seg_rows in fwd_pose.segments:
        xs = [(r.x1 + r.x2) / 2 for r in seg_rows if not r.coasted]
        assert abs(xs[0] - xs[-1]) < 30, \
            "a segment ended far from its origin: identities swapped"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
