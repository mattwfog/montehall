"""Contract tests for the ball spine (ball_track fusion + event derivation)."""

from __future__ import annotations

from montehall_cv.pipeline.ball_track import (
    BALL_EVENTS_SCHEMA,
    COAST_MAX_FRAMES,
    GRAZE_UP_VY,
    HOLD_MIN_FRAMES,
    STITCH_MAX_FRAMES,
    WASB_LOW_MIN,
    GAP_MIN_FRAMES,
    derive_events,
    fuse_track,
    spine_contacts,
    spine_shooters,
    uncovered_spans,
)
from montehall_cv.store.artifacts import ArtifactWriter


def _ts(frames):
    return {f: f * 33 for f in frames}


def test_fuse_bridges_gap_with_wasb_and_coasts():
    # det sees frames 0-9 moving right; det goes blind 10-19; wasb carries
    # the flight; det re-acquires 20-24. One segment, no break.
    det = {f: [(10.0 * f, 100.0, 0.9)] for f in range(10)}
    det.update({f: [(10.0 * f, 100.0, 0.9)] for f in range(20, 25)})
    wasb = {f: (10.0 * f, 100.0, 0.6) for f in range(10, 20)}
    rows = fuse_track(det, wasb, _ts(range(25)))
    assert {r["seg_id"] for r in rows} == {0}
    assert any(r["source"] == "wasb" for r in rows)
    assert all(abs(r["x"] - 10.0 * r["frame_idx"]) < 1e-6
               for r in rows if r["source"] != "interp")


def test_rim_pass_verdict_geometry():
    from montehall_cv.pipeline.ball_track import rim_pass_verdict

    rim = (100.0, 200.0, 140.0, 220.0)  # cx=120, w=40, mouth y1=200 y2=220
    mk = lambda pts: [{"ts_ms": i, "x": x, "y": y}
                      for i, (x, y) in enumerate(pts)]
    # clean make: descends over the mouth, exits below the rim
    assert rim_pass_verdict(
        mk([(118, 150), (120, 190), (121, 210), (122, 240)]), rim) is True
    # bounce-off: over the mouth, never below
    assert rim_pass_verdict(
        mk([(118, 150), (120, 190), (90, 180), (60, 160)]), rim) is False
    # side approach: never over the mouth in-x -> can't judge
    assert rim_pass_verdict(
        mk([(200, 150), (190, 210), (180, 240)]), rim) is None
    # no rim / too few rows -> can't judge
    assert rim_pass_verdict(mk([(120, 150), (120, 240)]), rim) is None
    assert rim_pass_verdict(mk([(118, 150), (120, 190), (121, 240)]),
                            None) is None


def test_uncovered_spans_and_tiled_refuse():
    # the ev13 class: a >= GAP_MIN_FRAMES hole between two confirmed
    # stretches is reported; tile peaks merged into the wasb map let
    # fuse_track cover it on the second pass.
    g = GAP_MIN_FRAMES
    det = {f: [(10.0 * f, 100.0, 0.9)] for f in range(6)}
    det.update({f: [(10.0 * f, 100.0, 0.9)] for f in range(6 + g, 12 + g)})
    rows = fuse_track(det, {}, _ts(range(12 + g)))
    spans = uncovered_spans(rows, rows[0]["frame_idx"],
                            rows[-1]["frame_idx"])
    assert spans == [(6, 5 + g)]
    # sub-GAP hole: not reported
    short = fuse_track({f: [(10.0 * f, 100.0, 0.9)]
                        for f in list(range(6)) + list(range(10, 16))},
                       {}, _ts(range(16)))
    assert uncovered_spans(short, short[0]["frame_idx"],
                           short[-1]["frame_idx"]) == []
    # tiled peaks over the hole -> second fuse covers it (one long
    # confirmed run; the two ends stitch through the supplied frames)
    tiled = {f: [(10.0 * f, 100.0, 0.6)] for f in range(6, 6 + g)}
    refused = fuse_track(det, tiled, _ts(range(12 + g)))
    spans2 = uncovered_spans(refused, refused[0]["frame_idx"],
                             refused[-1]["frame_idx"])
    assert spans2 == []


def test_stitch_bridges_plausible_gap_beyond_coast():
    # gap > coast horizon but within the stitch physics bound: one segment
    # with honest interp rows across the bridge (continuity v2)
    det = {f: [(5.0 * f, 50.0, 0.9)] for f in range(8)}
    far = 8 + COAST_MAX_FRAMES + 5
    det.update({f: [(5.0 * f, 50.0, 0.9)] for f in range(far, far + 8)})
    rows = fuse_track(det, {}, _ts(range(far + 8)))
    assert {r["seg_id"] for r in rows} == {0}
    assert any(r["source"] == "interp" and 8 <= r["frame_idx"] < far
               for r in rows)


def test_fuse_breaks_on_implausible_bridge():
    # far in space AND time: no stitch, two segments
    det = {f: [(5.0 * f, 50.0, 0.9)] for f in range(8)}
    far = 8 + STITCH_MAX_FRAMES + 10
    det.update({f: [(1800.0, 900.0, 0.9)] for f in range(far, far + 8)})
    rows = fuse_track(det, {}, _ts(range(far + 8)))
    assert {r["seg_id"] for r in rows} == {0, 1}


def test_events_hold_release_arrival_shot():
    # ball sits on body 7 for HOLD_MIN_FRAMES+, then flies to the rim
    track = []
    for f in range(HOLD_MIN_FRAMES + 3):
        track.append({"frame_idx": f, "ts_ms": f * 33, "x": 100.0,
                      "y": 500.0, "conf": .9, "source": "det", "seg_id": 0})
    for i, f in enumerate(range(HOLD_MIN_FRAMES + 3,
                                HOLD_MIN_FRAMES + 3 + 20)):
        track.append({"frame_idx": f, "ts_ms": f * 33,
                      "x": 100.0 + 40.0 * (i + 1),
                      "y": 500.0 - 18.0 * (i + 1),
                      "conf": .9, "source": "det", "seg_id": 0})
    bodies = {f: [(7, (80.0, 400.0, 120.0, 600.0))]
              for f in range(HOLD_MIN_FRAMES + 3)}
    rim_frame = HOLD_MIN_FRAMES + 3 + 19
    rims = {f: [(850.0, 120.0, 910.0, 160.0)]
            for f in range(rim_frame - 2, rim_frame + 1)}
    events = derive_events(track, bodies, rims)
    kinds = [e["kind"] for e in events]
    assert "release" in kinds and "rim_arrival" in kinds
    shots = [e for e in events if e["kind"] == "shot"]
    assert len(shots) == 1
    assert shots[0]["track_id"] == 7
    assert shots[0]["release_ts_ms"] < shots[0]["ts_ms"]


def test_catch_and_shoot_binds_last_contact_not_passer():
    # body 7 possesses the ball (long hold), passes; body 9 touches it for
    # only 3 frames (sub-possession) and shoots — the shot binds to 9.
    track = []
    f = 0
    for _ in range(HOLD_MIN_FRAMES + 5):
        track.append({"frame_idx": f, "ts_ms": f * 33, "x": 100.0,
                      "y": 500.0, "conf": .9, "source": "det", "seg_id": 0})
        f += 1
    for _ in range(6):  # pass in flight toward body 9
        track.append({"frame_idx": f, "ts_ms": f * 33, "x": 100.0 + 50.0 * (f - 9),
                      "y": 500.0, "conf": .9, "source": "det", "seg_id": 0})
        f += 1
    catch_frames = []
    for _ in range(3):  # 3-frame catch on body 9 — never a possession
        catch_frames.append(f)
        track.append({"frame_idx": f, "ts_ms": f * 33, "x": 420.0,
                      "y": 500.0, "conf": .9, "source": "det", "seg_id": 0})
        f += 1
    for i in range(18):  # shot flight to the rim
        track.append({"frame_idx": f, "ts_ms": f * 33,
                      "x": 420.0 + 30.0 * (i + 1),
                      "y": 500.0 - 20.0 * (i + 1),
                      "conf": .9, "source": "det", "seg_id": 0})
        f += 1
    bodies = {fr: [(7, (80.0, 400.0, 120.0, 600.0))]
              for fr in range(HOLD_MIN_FRAMES + 5)}
    for fr in catch_frames:
        bodies[fr] = [(9, (400.0, 400.0, 440.0, 600.0))]
    rim_f = f - 1
    rims = {fr: [(950.0, 120.0, 1010.0, 160.0)]
            for fr in range(rim_f - 2, rim_f + 1)}
    events = derive_events(track, bodies, rims)
    shots = [e for e in events if e["kind"] == "shot"]
    assert len(shots) == 1
    assert shots[0]["track_id"] == 9  # the catcher, not passer 7
    # body 7 still owns the possession record
    holds = [e for e in events if e["kind"] == "hold"]
    assert [h["track_id"] for h in holds] == [7]


def test_dribble_micro_holds_merge_into_one_possession():
    track = []
    f = 0
    for burst in range(3):  # 3 micro-holds by body 5, 100ms gaps
        for _ in range(HOLD_MIN_FRAMES + 1):
            track.append({"frame_idx": f, "ts_ms": f * 33, "x": 100.0,
                          "y": 500.0, "conf": .9, "source": "det",
                          "seg_id": 0})
            f += 1
        for _ in range(3):  # ball briefly off-body (dribble)
            track.append({"frame_idx": f, "ts_ms": f * 33, "x": 100.0,
                          "y": 640.0, "conf": .9, "source": "det",
                          "seg_id": 0})
            f += 1
    bodies = {r["frame_idx"]: [(5, (80.0, 400.0, 120.0, 600.0))]
              for r in track if r["y"] < 600}
    events = derive_events(track, bodies, {})
    holds = [e for e in events if e["kind"] == "hold"]
    assert len(holds) == 1  # merged, not three


def test_fuse_multipeak_second_peak_carries_track():
    # spine v3: top peak each blind frame is a static distractor far away;
    # the real ball is the SECOND peak, on the track's prediction. v2's
    # argmax-only supply lost the track here (the 114-131s hole class).
    det = {f: [(10.0 * f, 100.0, 0.9)] for f in range(10)}
    det.update({f: [(10.0 * f, 100.0, 0.9)] for f in range(20, 25)})
    wasb = {f: [(1800.0, 950.0, 0.9), (10.0 * f, 100.0, 0.5)]
            for f in range(10, 20)}
    rows = fuse_track(det, wasb, _ts(range(25)))
    assert {r["seg_id"] for r in rows} == {0}
    assert all(abs(r["x"] - 10.0 * r["frame_idx"]) < 1e-6
               for r in rows if r["source"] != "interp")


def test_fuse_single_peak_tuple_still_accepted():
    # v2 caches and older callers hand a bare tuple per frame
    det = {f: [(10.0 * f, 100.0, 0.9)] for f in range(10)}
    det.update({f: [(10.0 * f, 100.0, 0.9)] for f in range(20, 25)})
    wasb = {f: (10.0 * f, 100.0, 0.6) for f in range(10, 20)}
    rows = fuse_track(det, wasb, _ts(range(25)))
    assert {r["seg_id"] for r in rows} == {0}
    assert any(r["source"] == "wasb" for r in rows)
    assert WASB_LOW_MIN < 0.6  # the carried peaks were full-strength


def test_defender_graze_never_takes_shot_binding():
    # body 7 possesses and releases; the rising ball grazes defender 4's
    # bbox for 2 frames mid-flight, then arrives at the rim. v2 bound the
    # shot to the LAST contact = the defender (ev-9). The graze gate keeps
    # the binding on 7 and persists the graze as data.
    track = []
    f = 0
    for _ in range(HOLD_MIN_FRAMES + 5):  # possession at y=500
        track.append({"frame_idx": f, "ts_ms": f * 33, "x": 100.0,
                      "y": 500.0, "conf": .9, "source": "det", "seg_id": 0})
        f += 1
    rise = GRAZE_UP_VY + 8.0  # launched: rising well above the gate
    graze_frames = []
    for i in range(22):  # flight up and over to the rim
        y = 500.0 - rise * (i + 1)
        track.append({"frame_idx": f, "ts_ms": f * 33,
                      "x": 100.0 + 38.0 * (i + 1), "y": y,
                      "conf": .9, "source": "det", "seg_id": 0})
        if i in (4, 5):  # two frames inside the defender's reach
            graze_frames.append(f)
        f += 1
    bodies = {fr: [(7, (80.0, 400.0, 120.0, 600.0))]
              for fr in range(HOLD_MIN_FRAMES + 5)}
    for fr in graze_frames:
        row = next(r for r in track if r["frame_idx"] == fr)
        bodies[fr] = [(4, (row["x"] - 20.0, row["y"] - 100.0,
                           row["x"] + 20.0, row["y"] + 100.0))]
    rim_f = f - 1
    rim_row = next(r for r in track if r["frame_idx"] == rim_f)
    rims = {fr: [(rim_row["x"] - 30.0, rim_row["y"] - 20.0,
                  rim_row["x"] + 30.0, rim_row["y"] + 20.0)]
            for fr in range(rim_f - 2, rim_f + 1)}
    events = derive_events(track, bodies, rims)
    shots = [e for e in events if e["kind"] == "shot"]
    assert len(shots) == 1
    assert shots[0]["track_id"] == 7  # the shooter, not defender 4
    grazes = [e for e in events if e["kind"] == "contact_graze"]
    assert [g["track_id"] for g in grazes] == [4]


def test_contact_runs_persist_as_events():
    track = []
    for f in range(HOLD_MIN_FRAMES + 3):
        track.append({"frame_idx": f, "ts_ms": f * 33, "x": 100.0,
                      "y": 500.0, "conf": .9, "source": "det", "seg_id": 0})
    bodies = {f: [(7, (80.0, 400.0, 120.0, 600.0))]
              for f in range(HOLD_MIN_FRAMES + 3)}
    events = derive_events(track, bodies, {})
    contacts = [e for e in events if e["kind"] == "contact"]
    assert len(contacts) == 1
    assert contacts[0]["track_id"] == 7
    assert contacts[0]["n_track_frames"] == HOLD_MIN_FRAMES + 3


def test_spine_contacts_reader_excludes_grazes(tmp_path):
    ew = ArtifactWriter(tmp_path / "ball_events", BALL_EVENTS_SCHEMA)
    base = dict(job_id="j", x=1.0, y=2.0, release_ts_ms=None,
                n_track_frames=3)
    ew.add({**base, "kind": "contact", "ts_ms": 1000, "ts_end_ms": 1200,
            "track_id": 7})
    ew.add({**base, "kind": "contact_graze", "ts_ms": 1400, "ts_end_ms": 1450,
            "track_id": 4})
    ew.close()
    rows = spine_contacts(tmp_path)
    assert [r["track_id"] for r in rows] == [7]


def test_spine_shooters_roundtrip(tmp_path):
    ew = ArtifactWriter(tmp_path / "ball_events", BALL_EVENTS_SCHEMA)
    ew.add({"job_id": "j", "kind": "shot", "ts_ms": 5000, "ts_end_ms": None,
            "track_id": 42, "x": 1.0, "y": 2.0, "release_ts_ms": 4000,
            "n_track_frames": None})
    ew.add({"job_id": "j", "kind": "hold", "ts_ms": 1000, "ts_end_ms": 2000,
            "track_id": 9, "x": 1.0, "y": 2.0, "release_ts_ms": None,
            "n_track_frames": None})
    ew.close()
    assert spine_shooters(tmp_path) == {5000: 42}


def test_local_rims_fixed_footage_matches_static():
    # stationary camera: jittered boxes around two fixed rims — the
    # windowed median equals the whole-clip median (v9's jitter kill kept)
    from montehall_cv.pipeline.ball_track import local_rims, static_rims

    rims_by_frame = {}
    for f in range(0, 300, 3):
        j = (f % 5) - 2.0  # ±2px jitter
        rims_by_frame[f] = [(100.0 + j, 200.0, 160.0 + j, 240.0),
                            (900.0 - j, 210.0, 960.0 - j, 250.0)]
    rf, rb = [], []
    for f in sorted(rims_by_frame):
        for b in rims_by_frame[f]:
            rf.append(f)
            rb.append(b)
    stat = static_rims(rims_by_frame)
    loc = local_rims(rf, rb, 150)
    assert len(loc) == len(stat) == 2
    for lb, sb in zip(loc, stat):
        assert all(abs(a - b) <= 2.0 for a, b in zip(lb, sb))


def test_local_rims_follows_panning_camera():
    # panning camera: the rim's detected x drifts 2px/frame. The local
    # median tracks the view; the whole-clip median lands in no-man's-land
    from montehall_cv.pipeline.ball_track import (
        RIM_LOCAL_FRAMES,
        local_rims,
        static_rims,
    )

    rims_by_frame = {f: [(100.0 + 2.0 * f, 200.0, 160.0 + 2.0 * f, 240.0)]
                     for f in range(0, 600, 2)}
    rf, rb = [], []
    for f in sorted(rims_by_frame):
        for b in rims_by_frame[f]:
            rf.append(f)
            rb.append(b)
    early = local_rims(rf, rb, 50)
    late = local_rims(rf, rb, 550)
    assert len(early) == len(late) == 1
    # local reference sits at the rim's position around THAT frame
    assert abs(early[0][0] - (100.0 + 2.0 * 50)) <= 2.0 * RIM_LOCAL_FRAMES
    assert abs(late[0][0] - (100.0 + 2.0 * 550)) <= 2.0 * RIM_LOCAL_FRAMES
    # the whole-clip reference (midline-split medians) sits more than a
    # full rim-split distance from the early view — guaranteed misses
    from montehall_cv.pipeline.ball_track import RIM_SPLIT_PX

    stat = static_rims(rims_by_frame)
    assert min(abs(sb[0] - early[0][0]) for sb in stat) > RIM_SPLIT_PX


def test_local_rims_two_rims_in_window_split():
    from montehall_cv.pipeline.ball_track import local_rims

    rf, rb = [], []
    for f in range(100):
        for b in ((300.0, 200.0, 360.0, 240.0),
                  (900.0, 200.0, 960.0, 240.0)):
            rf.append(f)
            rb.append(b)
    assert len(local_rims(rf, rb, 50)) == 2


def test_arrival_fires_on_panning_footage():
    # the clip-2 class: hold, then flight into a rim whose DETECTED box
    # pans with the camera — a whole-clip median never contains the drop;
    # the local reference does, so the arrival + shot fire
    track = []
    for f in range(HOLD_MIN_FRAMES + 3):
        track.append({"frame_idx": f, "ts_ms": f * 33, "x": 100.0,
                      "y": 500.0, "conf": .9, "source": "det", "seg_id": 0})
    flight_start = HOLD_MIN_FRAMES + 3
    for i, f in enumerate(range(flight_start, flight_start + 20)):
        track.append({"frame_idx": f, "ts_ms": f * 33,
                      "x": 100.0 + 40.0 * (i + 1),
                      "y": 500.0 - 17.0 * (i + 1),
                      "conf": .9, "source": "det", "seg_id": 0})
    bodies = {f: [(7, (80.0, 400.0, 120.0, 600.0))]
              for f in range(flight_start)}
    # camera pans hard: rim boxes drift 8px/frame; at the arrival frame
    # the rim sits under the ball (~x870,y155); 300 frames of history
    # would put a whole-clip median far from every real view
    arrive = flight_start + 19
    rims = {f: [(840.0 + 8.0 * (f - arrive), 125.0,
                 900.0 + 8.0 * (f - arrive), 165.0)]
            for f in range(arrive - 40, arrive + 1)}
    events = derive_events(track, bodies, rims)
    kinds = [e["kind"] for e in events]
    assert "rim_arrival" in kinds
    shots = [e for e in events if e["kind"] == "shot"]
    assert len(shots) == 1 and shots[0]["track_id"] == 7
