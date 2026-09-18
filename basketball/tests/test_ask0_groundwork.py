"""Ask 0 groundwork: local-corpus pairing, aligner period validation,
ball latent-state chain filter."""

import json

from montehall_cv.eval.espn_pbp import pair_videos_local, title_date_window
from montehall_cv.eval.pbp_align import build_timeline
from montehall_cv.store.records import DetClass
from montehall_cv.training.mine_broadcast import _ball_latent_filter, _corroborate


class _Read:
    def __init__(self, video_t, clock_s, conf=0.95):
        self.video_t = video_t
        self.clock_s = clock_s
        self.conf = conf
        self.text = "x"


# --- 0c: aligner monotonic-within-period validation --------------------------

def test_confirmed_period_break_still_breaks():
    reads = [_Read(10, 100.0), _Read(12, 98.0), _Read(14, 1200.0), _Read(16, 1198.0)]
    timeline = build_timeline(reads)
    assert [(p, ck) for p, ck, _ in timeline] == [
        (1, 100.0), (1, 98.0), (2, 1200.0), (2, 1198.0)
    ]


def test_unconfirmed_spike_is_dropped_not_a_period():
    # the game06 failure: one misread mid-period re-perioded the rest
    reads = [_Read(10, 100.0), _Read(12, 98.0), _Read(14, 1200.0),
             _Read(16, 96.0), _Read(18, 94.0)]
    timeline = build_timeline(reads)
    assert [(p, ck) for p, ck, _ in timeline] == [
        (1, 100.0), (1, 98.0), (1, 96.0), (1, 94.0)
    ]


def test_isolated_inperiod_increase_dropped_sustained_kept():
    # isolated +20s misread: dropped
    reads = [_Read(10, 100.0), _Read(12, 120.0), _Read(14, 97.0), _Read(16, 95.0)]
    assert [ck for _, ck, _ in build_timeline(reads)] == [100.0, 97.0, 95.0]
    # sustained correction (+20s regime holds): kept, same period
    reads = [_Read(10, 100.0), _Read(12, 120.0), _Read(14, 119.0), _Read(16, 118.0)]
    timeline = build_timeline(reads)
    assert [(p, ck) for p, ck, _ in timeline] == [
        (1, 100.0), (1, 120.0), (1, 119.0), (1, 118.0)
    ]


# --- 0a: title date windows + local pairing -----------------------------------

def test_title_date_window_season_and_tournament():
    reg = "A vs. B Full Game Replay ｜ 2025-26 ACC Men's Basketball [x].mp4"
    assert title_date_window(reg) == ("20251101", "20260430")
    tour = "A vs. B Full Game Replay ｜ 2026 T. Rowe Price ACC Men's Basketball Tour [x].mp4"
    assert title_date_window(tour) == ("20260301", "20260415")
    dated = "A vs. B Full Game Replay 1/16/2026 ｜ Big 12 [x].mp4"
    assert title_date_window(dated) == ("20260115", "20260117")
    assert title_date_window("A vs. B.mp4") is None


def _summary(eid, iso_date, loc_a, loc_b, n_plays=5):
    return {
        "header": {"competitions": [{
            "id": eid, "date": iso_date,
            "competitors": [{"team": {"location": loc_a}},
                            {"team": {"location": loc_b}}],
        }]},
        "plays": [{"id": i} for i in range(n_plays)],
    }


def test_pair_videos_local(tmp_path):
    corpus = tmp_path / "corpus" / "mens-college-basketball" / "2025-26"
    corpus.mkdir(parents=True)
    (corpus / "401.json").write_text(json.dumps(
        _summary("401", "2025-12-02T00:00Z", "Elon", "Virginia Tech")))
    # home-and-home rematch: two Duke-UNC events -> ambiguous, not paired
    (corpus / "402.json").write_text(json.dumps(
        _summary("402", "2026-01-10T00:00Z", "Duke", "North Carolina")))
    (corpus / "403.json").write_text(json.dumps(
        _summary("403", "2026-02-20T00:00Z", "Duke", "North Carolina")))

    videos = tmp_path / "videos"
    videos.mkdir()
    for name in (
        "Elon vs. Virginia Tech Full Game Replay ｜ 2025-26 ACC Men's Basketball [a].mp4",
        "Duke vs. North Carolina Full Game Replay ｜ 2025-26 ACC Men's Basketball [b].mp4",
        "Nowhere vs. Nothing Full Game Replay ｜ 2025-26 ACC Men's Basketball [c].mp4",
        "not a matchup title [d].mp4",
    ):
        (videos / name).write_text("")

    out = tmp_path / "pbp"
    report = pair_videos_local(videos, tmp_path / "corpus", out)
    totals = report["totals"]
    assert totals == {
        "videos": 4, "paired_or_existing": 1, "pairable_fraction": 0.25,
        "ambiguous": 1, "no_match": 1, "unparseable": 1,
    }
    paired = json.loads(next(out.glob("Elon*")).read_text())
    assert len(paired["plays"]) == 5
    # resumable: second run reports the existing file, writes nothing new
    report2 = pair_videos_local(videos, tmp_path / "corpus", out)
    assert report2["totals"]["paired_or_existing"] == 1
    assert len(report2["already"]) == 1


# --- 0b: ball latent-state chain filter ---------------------------------------

def _ball(t, x, y, conf=0.9, fname=None):
    return {"fname": fname or f"f_t{int(t * 1000)}.jpg", "t": t,
            "box": [x - 8, y - 8, x + 8, y + 8], "conf": conf,
            "cls": int(DetClass.BALL)}


def _player(fname, x, y, w=40, h=90):
    return {"fname": fname, "t": 0.0, "box": [x, y, x + w, y + h],
            "conf": 0.9, "cls": int(DetClass.PERSON)}


def test_stationary_crowd_blob_rejected_despite_confidence():
    # the audit failure mode: a high-conf head that never moves
    balls = [_ball(t, 500.0, 80.0, conf=0.95) for t in (0.0, 0.2, 0.4, 0.6, 0.8)]
    assert _ball_latent_filter(balls, players=[], scale=1.0) == []


def test_flight_arc_accepted_without_anchor():
    balls = [_ball(t, 300 + 120 * t, 200 - 90 * t * (2 - t)) for t in
             (0.0, 0.2, 0.4, 0.6)]
    kept = _ball_latent_filter(balls, players=[], scale=1.0)
    assert len(kept) == 4


def test_dribble_bounce_accepted_only_when_anchored():
    fnames = [f"f{i}.jpg" for i in range(4)]
    balls = [_ball(0.2 * i, 320.0, 300.0 + (18 if i % 2 else 0), fname=fnames[i])
             for i in range(4)]
    # no player nearby: small spread in open space -> rejected
    assert _ball_latent_filter(balls, players=[], scale=1.0) == []
    # same chain next to a player box on each frame -> accepted
    players = [_player(f, 300.0, 250.0) for f in fnames]
    assert len(_ball_latent_filter(balls, players, scale=1.0)) == 4


def test_singleton_and_short_chains_rejected():
    assert _ball_latent_filter([_ball(0.0, 100, 100, conf=0.99)], [], 1.0) == []
    two = [_ball(0.0, 100, 100), _ball(0.2, 160, 140)]
    assert _ball_latent_filter(two, [], 1.0) == []


def test_scale_doubles_thresholds_at_720p():
    # same flight in 1280-wide coords (2x): spread doubles, so does the bar
    balls = [_ball(t, 600 + 240 * t, 400 - 180 * t * (2 - t)) for t in
             (0.0, 0.2, 0.4, 0.6)]
    assert len(_ball_latent_filter(balls, [], scale=2.0)) == 4


def test_corroborate_end_to_end_ball_path():
    fnames = [f"g_t{i}.jpg" for i in range(5)]
    moving = [_ball(0.2 * i, 100 + 40 * i, 200 - 30 * i, fname=fnames[i])
              for i in range(5)]
    static = [_ball(0.2 * i, 600, 90, conf=0.9, fname=fnames[i]) for i in range(5)]
    kept = _corroborate(moving + static, frame_w=640.0)
    xs = sorted(r["box"][0] for r in kept)
    assert len(kept) == 5 and all(x < 300 for x in xs)


# --- Ask 1: sealed holdout registry -------------------------------------------

def test_sealed_holdouts_registry():
    from montehall_cv.eval.holdouts import SEALED_EVENT_IDS, is_sealed
    assert is_sealed("401851175") and is_sealed("401851182")  # ACC pair
    assert is_sealed(401851175)  # int ids normalize
    assert not is_sealed("401820703")
    assert len(SEALED_EVENT_IDS) >= 2


# --- Ask 2: mega-mine work selection ------------------------------------------

def test_mine_all_work_items_seal_and_staging(tmp_path):
    from montehall_cv.training.mine_all import work_items
    videos, pbp = tmp_path / "v", tmp_path / "p"
    align_root, shard_root = tmp_path / "a", tmp_path / "s"
    videos.mkdir(), pbp.mkdir()

    def game(stem, eid):
        (videos / f"{stem}.mp4").write_text("")
        (pbp / f"{stem}.pbp.json").write_text(json.dumps(
            {"header": {"competitions": [{"id": eid}]}, "plays": [{}]}))

    game("normal", "555001")
    game("sealed", "401851182")          # Clemson-Duke holdout
    game("aligned_already", "555002")
    (pbp / "no_video.pbp.json").write_text(json.dumps(
        {"header": {"competitions": [{"id": "555003"}]}, "plays": [{}]}))
    marker = align_root / "eid_555002" / "pbp_alignment"
    marker.mkdir(parents=True)
    (marker / "_SUCCESS").touch()

    items = work_items(videos, pbp, align_root, shard_root)
    by_eid = {i["eid"]: i for i in items}
    assert set(by_eid) == {"555001", "555002"}  # sealed + missing video out
    assert not by_eid["555001"]["aligned"]
    assert by_eid["555002"]["aligned"] and not by_eid["555002"]["mined"]
