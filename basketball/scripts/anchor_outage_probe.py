"""Shot-anchor perception-outage probe (2026-08-03 ball-first finding).

The ball-v1 real eval measured that 84-92% of shot anchors have zero
player TRACKS in the +-2s window. This probe separates the candidate
mechanisms per anchor, read-only, from existing extract artifacts:

  - detections (pixel space): did the detector see people at all?
  - positions (court space):  did the court solve deliver coordinates?
  - clock/cut tokens:         is the window inside a replay/cut
                              (alignment-offset alternative)?

Verdict per anchor window:
  no_pixels      detections has no PERSON rows -> decode/detector/replay
  court_dead     pixels yes, positions no      -> court solve outage
  court_alive    positions rows exist          -> tokens/track layer
  (each also flagged in_cut when a cut token overlaps the window)

Usage (inside cvbench):
    python scripts/anchor_outage_probe.py /work/out-harvest/<game_dir> \
        --out /work/models/ball-v1/outage_<game>.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from montehall_cv.brain.eval_slots import SHOT_WINDOW_BINS, _shot_anchors
from montehall_cv.brain.dataset import BIN_MS
from montehall_cv.brain.tokens import CH_CLOCK, CH_CUT, CH_PLAYER
from montehall_cv.store.artifacts import read_stage, stage_complete

# detections + tracklets store DetClass values (records.py:21):
# PERSON=0, BALL=1, REF=2 (never emitted), RIM=3.
# ⚠ 2026-08-03: the first version of this probe used 1=person/2=ball
# (the raw fine-tune id space, which detect.py maps AWAY before
# persisting) — its "person" counts were BALL detections and its
# "ball" counts were always-empty REF. Every number from that version
# is retracted; see project memory.
CLS_PERSON = 0
CLS_BALL = 1
CLS_RIM = 3
CLS_PERSON_POSITIONS = 0  # positions stage: 0=person, 1=ball


def _window_ms(b: int) -> tuple[int, int]:
    return ((b - SHOT_WINDOW_BINS) * BIN_MS,
            (b + SHOT_WINDOW_BINS + 1) * BIN_MS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tokens = read_stage(args.game_dir / "tokens").to_pylist()
    anchors = _shot_anchors(tokens)
    det_ts, det_cls = [], []
    if stage_complete(args.game_dir / "detections"):
        table = read_stage(args.game_dir / "detections")
        det_ts = table.column("ts_ms").to_pylist()
        det_cls = table.column("cls").to_pylist()
    pos_ts, pos_cls = [], []
    if stage_complete(args.game_dir / "positions"):
        table = read_stage(args.game_dir / "positions")
        pos_ts = table.column("ts_ms").to_pylist()
        pos_cls = table.column("cls").to_pylist()

    det_person = sorted(t for t, c in zip(det_ts, det_cls)
                        if c == CLS_PERSON)
    det_ball = sorted(t for t, c in zip(det_ts, det_cls) if c == CLS_BALL)
    det_rim = sorted(t for t, c in zip(det_ts, det_cls) if c == CLS_RIM)
    pos_person = sorted(t for t, c in zip(pos_ts, pos_cls)
                        if c == CLS_PERSON_POSITIONS)

    # court solve health per sampled frame: h None = failed solve;
    # n_points = landmarks the solver had (zoom starves this)
    court_rows: list[tuple[int, bool, int]] = []
    if stage_complete(args.game_dir / "court_frames"):
        table = read_stage(args.game_dir / "court_frames")
        court_rows = sorted(zip(
            table.column("ts_ms").to_pylist(),
            [h is not None for h in table.column("h").to_pylist()],
            table.column("n_points").to_pylist()))
    court_ts = [r[0] for r in court_rows]
    clock_ms = sorted(t["t_ms"] for t in tokens if t["channel"] == CH_CLOCK)
    cut_ms = sorted(t["t_ms"] for t in tokens if t["channel"] == CH_CUT)
    player_tok_ms = sorted(t["t_ms"] for t in tokens
                           if t["channel"] == CH_PLAYER)

    import bisect

    def count_in(sorted_ms: list[int], lo: int, hi: int) -> int:
        return bisect.bisect_left(sorted_ms, hi) - bisect.bisect_left(
            sorted_ms, lo)

    rows = []
    verdicts: Counter[str] = Counter()
    for b, _jersey in anchors:
        lo, hi = _window_ms(b)
        n_det = count_in(det_person, lo, hi)
        n_pos = count_in(pos_person, lo, hi)
        n_tok = count_in(player_tok_ms, lo, hi)
        in_cut = count_in(cut_ms, lo - 5000, hi + 5000) > 0
        c_lo = bisect.bisect_left(court_ts, lo)
        c_hi = bisect.bisect_left(court_ts, hi)
        window_court = court_rows[c_lo:c_hi]
        n_court = len(window_court)
        n_court_ok = sum(1 for _t, ok, _n in window_court if ok)
        pts = [n for _t, _ok, n in window_court]
        if n_det == 0:
            verdict = "no_pixels"
        elif n_pos == 0:
            verdict = "court_dead"
        else:
            verdict = "court_alive"
        verdicts[verdict] += 1
        rows.append({"anchor_bin": b, "det_person": n_det,
                     "det_ball": count_in(det_ball, lo, hi),
                     "det_rim": count_in(det_rim, lo, hi),
                     "pos_person": n_pos, "player_tokens": n_tok,
                     "court_samples": n_court, "court_ok": n_court_ok,
                     "court_n_points_mean": (round(sum(pts) / len(pts), 1)
                                             if pts else None),
                     "clock_reads": count_in(clock_ms, lo, hi),
                     "in_cut": in_cut, "verdict": verdict})

    def _mean(vals: list) -> float | None:
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    out = {
        "game": args.game_dir.name,
        "anchors": len(anchors),
        "verdicts": dict(verdicts),
        "in_cut_by_verdict": {
            v: sum(1 for r in rows if r["verdict"] == v and r["in_cut"])
            for v in verdicts},
        "by_verdict": {
            v: {
                "court_ok_frac": _mean(
                    [r["court_ok"] / r["court_samples"] for r in rows
                     if r["verdict"] == v and r["court_samples"]]),
                "court_n_points_mean": _mean(
                    [r["court_n_points_mean"] for r in rows
                     if r["verdict"] == v]),
                "det_ball_mean": _mean(
                    [r["det_ball"] for r in rows if r["verdict"] == v]),
                "det_rim_mean": _mean(
                    [r["det_rim"] for r in rows if r["verdict"] == v]),
                "anchors_with_ball_px": sum(
                    1 for r in rows
                    if r["verdict"] == v and r["det_ball"] > 0),
                "anchors_with_rim_px": sum(
                    1 for r in rows
                    if r["verdict"] == v and r["det_rim"] > 0),
            } for v in verdicts},
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("game", "anchors", "verdicts", "in_cut_by_verdict",
                       "by_verdict")}))


if __name__ == "__main__":
    main()
