"""Silent-swap proxy: jersey-number conflicts inside one track/segment.

A binding row stream (ByteTrack's track_binding or the quark layer's
quark_binding) claims continuity; contact_reads carry sparse hard
identity facts. A single track_id accumulating TWO different
quorum-confident jersey numbers has bridged two humans — the exact
failure the quark layer exists to make non-silent. Running the same
measurement over both stages is the apples-to-apples swap comparison
(SoccerNet GSR uses number-conflict as its tracklet split criterion;
here it is the metric).

Usage (inside cvbench):
    python scripts/quark_swap_proxy.py \
        --binding-stage /work/models/quark-v1/cal_smoke2/quark_binding \
        --reads-dir /work/out-eval-v3/cal_fsu_acc26 \
        --out /work/models/quark-v1/swap_proxy_quark_cal.json \
        [--start-s 600 --duration-s 120]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from montehall_cv.store.artifacts import read_stage

from people_relink_smoke import READ_CONF_MIN, bridge_stored_votes

CONFLICT_MIN_VOTES = 5  # bridged frames a number needs before it counts
                        # toward a conflict (below READ_MIN_VOTES=10 by
                        # design: a swap's second number rarely gets a
                        # full segment's worth of reads)


def load_stage_binding(stage_dir: Path, lo_ms: float, hi_ms: float):
    t = read_stage(stage_dir)
    bts = t.column("ts_ms").to_numpy()
    btrack = t.column("track_id").to_numpy()
    boxes = np.stack([t.column(c).to_numpy()
                      for c in ("x1", "y1", "x2", "y2")], axis=1)
    win = (bts >= lo_ms) & (bts < hi_ms)
    if "coasted" in t.schema.names:
        # a coasted box is synthesized, not observed — letting it
        # collect bridged jersey votes through a pile manufactures
        # conflicts out of interpolation (v2.1 read .373 vs v1 .282 on
        # exactly this artifact)
        win &= ~t.column("coasted").to_numpy()
    bts, btrack, boxes = bts[win], btrack[win], boxes[win]
    order = np.argsort(bts, kind="stable")
    return bts[order], btrack[order], boxes[order]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binding-stage", type=Path, required=True)
    ap.add_argument("--reads-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=0.0)
    args = ap.parse_args()

    lo_ms = args.start_s * 1000
    hi_ms = (args.start_s + args.duration_s) * 1000 if args.duration_s \
        else float("inf")
    bts, btrack, boxes = load_stage_binding(args.binding_stage, lo_ms, hi_ms)

    number_of_stored = {
        r["track_id"]: r["number"]
        for r in read_stage(args.reads_dir / "contact_reads").to_pylist()
        if r["number"] is not None and r["confidence"] >= READ_CONF_MIN
    }
    votes = bridge_stored_votes(args.reads_dir, number_of_stored,
                                bts, btrack, boxes, lo_ms, hi_ms)

    by_track: dict[int, Counter] = defaultdict(Counter)
    vote_ts: dict[tuple[int, str], list[int]] = defaultdict(list)
    for tid, ts, num in votes:
        by_track[tid][num] += 1
        vote_ts[(tid, num)].append(ts)

    # flag lookup: when the stage carries the quark ambiguity channel,
    # a conflict whose minority-number votes land on contested/disputed
    # rows was REPORTED — only a conflict on clean rows is a silent swap
    flagged_ts: dict[int, list[int]] = defaultdict(list)
    stage_table = read_stage(args.binding_stage)
    has_flags = "contested" in stage_table.schema.names
    if has_flags:
        f_tid = stage_table.column("track_id").to_numpy()
        f_ts = stage_table.column("ts_ms").to_numpy()
        f_bad = (stage_table.column("contested").to_numpy()
                 | stage_table.column("disputed").to_numpy())
        for tid, ts in zip(f_tid[f_bad].tolist(), f_ts[f_bad].tolist()):
            flagged_ts[int(tid)].append(int(ts))
        for lst in flagged_ts.values():
            lst.sort()

    def near_flag(tid: int, ts_list: list[int], slack_ms: int = 100) -> float:
        from bisect import bisect_left
        flags = flagged_ts.get(tid, [])
        if not flags:
            return 0.0
        hit = 0
        for ts in ts_list:
            i = bisect_left(flags, ts)
            for k in (i - 1, i):
                if 0 <= k < len(flags) and abs(flags[k] - ts) <= slack_ms:
                    hit += 1
                    break
        return hit / len(ts_list)

    conflicts = []
    n_read_tracks = 0
    for tid, counter in sorted(by_track.items()):
        quorum = {num: n for num, n in counter.items()
                  if n >= CONFLICT_MIN_VOTES}
        if quorum:
            n_read_tracks += 1
        if len(quorum) >= 2:
            minority = min(quorum, key=lambda num: quorum[num])
            entry = {"track_id": int(tid), "numbers": quorum}
            if has_flags:
                entry["minority_votes_near_flag"] = round(
                    near_flag(int(tid), vote_ts[(tid, minority)]), 3)
            conflicts.append(entry)

    out = {
        "binding_stage": str(args.binding_stage),
        "window_s": [args.start_s,
                     args.start_s + args.duration_s if args.duration_s
                     else None],
        "tracks_in_window": len(set(btrack.tolist())),
        "bridged_votes": len(votes),
        "tracks_with_quorum_number": n_read_tracks,
        "tracks_with_conflict": len(conflicts),
        "conflict_rate": round(len(conflicts) / n_read_tracks, 4)
        if n_read_tracks else None,
        "conflicts": conflicts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print("SWAP_PROXY_DONE " + json.dumps(
        {k: out[k] for k in ("tracks_in_window", "bridged_votes",
                             "tracks_with_quorum_number",
                             "tracks_with_conflict", "conflict_rate")}),
        flush=True)


if __name__ == "__main__":
    main()
