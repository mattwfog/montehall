"""Quark layer probe: run the seeker engine over a game's detections and
measure shot-anchor continuity supply.

Consumes the detections stage (person class), runs the bidirectional
quark engine (pipeline/quark.py), writes a binding-shaped quark_binding
stage the relink/lock/render chain can consume, and reports the ruled
supply metric: the fraction of shot anchors with a quark segment alive
through the ±2s window. Baseline to beat [observed 2026-08-03, eval-real
candidate_supply]: ByteTrack-era track supply 8-16% of anchors, while
person detections are pixel-present at ~100%.

CPU-only — detections already exist; no GPU stage runs here.

Usage (inside cvbench, or anywhere the artifacts are mounted):
    python scripts/quark_probe.py /work/out-harvest/cal_fsu_v3real \
        --out-root /work/models/quark-v1/cal_fsu_v3real \
        --report /work/models/quark-v1/quark_cal.json \
        [--start-s 600 --duration-s 120]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa

from montehall_cv.brain.dataset import BIN_MS
from montehall_cv.brain.eval_slots import SHOT_WINDOW_BINS, _shot_anchors
from montehall_cv.pipeline.quark import run_bidirectional
from montehall_cv.store.artifacts import ArtifactWriter, read_stage, stage_complete
from montehall_cv.store.records import DetClass

QUARK_SCHEMA = pa.schema([
    ("frame_idx", pa.int32()),
    ("ts_ms", pa.int64()),
    ("det_idx", pa.int32()),
    ("track_id", pa.int32()),
    ("x1", pa.float32()),
    ("y1", pa.float32()),
    ("x2", pa.float32()),
    ("y2", pa.float32()),
    ("conf", pa.float32()),
    ("contested", pa.bool_()),
    ("coasted", pa.bool_()),
    ("disputed", pa.bool_()),
])

COVER_FRAC = 0.8  # window coverage a single segment needs to count as
                  # continuity THROUGH the shot moment


def load_frames(game_dir: Path, lo_ms: float, hi_ms: float):
    table = read_stage(game_dir / "detections")
    cols = {c: table.column(c).to_numpy() for c in
            ("frame_idx", "ts_ms", "det_idx", "cls", "x1", "y1", "x2", "y2",
             "conf")}
    keep = (cols["cls"] == int(DetClass.PERSON)) \
        & (cols["ts_ms"] >= lo_ms) & (cols["ts_ms"] < hi_ms)
    dets = np.stack([cols[c][keep].astype(np.float64) for c in
                     ("x1", "y1", "x2", "y2", "conf", "det_idx")], axis=1)
    fidx = cols["frame_idx"][keep]
    ts = cols["ts_ms"][keep]
    by_frame: dict[int, list[int]] = defaultdict(list)
    for i, f in enumerate(fidx.tolist()):
        by_frame[f].append(i)
    frames = []
    for f in sorted(by_frame, key=lambda f: ts[by_frame[f][0]]):
        idxs = by_frame[f]
        frames.append((int(f), float(ts[idxs[0]]), dets[idxs]))
    return frames


def anchor_supply(game_dir: Path, rows: list[dict],
                  lo_ms: float, hi_ms: float) -> dict:
    """Ruled metric: shot anchors (brain tokens) vs quark segment
    presence and continuity in the ±SHOT_WINDOW window. Only anchors
    inside the probed [lo_ms, hi_ms) window count — comparing a
    windowed run against whole-game anchors reads as near-zero supply
    no matter what the engine did (the 08-05 smoke's 6/163 artifact)."""
    if not stage_complete(game_dir / "tokens"):
        return {"anchors": 0, "note": "no tokens stage; supply skipped"}
    tokens = read_stage(game_dir / "tokens").to_pylist()
    anchors = [(b, j) for b, j in _shot_anchors(tokens)
               if lo_ms <= b * BIN_MS < hi_ms]

    span_of: dict[int, list[int]] = {}
    rows_ts = defaultdict(list)
    for r in rows:
        s = span_of.setdefault(r["track_id"], [r["ts_ms"], r["ts_ms"]])
        s[0], s[1] = min(s[0], r["ts_ms"]), max(s[1], r["ts_ms"])
        rows_ts[r["track_id"]].append(r["ts_ms"])
    spans = sorted((t0, t1) for t0, t1 in span_of.values())

    n_any = n_covered = 0
    for b, _jersey in anchors:
        w0 = (b - SHOT_WINDOW_BINS) * BIN_MS
        w1 = (b + SHOT_WINDOW_BINS + 1) * BIN_MS
        need = COVER_FRAC * (w1 - w0)
        any_hit = covered = False
        for t0, t1 in spans:
            if t0 >= w1:
                break
            overlap = min(t1, w1) - max(t0, w0)
            if overlap > 0:
                any_hit = True
                if overlap >= need:
                    covered = True
                    break
        n_any += int(any_hit)
        n_covered += int(covered)
    return {
        "anchors": len(anchors),
        "anchors_any_quark": n_any,
        "anchors_covered_80": n_covered,
        "any_frac": round(n_any / len(anchors), 4) if anchors else None,
        "covered_frac": round(n_covered / len(anchors), 4) if anchors else None,
    }


def build_pose_lookup(pose_stage: Path):
    """ts_ms -> (boxes[N,4], kps[N,17,3]) from the extremities stage,
    nearest pose frame within POSE_MATCH_MS (pose runs at stride-2)."""
    t = read_stage(pose_stage)
    ts = t.column("ts_ms").to_numpy()
    boxes = np.stack([t.column(c).to_numpy()
                      for c in ("x1", "y1", "x2", "y2")], axis=1)
    kps = np.stack(
        [np.asarray(v, dtype=np.float64).reshape(17, 3)
         for v in t.column("kp").to_pylist()]) if len(ts) \
        else np.zeros((0, 17, 3))
    order = np.argsort(ts, kind="stable")
    ts, boxes, kps = ts[order], boxes[order], kps[order]
    uniq = np.unique(ts)

    def lookup(query_ms: float):
        i = np.searchsorted(uniq, query_ms)
        best = None
        for k in (i - 1, i):
            if 0 <= k < len(uniq) and abs(uniq[k] - query_ms) \
                    <= POSE_MATCH_MS:
                if best is None or abs(uniq[k] - query_ms) \
                        < abs(best - query_ms):
                    best = uniq[k]
        if best is None:
            return None
        sel = ts == best
        return boxes[sel], kps[sel]

    return lookup


POSE_MATCH_MS = 25


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game_dir", type=Path)
    ap.add_argument("--out-root", type=Path, required=True,
                    help="quark_binding stage lands at <out-root>/"
                         "quark_binding (NOT inside game_dir unless the "
                         "run covers the whole game)")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=0.0,
                    help="0 = whole game")
    ap.add_argument("--pose-stage", type=Path, default=None,
                    help="extremities stage (pose_stage.py) — enables "
                         "the limb-continuity claim cost")
    args = ap.parse_args()

    lo_ms = args.start_s * 1000
    hi_ms = (args.start_s + args.duration_s) * 1000 if args.duration_s \
        else float("inf")

    t0 = time.monotonic()
    frames = load_frames(args.game_dir, lo_ms, hi_ms)
    t_load = time.monotonic() - t0
    n_dets = int(sum(len(f[2]) for f in frames))

    pose_lookup = None
    if args.pose_stage is not None:
        pose_lookup = build_pose_lookup(args.pose_stage)

    t0 = time.monotonic()
    rows, stats = run_bidirectional(frames, pose_lookup)
    t_engine = time.monotonic() - t0

    stage_dir = args.out_root / "quark_binding"
    writer = ArtifactWriter(stage_dir, QUARK_SCHEMA)
    writer.add_many(rows)
    writer.close()

    window_min = ((frames[-1][1] - frames[0][1]) / 60000) if frames else 0
    report = {
        "game_dir": str(args.game_dir),
        "window_s": [args.start_s,
                     args.start_s + args.duration_s if args.duration_s
                     else None],
        "frames": len(frames),
        "person_dets": n_dets,
        **stats,
        "segments_per_2min": round(
            stats["segments_fwd"] / max(window_min / 2, 1e-9), 1),
        "supply": anchor_supply(args.game_dir, rows, lo_ms, hi_ms),
        "load_s": round(t_load, 2),
        "engine_s": round(t_engine, 2),
        "engine_fps": round(len(frames) / max(t_engine, 1e-9), 1),
        "stage_dir": str(stage_dir),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print("QUARK_PROBE_DONE " + json.dumps(
        {k: report[k] for k in ("frames", "person_dets", "segments_fwd",
                                "segments_per_2min", "coalescences_fwd",
                                "rows_contested", "rows_disputed",
                                "rows_coasted", "supply", "engine_fps")}),
        flush=True)


if __name__ == "__main__":
    main()
