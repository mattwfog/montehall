"""Render-churn probe: measure what the people renderer actually DRAWS.

2026-08-05 visual review: people_quark_smoke_v2.mp4 (quark
v2.3b) is a clear regression from v1's render, while v2.3b's reported
metrics improved (conflict rate .2954 -> .1477, silent swaps 22 -> 2,
supply .9693 -> .9755). The candidate mechanism was recorded as
INFERRED and UNPROBED. This probe measures it instead of assuming it.

What the renderer does (people_track_video.py, verified 2026-08-17):
  * reads EVERY row of the binding stage -- there is no observed/coasted
    filter (lines 70-81), so coasted rows are drawn as boxes;
  * without --relink-map, label = "T{n}" and color = palette[n % 20],
    keyed on track_id by first appearance (lines 145-162).
So a viewer sees: one color per SEGMENT, recycled every 20 segments,
plus a box wherever a seeker coasts -- even with no body under it.

Measured here, over the exact rendered window:
  1. rows drawn, split observed / coasted / contested / disputed
  2. boxes per frame (mean / median / p95 / max)
  3. distinct segments drawn = distinct colors the eye must track
  4. ID CHURN along visual trajectories: boxes are IoU-chained frame to
     frame (what the eye follows); a chain whose track_id changes is a
     body that changed color mid-motion. This is the regression claim,
     made falsifiable.
  5. COLOR COLLISIONS: with a 20-entry palette, two different segments
     on screen at once sharing a color read as one person in two places.

CPU-only; consumes stored binding stages, runs no model, writes no
stage. Usage:

    python scripts/render_churn_probe.py \
        --binding v1=/work/models/quark-v1/cal_full/quark_binding \
        --binding v23b=/work/models/quark-v1/cal_full_v23b/quark_binding \
        --start-s 600 --duration-s 120 \
        --report /work/models/quark-v1/render_churn_cal.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from montehall_cv.store.artifacts import read_stage

PALETTE_N = 20      # people_track_video.py palette length
IOU_MATCH = 0.5     # frame-to-frame chaining threshold
MIN_CHAIN = 5       # frames; shorter chains are not a trajectory the eye follows


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (N,4) and (M,4) box arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def _greedy_match(iou: np.ndarray) -> list[tuple[int, int]]:
    """Greedy highest-IoU-first pairing above IOU_MATCH. Returns (i, j)."""
    pairs: list[tuple[int, int]] = []
    if iou.size == 0:
        return pairs
    flat = np.argsort(iou, axis=None)[::-1]
    used_i: set[int] = set()
    used_j: set[int] = set()
    for k in flat:
        i, j = int(k // iou.shape[1]), int(k % iou.shape[1])
        if iou[i, j] < IOU_MATCH:
            break
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j))
    return pairs


def _load_window(stage: Path, lo_ms: float, hi_ms: float) -> dict:
    """Read a binding stage and slice it to the rendered window."""
    t = read_stage(stage)
    ts = t.column("ts_ms").to_numpy()
    win = (ts >= lo_ms) & (ts < hi_ms)
    if not win.any():
        raise SystemExit(f"{stage}: no rows in window [{lo_ms}, {hi_ms}) ms")
    cols = ("frame_idx", "ts_ms", "track_id", "x1", "y1", "x2", "y2",
            "contested", "coasted", "disputed")
    return {c: t.column(c).to_numpy(zero_copy_only=False)[win] for c in cols}


def _chain_churn(frames: list[int], per_frame: dict) -> dict:
    """IoU-chain boxes across frames; count id/color changes per chain."""
    chains: list[list[int]] = []          # each chain = list of track_ids seen
    active: dict[int, int] = {}           # row-index-in-frame -> chain index
    for fi in range(len(frames)):
        boxes, tids = per_frame[frames[fi]]
        if fi == 0:
            for r in range(len(boxes)):
                chains.append([int(tids[r])])
                active[r] = len(chains) - 1
            continue
        prev_boxes, _ = per_frame[frames[fi - 1]]
        pairs = _greedy_match(_iou_matrix(prev_boxes, boxes))
        nxt: dict[int, int] = {}
        matched = set()
        for pi, ci in pairs:
            if pi not in active:
                continue
            idx = active[pi]
            chains[idx].append(int(tids[ci]))
            nxt[ci] = idx
            matched.add(ci)
        for r in range(len(boxes)):
            if r in matched:
                continue
            chains.append([int(tids[r])])
            nxt[r] = len(chains) - 1
        active = nxt

    long_chains = [c for c in chains if len(c) >= MIN_CHAIN]
    id_flips = [sum(1 for a, b in zip(c, c[1:]) if a != b) for c in long_chains]
    return {
        "chains_total": len(chains),
        "chains_ge_min": len(long_chains),
        "min_chain_frames": MIN_CHAIN,
        "chains_with_id_flip": int(sum(1 for f in id_flips if f > 0)),
        "chain_flip_rate": round(
            sum(1 for f in id_flips if f > 0) / max(len(long_chains), 1), 4),
        "id_flips_total": int(sum(id_flips)),
        "flips_per_chain_mean": round(float(np.mean(id_flips)), 3) if id_flips else 0.0,
        "flips_per_chain_max": int(max(id_flips)) if id_flips else 0,
    }


def _color_collisions(frames: list[int], per_frame: dict,
                      color_of: dict[int, int]) -> dict:
    """Frames where two live segments on screen share a palette colour."""
    collide_frames = 0
    worst = 0
    for f in frames:
        _, tids = per_frame[f]
        colors = [color_of[int(t)] for t in tids]
        dupes = len(colors) - len(set(colors))
        if dupes > 0:
            collide_frames += 1
            worst = max(worst, dupes)
    return {
        "frames_with_colour_collision": collide_frames,
        "collision_frame_rate": round(collide_frames / max(len(frames), 1), 4),
        "max_simultaneous_duplicate_colours": worst,
        "palette_size": PALETTE_N,
    }


def probe(tag: str, stage: Path, lo_ms: float, hi_ms: float) -> dict:
    d = _load_window(stage, lo_ms, hi_ms)
    boxes = np.stack([d["x1"], d["y1"], d["x2"], d["y2"]], axis=1)
    frame_ids = d["frame_idx"]
    frames = sorted(set(int(f) for f in frame_ids))

    per_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for f in frames:
        m = frame_ids == f
        per_frame[f] = (boxes[m], d["track_id"][m])

    # Replicate the renderer's colour assignment: compact id by first
    # appearance, colour = palette[n % 20].
    color_of: dict[int, int] = {}
    for f in frames:
        for t in per_frame[f][1]:
            t = int(t)
            if t not in color_of:
                color_of[t] = len(color_of) % PALETTE_N

    counts = np.array([len(per_frame[f][0]) for f in frames], dtype=np.float64)
    n = len(d["track_id"])
    return {
        "tag": tag,
        "stage": str(stage),
        "frames_drawn": len(frames),
        "rows_drawn": int(n),
        "rows_observed": int(np.sum(~d["coasted"])),
        "rows_coasted": int(np.sum(d["coasted"])),
        "coast_draw_rate": round(float(np.sum(d["coasted"]) / max(n, 1)), 4),
        "rows_contested": int(np.sum(d["contested"])),
        "rows_disputed": int(np.sum(d["disputed"])),
        "distinct_segments_drawn": int(len(color_of)),
        "boxes_per_frame": {
            "mean": round(float(counts.mean()), 2),
            "median": float(np.median(counts)),
            "p95": float(np.percentile(counts, 95)),
            "max": int(counts.max()),
        },
        "churn": _chain_churn(frames, per_frame),
        "colour": _color_collisions(frames, per_frame, color_of),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binding", action="append", required=True,
                    metavar="TAG=PATH",
                    help="binding stage to probe; repeatable")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=0.0,
                    help="0 = to end of stage")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    lo_ms = args.start_s * 1000.0
    hi_ms = (lo_ms + args.duration_s * 1000.0) if args.duration_s > 0 else float("inf")

    runs = []
    for spec in args.binding:
        if "=" not in spec:
            raise SystemExit(f"--binding expects TAG=PATH, got {spec!r}")
        tag, path = spec.split("=", 1)
        runs.append(probe(tag, Path(path), lo_ms, hi_ms))

    out = {
        "window_s": [args.start_s,
                     None if args.duration_s <= 0 else args.start_s + args.duration_s],
        "renderer": "people_track_video.py, no --relink-map (raw per-track colours)",
        "runs": runs,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
