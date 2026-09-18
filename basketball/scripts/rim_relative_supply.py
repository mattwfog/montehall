"""Rim-relative candidate supply — stage-1 of the shot-window court fix.

The court homography is dead in 84-92% of shot windows, but the rim is
pixel-visible in ~99% of them and player detections carry track_id
straight through. This measures how much shooter-candidate supply exists
WITHOUT any homography: rim pixel center + rim physical size give a
ft-per-px scale; player bbox bottom-centers give distances-from-rim in
feet inside each +-2s anchor window.

Geometry caveats (named, stage-1): the rim det box is assumed to box the
hoop (1.5 ft) — if it boxes backboard the scale is ~4x off, so both a
raw and an x-only distance are reported (side-on broadcast: image-x
tracks court length; image-y mixes height and depth, and the rim sits
10 ft above the floor).

Usage (inside cvbench):
    python scripts/rim_relative_supply.py /work/out-harvest/<game_dir> \
        --out /work/models/ball-v1/rimrel_<game>.json
"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from pathlib import Path

import numpy as np

from montehall_cv.brain.dataset import BIN_MS
from montehall_cv.brain.eval_slots import SHOT_WINDOW_BINS, _shot_anchors
from montehall_cv.store.artifacts import read_stage

CLS_PERSON = 0  # DetClass space (⚠ was wrongly 1=BALL until 2026-08-03)
CLS_RIM = 3
RIM_TRUE_FT = 1.5  # hoop outer diameter 18 in
RIM_CONF_MIN = 0.3
RADII_FT = (4.0, 6.0, 8.0, 10.0, 15.0)
MIN_FRAMES = 3  # anti-noise: near-rim presence must persist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tokens = read_stage(args.game_dir / "tokens").to_pylist()
    anchors = _shot_anchors(tokens)
    det = read_stage(args.game_dir / "detections")
    ts = det.column("ts_ms").to_numpy()
    cls = det.column("cls").to_numpy()
    x1 = det.column("x1").to_numpy()
    y1 = det.column("y1").to_numpy()
    x2 = det.column("x2").to_numpy()
    y2 = det.column("y2").to_numpy()
    conf = det.column("conf").to_numpy()
    # detections.track_id is null in these dirs (binding lives in the
    # tracklets span stage) — stage-1 supply counts DETECTIONS near the
    # rim; per-track binding is the stage-2 wiring.
    order = np.argsort(ts, kind="stable")
    ts, cls, x1, y1, x2, y2, conf = (
        a[order] for a in (ts, cls, x1, y1, x2, y2, conf))

    rows = []
    supply_2d = {r: 0 for r in RADII_FT}
    supply_x = {r: 0 for r in RADII_FT}
    anchors_with_rim = anchors_with_tracks = 0
    for b, _jersey in anchors:
        lo = (b - SHOT_WINDOW_BINS) * BIN_MS
        hi = (b + SHOT_WINDOW_BINS + 1) * BIN_MS
        i, j = bisect_left(ts, lo), bisect_left(ts, hi)
        w_cls, w_conf = cls[i:j], conf[i:j]
        rims = (w_cls == CLS_RIM) & (w_conf >= RIM_CONF_MIN)
        people = w_cls == CLS_PERSON
        row: dict = {"anchor_bin": b, "rim_dets": int(rims.sum()),
                     "person_dets": int(people.sum())}
        if rims.any():
            anchors_with_rim += 1
            rim_cx = float(np.median((x1[i:j][rims] + x2[i:j][rims]) / 2))
            rim_cy = float(np.median((y1[i:j][rims] + y2[i:j][rims]) / 2))
            rim_w = float(np.median(x2[i:j][rims] - x1[i:j][rims]))
            ft_per_px = RIM_TRUE_FT / max(rim_w, 1e-6)
            row["rim_w_px"] = round(rim_w, 1)
            row["ft_per_px"] = round(ft_per_px, 4)
            if people.any():
                anchors_with_tracks += 1
                px = (x1[i:j][people] + x2[i:j][people]) / 2
                py = y2[i:j][people]  # bbox bottom = feet
                w_ts = ts[i:j][people]
                d2d = np.hypot(px - rim_cx, py - rim_cy) * ft_per_px
                dx = np.abs(px - rim_cx) * ft_per_px
                row["min_d2d_ft"] = round(float(d2d.min()), 1)
                row["min_dx_ft"] = round(float(dx.min()), 1)
                for radius in RADII_FT:
                    # persistence gate: candidate must appear within the
                    # radius in >= MIN_FRAMES distinct frames
                    f2d = np.unique(w_ts[d2d <= radius]).size
                    fx = np.unique(w_ts[dx <= radius]).size
                    row[f"frames_2d_{radius:g}"] = int(f2d)
                    row[f"frames_x_{radius:g}"] = int(fx)
                    supply_2d[radius] += int(f2d >= MIN_FRAMES)
                    supply_x[radius] += int(fx >= MIN_FRAMES)
        rows.append(row)

    n = max(len(anchors), 1)
    out = {
        "game": args.game_dir.name,
        "anchors": len(anchors),
        "anchors_with_rim": anchors_with_rim,
        "anchors_with_rim_and_tracks": anchors_with_tracks,
        "supply_2d": {str(r): round(v / n, 4) for r, v in supply_2d.items()},
        "supply_x_only": {str(r): round(v / n, 4)
                          for r, v in supply_x.items()},
        "median_frames_within_10ft_x": float(np.median(
            [r.get("frames_x_10", 0) for r in rows])),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("game", "anchors", "anchors_with_rim",
                       "anchors_with_rim_and_tracks", "supply_2d",
                       "supply_x_only", "median_frames_within_10ft_x")}))


if __name__ == "__main__":
    main()
