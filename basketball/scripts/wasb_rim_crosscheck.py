"""Discriminating control for the WASB shot-window probe.

A 94% found-fraction could mean "WASB tracks the ball" or "WASB fires
confidently on some blob in every frame". The control: at/after the
anchor moment the true ball converges to the rim, so for anchors whose
best triplet sits at offset >= 0, the WASB peak should be NEAR the rim
— and matched (anchor, peak) pairs should be closer to that anchor's
rim than SHUFFLED pairs are. If matched ~= shuffled, the peaks carry no
ball signal and the found-fraction is noise.

Peak px is in WASB input space (512x288); rim centers come from rim
detections at native res — the scale factor is derived per game.

Usage (inside cvbench):
    python scripts/wasb_rim_crosscheck.py /work/out-harvest/<game_dir> \
        --wasb-json /work/models/ball-v1/wasb_<g>.json \
        --out /work/models/ball-v1/wasb_rimcheck_<g>.json
"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from pathlib import Path

import numpy as np

from montehall_cv.brain.dataset import BIN_MS
from montehall_cv.brain.eval_slots import SHOT_WINDOW_BINS
from montehall_cv.store.artifacts import read_stage

CLS_RIM = 3
RIM_CONF_MIN = 0.3
SCORE_MIN = 0.5
WASB_W, WASB_H = 512, 288
NEAR_PX = (60, 100, 150)
SHUFFLE_SEED = 7


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game_dir", type=Path)
    ap.add_argument("--wasb-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--native-w", type=int, default=1280)
    args = ap.parse_args()

    scale = args.native_w / WASB_W
    wasb = json.loads(args.wasb_json.read_text())
    det = read_stage(args.game_dir / "detections")
    ts = det.column("ts_ms").to_numpy()
    cls = det.column("cls").to_numpy()
    x1 = det.column("x1").to_numpy()
    y1 = det.column("y1").to_numpy()
    x2 = det.column("x2").to_numpy()
    y2 = det.column("y2").to_numpy()
    conf = det.column("conf").to_numpy()
    order = np.argsort(ts, kind="stable")
    ts, cls, x1, y1, x2, y2, conf = (
        a[order] for a in (ts, cls, x1, y1, x2, y2, conf))

    pairs = []  # (peak_x_native, peak_y_native, rim_cx, rim_cy)
    for r in wasb["rows"]:
        if r.get("score", 0.0) < SCORE_MIN or r.get("offset_s", -9) < 0:
            continue
        b = r["anchor_bin"]
        lo = (b - SHOT_WINDOW_BINS) * BIN_MS
        hi = (b + SHOT_WINDOW_BINS + 1) * BIN_MS
        i, j = bisect_left(ts, lo), bisect_left(ts, hi)
        rims = (cls[i:j] == CLS_RIM) & (conf[i:j] >= RIM_CONF_MIN)
        if not rims.any():
            continue
        rim_cx = float(np.median((x1[i:j][rims] + x2[i:j][rims]) / 2))
        rim_cy = float(np.median((y1[i:j][rims] + y2[i:j][rims]) / 2))
        pairs.append((r["px_x"] * scale, r["px_y"] * scale, rim_cx, rim_cy))

    n = len(pairs)
    out: dict = {"game": args.game_dir.name, "pairs": n,
                 "score_min": SCORE_MIN, "offsets": ">=0"}
    if n >= 10:
        arr = np.array(pairs)
        matched = np.hypot(arr[:, 0] - arr[:, 2], arr[:, 1] - arr[:, 3])
        rng = np.random.default_rng(SHUFFLE_SEED)
        perm = rng.permutation(n)
        shuffled = np.hypot(arr[:, 0] - arr[perm, 2],
                            arr[:, 1] - arr[perm, 3])
        out.update({
            "matched_median_px": round(float(np.median(matched)), 1),
            "shuffled_median_px": round(float(np.median(shuffled)), 1),
            "matched_within": {str(p): round(float((matched <= p).mean()), 4)
                               for p in NEAR_PX},
            "shuffled_within": {str(p): round(float((shuffled <= p).mean()), 4)
                                for p in NEAR_PX},
        })
    else:
        out["note"] = "too few qualifying pairs"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print("WASB_RIMCHECK " + json.dumps(out))


if __name__ == "__main__":
    main()
