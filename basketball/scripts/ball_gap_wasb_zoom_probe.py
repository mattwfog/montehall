"""WASB resolution probe over a ball-track supply hole.

The ev13 gap (HS clip, frames ~3402-3498): full-frame WASB at stride 1
found peaks on 1/97 frames and the detector holds only 6 rows >= .3 with
21-36-frame gaps — no segment can form, so the shot's pre-release contact
never exists (spine_coverage_probe, 2026-08-25). Full-frame WASB squeezes
1920px into 512 (3.75x down); a distant ball can vanish at that scale.

This probe measures the one untested sensor axis: EFFECTIVE RESOLUTION.
Per frame in the window it runs the same WASB triplet inference on
  full    the whole frame resized to model input (the shipped pass)
  tiles   a 2x2 grid with overlap, each tile resized to model input
          (~2x effective resolution)
and reports per-frame max scores + best tile peak in frame coordinates,
so "would a zoomed pass supply this gap" is observed before any fusion
work. Read-only; touches no stage artifacts.

Usage (inside cvbench, GPU):
    python scripts/ball_gap_wasb_zoom_probe.py \
        --video /work/hs_clip_20260824.mp4 \
        --frames 3390 3510 \
        --wasb-root /work/wasb \
        --wasb-weights /work/weights/wasb/wasb_basketball_best.pth.tar \
        --out /work/models/quark-v1/ball_gap_wasb_zoom_hsclip.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from montehall_cv.pipeline.flight import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_wasb,
    heatmap_of,
)

TILE_OVERLAP = 0.25   # fraction of tile size shared with the neighbour
SCORE_BANDS = (0.08, 0.2, 0.5)   # WASB_LOW_MIN / mid / strong


def tile_regions(w: int, h: int) -> list[tuple[int, int, int, int]]:
    tw, th = int(w / (2 - TILE_OVERLAP)), int(h / (2 - TILE_OVERLAP))
    xs = (0, w - tw)
    ys = (0, h - th)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def prep(img: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    import cv2

    img = cv2.resize(img, wh)
    return ((img.astype(np.float32) / 255 - IMAGENET_MEAN)
            / IMAGENET_STD).transpose(2, 0, 1)


def peak_of(hm: np.ndarray) -> tuple[float, int, int]:
    yx = np.unravel_index(int(hm.argmax()), hm.shape)
    return float(hm.max()), int(yx[1]), int(yx[0])


def main() -> None:
    import av
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--frames", nargs=2, type=int, required=True)
    ap.add_argument("--wasb-root", type=Path, default=Path("/work/wasb"))
    ap.add_argument("--wasb-weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model, dev, model_cfg = load_wasb(args.wasb_root, args.wasb_weights)
    wh = (model_cfg["inp_width"], model_cfg["inp_height"])
    f0, f1 = args.frames

    container = av.open(str(args.video))
    stream = container.streams.video[0]
    fps = float(stream.average_rate or 30.0)
    regions = None

    rows: list[dict] = []
    buf: list[np.ndarray] = []
    buf_idx: list[int] = []
    for frame in container.decode(stream):
        if frame.time is None:
            continue
        idx = round(frame.time * fps)
        if idx < f0 - 1:
            continue
        if idx > f1 + 1:
            break
        rgb = frame.to_ndarray(format="rgb24")
        if regions is None:
            regions = tile_regions(rgb.shape[1], rgb.shape[0])
        buf.append(rgb)
        buf_idx.append(idx)
        if len(buf) < 3:
            continue
        mid = buf_idx[1]
        views = [(None, buf)]  # full frame
        for r in regions:
            x1, y1, x2, y2 = r
            views.append((r, [b[y1:y2, x1:x2] for b in buf]))
        trips = np.stack([
            np.concatenate([prep(v, wh) for v in view], axis=0)
            for _, view in views])
        with torch.no_grad():
            preds = model(torch.from_numpy(trips).to(dev))
        row = {"frame": mid}
        best_tile = None
        for k, (region, _) in enumerate(views):
            hm = heatmap_of(
                preds[k:k + 1] if not isinstance(preds, dict)
                else {s: v[k:k + 1] for s, v in preds.items()})
            score, hx, hy = peak_of(hm[min(1, hm.shape[0] - 1)])
            if region is None:
                scale = buf[0].shape[1] / wh[0]
                row["full"] = [round(score, 3),
                               int(hx * scale), int(hy * scale)]
            else:
                x1, y1, x2, y2 = region
                sx = (x2 - x1) / wh[0]
                sy = (y2 - y1) / wh[1]
                cand = (round(score, 3),
                        int(x1 + hx * sx), int(y1 + hy * sy))
                if best_tile is None or cand[0] > best_tile[0]:
                    best_tile = cand
        row["tile"] = list(best_tile) if best_tile else None
        rows.append(row)
        buf.pop(0)
        buf_idx.pop(0)
    container.close()

    summary = {}
    for band in SCORE_BANDS:
        summary[f"full_ge_{band}"] = sum(
            1 for r in rows if r["full"][0] >= band)
        summary[f"tile_ge_{band}"] = sum(
            1 for r in rows if r["tile"] and r["tile"][0] >= band)
    report = {"video": str(args.video), "frames": [f0, f1],
              "n_frames": len(rows), "summary": summary, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"n_frames": len(rows), **summary}, indent=2))


if __name__ == "__main__":
    main()
