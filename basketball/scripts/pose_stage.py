"""Extremities stage: batched KeypointRCNN over ambiguity windows.

Design decision 08-05: locking must consider extremities, not just the
body center. This stage supplies them where they matter — the frames
where identity is actually ambiguous: rows the quark layer flagged
contested/disputed (piles, near-ambiguous claims), plus shot-anchor
windows. Full-game pose is not affordable (~3fps unbatched on GB10,
people_track_video --pose measurement); ambiguity windows are ~10-20%
of a game and get per-interval PyAV seeks, never cv2 (the AV1 holdouts
silently read-fail under cv2.VideoCapture — mine_broadcast.py:274).

Output stage `extremities`: person box + 17 COCO keypoints per row,
consumed by (a) the extremity-consistency probe over remaining swap
conflicts and (b) the quark engine's limb-continuity claim cost.

Usage (inside cvbench, GPU):
    python scripts/pose_stage.py --video "<file>" \
        --game-dir /work/out-harvest/cal_fsu_v3real \
        --binding-stage /work/models/quark-v1/cal_full_v2/quark_binding \
        --out-root /work/models/quark-v1/cal_full_v2 \
        --report /work/models/quark-v1/pose_cal.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from montehall_cv.brain.dataset import BIN_MS
from montehall_cv.brain.eval_slots import SHOT_WINDOW_BINS, _shot_anchors
from montehall_cv.store.artifacts import ArtifactWriter, read_stage, stage_complete

PAD_MS = 250.0
POSE_SCORE_MIN = 0.7  # matches the render's confidence bar
BATCH = 8

POSE_SCHEMA = pa.schema([
    ("ts_ms", pa.int64()),
    ("x1", pa.float32()),
    ("y1", pa.float32()),
    ("x2", pa.float32()),
    ("y2", pa.float32()),
    ("score", pa.float32()),
    ("kp", pa.list_(pa.float32())),  # 17 * (x, y, score)
])


def merge_intervals(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


MAX_FLAG_COVER = 0.5  # if flagged rows window more than half the game,
                      # the flags are uninformative AS WINDOWS (measured
                      # 08-05: margin-band proximity is near-constant on
                      # broadcast — contested-only windows covered 96%
                      # of Cal) and pose falls back to shot anchors only


def ambiguity_windows(game_dir: Path, binding_stage: Path) -> list[tuple[float, float]]:
    t = read_stage(binding_stage)
    ts_all = t.column("ts_ms").to_numpy()
    flagged = (t.column("contested").to_numpy()
               | t.column("disputed").to_numpy())
    anchor_spans: list[tuple[float, float]] = []
    if stage_complete(game_dir / "tokens"):
        tokens = read_stage(game_dir / "tokens").to_pylist()
        for b, _jersey in _shot_anchors(tokens):
            anchor_spans.append(((b - SHOT_WINDOW_BINS) * BIN_MS,
                                 (b + SHOT_WINDOW_BINS + 1) * BIN_MS))
    flag_spans = [(ts - PAD_MS, ts + PAD_MS)
                  for ts in ts_all[flagged].tolist()]
    merged = merge_intervals(flag_spans + anchor_spans)
    game_span = float(ts_all.max() - ts_all.min()) if len(ts_all) else 0.0
    covered = sum(b - a for a, b in merged)
    if game_span and covered > MAX_FLAG_COVER * game_span:
        print(f"POSE window guard: flagged windows cover "
              f"{covered / game_span:.0%} of the game — falling back to "
              f"shot-anchor windows only", flush=True)
        return merge_intervals(anchor_spans)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--game-dir", type=Path, required=True)
    ap.add_argument("--binding-stage", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=2,
                    help="decode stride inside windows (2 at 60fps "
                         "= 30fps effective pose cadence)")
    args = ap.parse_args()

    import av
    import torch
    from torchvision.models.detection import (
        KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn)

    # gate on torch.cuda, NEVER nvidia-smi memory.free (GB10 unified
    # memory reports "[N/A]" — 5b6445bca)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1).to(dev).eval()

    windows = ambiguity_windows(args.game_dir, args.binding_stage)
    total_s = sum(b - a for a, b in windows) / 1000
    print(f"POSE windows={len(windows)} covering {total_s:.0f}s "
          f"dev={dev}", flush=True)

    writer = ArtifactWriter(args.out_root / "extremities", POSE_SCHEMA)
    container = av.open(str(args.video))
    stream = container.streams.video[0]
    tb = stream.time_base

    batch_ts: list[int] = []
    batch_t: list = []
    n_frames = n_rows = 0
    t_start = time.monotonic()

    def flush() -> None:
        nonlocal n_rows
        if not batch_t:
            return
        with torch.no_grad():
            preds = model([t.to(dev) for t in batch_t])
        for ts, pred in zip(batch_ts, preds, strict=True):
            keep = pred["scores"] >= POSE_SCORE_MIN
            for box, score, kps in zip(
                    pred["boxes"][keep].cpu().numpy(),
                    pred["scores"][keep].cpu().numpy(),
                    pred["keypoints"][keep].cpu().numpy()):
                writer.add({
                    "ts_ms": int(ts),
                    "x1": float(box[0]), "y1": float(box[1]),
                    "x2": float(box[2]), "y2": float(box[3]),
                    "score": float(score),
                    "kp": kps.reshape(-1).astype(np.float32).tolist(),
                })
                n_rows += 1
        batch_ts.clear()
        batch_t.clear()

    for w0, w1 in windows:
        container.seek(int(w0 / 1000 / tb), stream=stream)
        seen = 0
        for frame in container.decode(stream):
            stamp = frame.pts if frame.pts is not None else frame.dts
            if stamp is None:
                continue
            ts = float(stamp * tb) * 1000
            if ts < w0 - 20:
                continue
            if ts > w1:
                break
            if seen % args.stride:
                seen += 1
                continue
            seen += 1
            rgb = frame.to_ndarray(format="rgb24")
            batch_ts.append(int(ts))
            batch_t.append(torch.from_numpy(rgb).permute(2, 0, 1)
                           .float().div(255))
            n_frames += 1
            if len(batch_t) >= BATCH:
                flush()
            if n_frames % 500 == 0:
                rate = n_frames / max(time.monotonic() - t_start, 1e-9)
                print(f"POSE progress frames={n_frames} rows={n_rows} "
                      f"ts={int(ts)} fps={rate:.1f}", flush=True)
    flush()
    writer.close()
    container.close()

    report = {
        "windows": len(windows),
        "window_seconds": round(total_s, 1),
        "frames_posed": n_frames,
        "pose_rows": n_rows,
        "elapsed_s": round(time.monotonic() - t_start, 1),
        "fps": round(n_frames / max(time.monotonic() - t_start, 1e-9), 2),
        "stage_dir": str(args.out_root / "extremities"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print("POSE_STAGE_DONE " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
