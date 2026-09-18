"""WASB zero-shot probe over shot windows — stage-1 of shot-window ball.

The frame detector produced ZERO pixel-space ball detections in all 313
shot windows of both holdouts (anchor_outage_probe, 2026-08-03). WASB
(HRNet heatmap tracker, 3-frame input, MIT, basketball weights staged
since 2026-07-08 but never run) is the motion-based alternative. This
probe answers one question: does WASB see the ball inside shot windows
where the frame detector saw nothing?

Per anchor: sample TRIPLETS_PER_WINDOW consecutive-frame triplets across
the +-2s window, run WASB, record the max heatmap score and its pixel.
No tracking, no court mapping — presence measurement only.

Usage (inside cvbench, GPU):
    python scripts/wasb_shot_window_probe.py \
        --video "/work/pairing/videos/<file>.mp4" \
        --game-dir /work/out-harvest/cal_fsu_v3real \
        --wasb-root /work/wasb --out /work/models/ball-v1/wasb_cal.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from montehall_cv.brain.dataset import BIN_MS
from montehall_cv.brain.eval_slots import _shot_anchors
from montehall_cv.store.artifacts import read_stage

TRIPLET_OFFSETS_S = (-2.0, -1.0, 0.0, 1.0, 2.0)
SCORE_THRESHOLDS = (0.3, 0.5)

# canonical home since 2026-08-25 (structural fix A): pipeline/flight.py
from montehall_cv.pipeline.flight import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_wasb,
)


def triplet_tensor(container, stream, t_s: float, wh: tuple[int, int],
                   stride: int):
    """3 frames starting at t_s, `stride` apart — PyAV, never cv2 capture
    (these replays are AV1; opencv's bundled ffmpeg silently read-fails,
    the mine_broadcast.py:274 trap)."""
    import cv2

    tb = stream.time_base
    container.seek(int(max(t_s, 0.0) / tb), stream=stream)
    frames, seen = [], 0
    for frame in container.decode(stream):
        stamp = frame.pts if frame.pts is not None else frame.dts
        if stamp is None:
            continue
        if not frames and not seen and float(stamp * tb) < t_s - 0.02:
            continue
        if seen % stride == 0:
            img = cv2.resize(frame.to_ndarray(format="rgb24"), wh)
            img = img.astype(np.float32) / 255
            frames.append((img - IMAGENET_MEAN) / IMAGENET_STD)
            if len(frames) == 3:
                break
        seen += 1
    if len(frames) < 3:
        return None
    return np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)


from montehall_cv.pipeline.flight import heatmap_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--game-dir", type=Path, required=True)
    ap.add_argument("--wasb-root", type=Path, default=Path("/work/wasb"))
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import av
    import torch

    weights = args.weights or args.wasb_root / "wasb_basketball_best.pth.tar"
    model, dev, model_cfg = load_wasb(args.wasb_root, weights)
    wh = (model_cfg["inp_width"], model_cfg["inp_height"])

    tokens = read_stage(args.game_dir / "tokens").to_pylist()
    anchors = _shot_anchors(tokens)
    container = av.open(str(args.video))
    stream = container.streams.video[0]
    fps = float(stream.average_rate or 30.0)
    stride = 2 if fps > 45 else 1  # mimic WASB's ~30fps training cadence
    print(f"WASB-PROBE: codec={stream.codec_context.name} fps={fps:.2f} "
          f"stride={stride}", flush=True)

    rows = []
    n_decoded = 0
    for n_done, (b, _jersey) in enumerate(anchors):
        t0 = b * BIN_MS / 1000.0
        best = {"score": 0.0}
        for off in TRIPLET_OFFSETS_S:
            x = triplet_tensor(container, stream, t0 + off, wh, stride)
            if x is None:
                continue
            n_decoded += 1
            with torch.no_grad():
                preds = model(torch.from_numpy(x[None]).to(dev))
            hm = heatmap_of(preds)
            mid = hm[min(1, hm.shape[0] - 1)]
            score = float(mid.max())
            if score > best["score"]:
                yx = np.unravel_index(int(mid.argmax()), mid.shape)
                best = {"score": round(score, 4), "offset_s": off,
                        "px_x": int(yx[1]), "px_y": int(yx[0])}
        rows.append({"anchor_bin": b, **best})
        if (n_done + 1) % 25 == 0:
            print(f"WASB-PROBE: {n_done + 1}/{len(anchors)}", flush=True)
    container.close()

    n = max(len(anchors), 1)
    scores = [r["score"] for r in rows]
    out = {
        "game": args.game_dir.name, "video": args.video.name,
        "anchors": len(anchors),
        "triplets_decoded": n_decoded,
        "found_frac": {str(t): round(sum(s >= t for s in scores) / n, 4)
                       for t in SCORE_THRESHOLDS},
        "score_median": round(float(np.median(scores)), 4),
        "score_p90": round(float(np.percentile(scores, 90)), 4),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print("WASB_PROBE_RESULT " + json.dumps(
        {k: out[k] for k in ("game", "anchors", "found_frac",
                             "score_median", "score_p90")}), flush=True)


if __name__ == "__main__":
    main()
