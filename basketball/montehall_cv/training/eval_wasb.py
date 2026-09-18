"""WASB ball-detector coverage on our footage (D7 smoke).

Drives the WASB HRNet (MIT, basketball checkpoint) directly — build_model +
checkpoint + sigmoid + threshold — bypassing its hydra/dataset machinery.
Protocol mirrors eval_finetune: sampled frames, ball coverage = fraction of
sampled windows whose CENTER heatmap peaks above threshold, swept across
thresholds so calibration is visible. WASB consumes 3 CONSECUTIVE frames per
window; we sample window starts every --sample-every frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CONF_SWEEP = (0.3, 0.4, 0.5, 0.6)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def evaluate(video: Path, wasb_src: Path, weights: Path, model_cfg: Path,
             sample_every: int = 10, max_windows: int = 900,
             batch: int = 8) -> dict:
    import cv2
    import numpy as np
    import torch
    import yaml
    from omegaconf import OmegaConf

    sys.path.insert(0, str(wasb_src))
    from models import build_model

    from montehall_cv.pipeline.video import decode_frames

    with model_cfg.open() as fh:
        # WASB's HRNet reads nested keys by attribute (cfg.MODEL.EXTRA),
        # so the dict must be an OmegaConf node, not a plain dict.
        cfg = OmegaConf.create({"model": yaml.safe_load(fh)})
    inp_w, inp_h = cfg["model"]["inp_width"], cfg["model"]["inp_height"]
    frames_in = cfg["model"]["frames_in"]

    device = torch.device("cuda")
    model = build_model(cfg).to(device).eval()
    ckpt = torch.load(weights, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

    def prep(frame_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_rgb, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR)
        chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        return (chw - mean) / std

    started = time.monotonic()
    window: list[np.ndarray] = []
    batched: list[np.ndarray] = []
    peak_scores: list[float] = []

    def flush() -> None:
        if not batched:
            return
        tensor = torch.from_numpy(np.stack(batched)).to(device)
        with torch.no_grad():
            out = model(tensor)
        # scale 0 -> (B, frames_out, H, W) logits; center heatmap = index 1
        hms = torch.sigmoid(out[0])
        centers = hms[:, frames_in // 2]
        peak_scores.extend(centers.amax(dim=(1, 2)).float().cpu().tolist())
        batched.clear()

    n_seen = 0
    for frame in decode_frames(video):
        if len(peak_scores) + len(batched) >= max_windows:
            break
        window.append(prep(frame.image))
        if len(window) > frames_in:
            window.pop(0)
        n_seen += 1
        if len(window) == frames_in and (n_seen - frames_in) % sample_every == 0:
            batched.append(np.concatenate(window, axis=0))
            if len(batched) >= batch:
                flush()
    flush()

    scores = np.array(peak_scores)
    return {
        "windows_evaluated": int(len(scores)),
        "ball_coverage": {
            f"conf_{c}": round(float((scores >= c).mean()), 3) for c in CONF_SWEEP
        },
        "max_peak": round(float(scores.max()), 3) if len(scores) else None,
        "median_peak": round(float(np.median(scores)), 3) if len(scores) else None,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--wasb-src", type=Path, required=True,
                        help="path to the WASB repo's src/ dir")
    parser.add_argument("--weights", type=Path, required=True,
                        help="wasb_basketball_best.pth.tar")
    parser.add_argument("--model-cfg", type=Path, required=True,
                        help="WASB configs/model/wasb.yaml")
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--max-windows", type=int, default=900)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.video, args.wasb_src, args.weights,
                              args.model_cfg, args.sample_every,
                              args.max_windows), indent=2))


if __name__ == "__main__":
    main()
