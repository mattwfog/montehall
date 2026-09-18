"""Brain v0 holdout eval: peak-extracted events vs aligned PBP shots.

Event-level semantics match the e2e scorer's convention (and the retired
eval.pbp_score's): greedy 1:1 nearest-match within a tolerance, reported
at +-2s and +-5s. Truth comes from the
holdout's pbp_anchor tokens (their sole purpose); features never see
them (dataset contract). Thresholds are FIXED before looking at the
holdout — the sweep is reported at face value, never tuned per game.

CLI:
    python -m montehall_cv.brain.eval_v0 --weights /work/models/brain-v0/last.pt \
        --game-dirs /work/out-eval-v3/clemson_duke_acc26 --out /work/models/brain-v0/eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from montehall_cv.brain.dataset import (
    BIN_MS,
    CHANNEL_SETS,
    featurize,
    shot_times_ms,
)
from montehall_cv.brain.train_v0 import build_model
from montehall_cv.store.artifacts import read_stage

THRESHOLDS = (0.3, 0.5, 0.7)
MIN_SEP_BINS = 8  # 4s between distinct predicted events
TOLERANCES_S = (2.0, 5.0)


def extract_events(probs: np.ndarray, threshold: float) -> list[int]:
    """Local maxima above threshold with a minimum separation -> bin idx."""
    peaks: list[int] = []
    for i in range(len(probs)):
        if probs[i] < threshold:
            continue
        lo, hi = max(0, i - MIN_SEP_BINS), min(len(probs), i + MIN_SEP_BINS + 1)
        if probs[i] >= probs[lo:hi].max() and (
                not peaks or i - peaks[-1] >= MIN_SEP_BINS):
            peaks.append(i)
    return peaks


def greedy_match(pred_ms: list[int], truth_ms: list[int],
                 tol_s: float) -> int:
    """Greedy 1:1 nearest pairing within tolerance; returns match count."""
    pairs = sorted(
        (abs(p - t), pi, ti)
        for pi, p in enumerate(pred_ms) for ti, t in enumerate(truth_ms)
        if abs(p - t) <= tol_s * 1000
    )
    used_p: set[int] = set()
    used_t: set[int] = set()
    matches = 0
    for _gap, pi, ti in pairs:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        matches += 1
    return matches


def eval_game(game_dir: Path, weights: Path, device: str | None = None,
              channels_name: str = "all") -> dict:
    import torch

    tokens = read_stage(Path(game_dir) / "tokens").to_pylist()
    x = featurize(tokens, CHANNEL_SETS[channels_name])
    truth = [t for t, _made in shot_times_ms(tokens)]
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(dev)
    model.load_state_dict(torch.load(weights, map_location=dev,
                                     weights_only=True))
    model.eval()
    with torch.no_grad():
        shot_l, _made_l, _mode_l = model(
            torch.from_numpy(x).unsqueeze(0).to(dev))
        probs = torch.sigmoid(shot_l[0]).cpu().numpy()

    result: dict = {"game": Path(game_dir).name, "truth_shots": len(truth),
                    "bins": len(x), "channels": channels_name,
                    "by_threshold": {}}
    for thr in THRESHOLDS:
        pred = [b * BIN_MS for b in extract_events(probs, thr)]
        row: dict = {"predicted": len(pred)}
        for tol in TOLERANCES_S:
            m = greedy_match(pred, truth, tol)
            row[f"recall@{tol:g}s"] = round(m / len(truth), 3) if truth else None
            row[f"precision@{tol:g}s"] = round(m / len(pred), 3) if pred else None
        result["by_threshold"][str(thr)] = row
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--game-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--channels", choices=sorted(CHANNEL_SETS), default="all")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for game_dir in args.game_dirs:
        result = eval_game(game_dir, args.weights, device=args.device,
                           channels_name=args.channels)
        path = args.out / f"{game_dir.name}.json"
        path.write_text(json.dumps(result, indent=2))
        print(f"BRAIN-EVAL: {json.dumps(result)}", flush=True)
    print("BRAIN_V0_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
