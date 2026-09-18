"""Ball estimator eval: invisible-bin position error + real anchor supply.

Two modes, auto-detected per game dir:

- SIM (sim_states present): position error in FEET — overall, on bins
  with NO ball observation (the "where is it when invisible" number),
  and per mode — plus mode accuracy, holder accuracy, and two
  observation-only baselines the model must beat where it matters:
  sticky (carry the last observed position) and interp (offline linear
  interpolation between observations).
- REAL (no sim_states): graded at external anchors only, no ball truth
  exists. (a) rim-proximity: fraction of shooting pbp anchors where the
  predicted ball comes within RIM_NEAR_FT of an observed rim position
  inside the anchor window, vs raw ball observations doing the same.
  (b) candidate supply: fraction of anchors with a present track within
  D ft of the predicted ball — against the eval_slots evidence
  definition (any F_BALL_DIST/F_CONTROL_DIST > 0 in-window) that
  produced the 18.4% mass-eval supply ceiling.

CLI:
    python -m montehall_cv.brain.eval_ball --weights .../ball-v1/last.pt \
        --game-dirs /work/sim/sim_0007_045 --out .../ball-v1/eval-sim
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from montehall_cv.brain.dataset import MODE_IGNORE, MODES
from montehall_cv.brain.dataset_ball import (
    B_AGE,
    B_LAST_X,
    B_LAST_Y,
    B_PRESENT,
    B_X,
    B_Y,
    HOLDER_IGNORE,
    load_ball_features,
    load_ball_game,
)
from montehall_cv.brain.dataset_slots import (
    F_BALL_DIST,
    F_CONTROL_DIST,
    F_PRESENT,
    F_X,
    F_Y,
    featurize_tracks,
)
from montehall_cv.brain.eval_slots import (
    SHOT_WINDOW_BINS,
    _shot_anchors,
    _tracks_near_anchor,
)
from montehall_cv.brain.tokens import CH_RIM
from montehall_cv.store.artifacts import stage_complete

RIM_NEAR_FT = 8.0
SUPPLY_RADII_FT = (4.0, 6.0, 8.0, 10.0)


def _predict(weights: Path, feats: dict, device: str | None = None):
    import torch

    from montehall_cv.brain.train_ball import build_model

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta_path = Path(weights).parent / "train_meta.json"
    bidirectional = (json.loads(meta_path.read_text()).get(
        "bidirectional", False) if meta_path.exists() else False)
    model = build_model(bidirectional).to(dev)
    model.load_state_dict(torch.load(weights, map_location=dev,
                                     weights_only=True))
    model.eval()
    with torch.no_grad():
        mode_logits, pos, holder_logits = model(
            torch.from_numpy(feats["x_tracks"]).to(dev),
            torch.from_numpy(feats["x_global"]).to(dev),
            torch.from_numpy(feats["x_ball"]).to(dev))
        return {
            "mode": mode_logits.argmax(-1).cpu().numpy(),
            "pos": np.clip(pos.cpu().numpy(), 0.0, 1.0),
            "holder": holder_logits.argmax(-1).cpu().numpy(),
        }


def _err_ft(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.hypot((pred[:, 0] - truth[:, 0]) * 94.0,
                    (pred[:, 1] - truth[:, 1]) * 50.0)


def sticky_ball_baseline(x_ball: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pred [T,2], valid [T]) — carry the last observed position."""
    valid = x_ball[:, B_AGE] < 1.0
    return x_ball[:, (B_LAST_X, B_LAST_Y)].astype(np.float32), valid


def interp_ball_baseline(x_ball: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pred [T,2], valid [T]) — offline linear interp between observations."""
    obs = np.flatnonzero(x_ball[:, B_PRESENT] > 0)
    t_n = x_ball.shape[0]
    pred = np.zeros((t_n, 2), dtype=np.float32)
    if obs.size == 0:
        return pred, np.zeros(t_n, dtype=bool)
    bins = np.arange(t_n)
    for axis, col in ((0, B_X), (1, B_Y)):
        pred[:, axis] = np.interp(bins, obs, x_ball[obs, col])
    valid = (bins >= obs[0]) & (bins <= obs[-1])
    return pred, valid


def _pos_report(pred: np.ndarray, valid: np.ndarray, truth_xy: np.ndarray,
                sup: np.ndarray, invisible: np.ndarray) -> dict:
    err = _err_ft(pred, np.nan_to_num(truth_xy))
    scoped = sup & valid
    inv = scoped & invisible
    return {
        "coverage": round(float(scoped.sum() / max(sup.sum(), 1)), 4),
        "mae_ft": (round(float(err[scoped].mean()), 2)
                   if scoped.any() else None),
        "mae_ft_invisible": (round(float(err[inv].mean()), 2)
                             if inv.any() else None),
    }


EVAL_DROPOUT_SEED = 99  # fixed so every arm sees the identical degraded stream


def eval_sim_game(game_dir: Path, weights: Path,
                  device: str | None = None,
                  ball_dropout: float = 0.0) -> dict:
    game = load_ball_game(game_dir, ball_dropout, seed=EVAL_DROPOUT_SEED)
    assert game is not None, f"not ball-evaluable: {game_dir}"
    pred = _predict(weights, game, device)
    sup = game["pos_mask"]
    invisible = ~game["obs_mask"]
    err = _err_ft(pred["pos"], np.nan_to_num(game["ball_xy"]))

    by_mode = {}
    for m_i, m in enumerate(MODES):
        sel = sup & (game["mode"] == m_i)
        by_mode[m] = {
            "bins": int(sel.sum()),
            "mae_ft": round(float(err[sel].mean()), 2) if sel.any() else None,
            "mae_ft_invisible": (
                round(float(err[sel & invisible].mean()), 2)
                if (sel & invisible).any() else None),
        }

    mode_sup = game["mode"] != MODE_IGNORE
    holder_sup = game["holder"] != HOLDER_IGNORE
    result = {
        "game": Path(game_dir).name, "mode": "sim",
        "bins": int(sup.shape[0]),
        "eval_ball_dropout": ball_dropout,
        "ball_observed_frac": round(float(game["obs_mask"].mean()), 4),
        "model": {
            "mae_ft": round(float(err[sup].mean()), 2),
            "mae_ft_invisible": round(float(err[sup & invisible].mean()), 2),
            "median_ft_invisible": round(
                float(np.median(err[sup & invisible])), 2),
            "by_mode": by_mode,
            "mode_acc": round(
                float((pred["mode"] == game["mode"])[mode_sup].mean()), 4),
            "holder_acc": round(
                float((pred["holder"] == game["holder"])[holder_sup].mean()),
                4),
        },
        "baselines": {},
    }
    for name, (b_pred, b_valid) in (
            ("sticky-last-obs", sticky_ball_baseline(game["x_ball"])),
            ("linear-interp", interp_ball_baseline(game["x_ball"]))):
        result["baselines"][name] = _pos_report(
            b_pred, b_valid, game["ball_xy"], sup, invisible)
    return result


def _rim_positions(tokens: list[dict]) -> np.ndarray:
    """[N, 3] (bin, x_ft, y_ft) for every rim token with court coords."""
    from montehall_cv.brain.dataset import _bin_of

    rows = [(_bin_of(t["t_ms"]), t["court_x"], t["court_y"])
            for t in tokens
            if t["channel"] == CH_RIM and t["court_x"] is not None]
    return (np.array(rows, dtype=np.float32)
            if rows else np.zeros((0, 3), dtype=np.float32))


def eval_real_game(game_dir: Path, weights: Path,
                   device: str | None = None) -> dict:
    feats = load_ball_features(game_dir)
    assert feats is not None, f"no tokens stage: {game_dir}"
    tokens = feats["tokens"]
    anchors = _shot_anchors(tokens)
    pred = _predict(weights, feats, device)
    t_n = pred["pos"].shape[0]
    rims = _rim_positions(tokens)
    pred_ft = pred["pos"] * np.array([94.0, 50.0], dtype=np.float32)
    obs_ft = feats["x_ball"][:, (B_X, B_Y)] * np.array(
        [94.0, 50.0], dtype=np.float32)
    obs_present = feats["x_ball"][:, B_PRESENT] > 0

    result: dict = {
        "game": Path(game_dir).name, "mode": "real",
        "truth_shots": len(anchors),
        "ball_observed_frac": round(float(obs_present.mean()), 4),
        "rim_near_ft": RIM_NEAR_FT,
    }
    if not anchors:
        return result

    rim_hits_pred = rim_hits_obs = 0
    anchors_with_rim = 0
    supply_pred = {d: 0 for d in SUPPLY_RADII_FT}
    supply_evidence = 0  # the eval_slots definition (the 18.4% metric)
    anchors_no_track = 0
    for b, _jersey in anchors:
        lo = max(0, b - SHOT_WINDOW_BINS)
        hi = min(t_n, b + SHOT_WINDOW_BINS + 1)
        window = np.arange(lo, hi)
        rims_w = rims[(rims[:, 0] >= lo) & (rims[:, 0] < hi)]
        if rims_w.shape[0]:
            anchors_with_rim += 1
            d_pred = np.hypot(pred_ft[rims_w[:, 0].astype(int), 0]
                              - rims_w[:, 1],
                              pred_ft[rims_w[:, 0].astype(int), 1]
                              - rims_w[:, 2])
            if (d_pred <= RIM_NEAR_FT).any():
                rim_hits_pred += 1
            obs_bins = rims_w[:, 0].astype(int)
            sel = obs_present[obs_bins]
            d_obs = np.hypot(obs_ft[obs_bins, 0] - rims_w[:, 1],
                             obs_ft[obs_bins, 1] - rims_w[:, 2])
            if sel.any() and (d_obs[sel] <= RIM_NEAR_FT).any():
                rim_hits_obs += 1

        local = _tracks_near_anchor(tokens, b)
        if not local:
            anchors_no_track += 1
            continue
        x_tracks = featurize_tracks(tokens, local)
        w = window[window < x_tracks.shape[1]]
        evidence = (
            (x_tracks[:, w, F_BALL_DIST] > 0)
            | (x_tracks[:, w, F_CONTROL_DIST] > 0)).any()
        supply_evidence += int(evidence)
        present = x_tracks[:, w, F_PRESENT] > 0
        tx = x_tracks[:, w, F_X] * 94.0
        ty = x_tracks[:, w, F_Y] * 50.0
        d = np.hypot(tx - pred_ft[w, 0], ty - pred_ft[w, 1])
        d = np.where(present, d, np.inf)
        for radius in SUPPLY_RADII_FT:
            if (d <= radius).any():
                supply_pred[radius] += 1

    n = len(anchors)
    result.update({
        "rim_proximity": {
            "anchors_with_rim_obs": anchors_with_rim,
            "pred_hit": (round(rim_hits_pred / anchors_with_rim, 4)
                         if anchors_with_rim else None),
            "raw_obs_hit": (round(rim_hits_obs / anchors_with_rim, 4)
                            if anchors_with_rim else None),
        },
        "candidate_supply": {
            "anchors": n,
            "anchors_no_track_in_window": anchors_no_track,
            "evidence_definition_eval_slots": round(supply_evidence / n, 4),
            "pred_ball_within_ft": {
                str(radius): round(hits / n, 4)
                for radius, hits in supply_pred.items()},
        },
    })
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--game-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ball-dropout", type=float, default=0.0,
                    help="degrade sim eval streams to deployment density")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for game_dir in args.game_dirs:
        if stage_complete(game_dir / "sim_states"):
            result = eval_sim_game(game_dir, args.weights, args.device,
                                   args.ball_dropout)
        else:
            result = eval_real_game(game_dir, args.weights, args.device)
        (args.out / f"{game_dir.name}.json").write_text(
            json.dumps(result, indent=2))
        print(f"BALL-EVAL: {json.dumps(result)}", flush=True)
    print("BALL_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
