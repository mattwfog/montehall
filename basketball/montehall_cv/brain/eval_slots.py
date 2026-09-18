"""Slot estimator eval: identity accuracy-at-coverage + the sticky baseline.

Two modes, auto-detected per game dir:

- SIM (sim_identity present): per-(track, bin) identity accuracy against
  the replayed permutation, accuracy-at-coverage over max-prob
  thresholds, swap-recovery latency, and the sticky-last-jersey-read
  baseline the model must beat.
- REAL (no sim_identity): shot-attribution precision-at-coverage — for
  each pbp anchor with a parseable shooter_jersey, attribute the shot to
  the nearest-ball track (policy nearest-ball-v1, same policy for model
  and baseline) and compare the track's believed jersey to truth.

Abstention = max-prob below threshold (never a forced guess). Thresholds
are fixed here, never tuned per game.

CLI:
    python -m montehall_cv.brain.eval_slots --weights .../slots-v1/last.pt \
        --game-dirs /work/sim/sim_0007_045 --out .../slots-v1/eval-sim
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from montehall_cv.brain.dataset import BIN_MS, featurize
from montehall_cv.brain.dataset_slots import (
    F_BALL_DIST,
    F_CONTROL_DIST,
    F_PRESENT,
    F_READ,
    F_READ_VALUE,
    LABEL_IGNORE,
    MAX_JERSEY,
    featurize_tracks,
    load_slot_game,
    parse_jersey,
)
from montehall_cv.brain.tokens import CH_PBP_ANCHOR, CH_PLAYER
from montehall_cv.store.artifacts import read_stage, stage_complete
from montehall_cv.brain.train_slots import build_model

THRESHOLDS = (0.1, 0.15, 0.3, 0.5, 0.7, 0.9)
SHOT_WINDOW_BINS = 4  # +-2s around the anchor for shooter-track search
RECOVERY_CAP_BINS = 120


def _predict(weights: Path, game: dict, roster: np.ndarray,
             device: str | None = None) -> np.ndarray:
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(dev)
    model.load_state_dict(torch.load(weights, map_location=dev,
                                     weights_only=True))
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(game["x_tracks"]).to(dev),
            torch.from_numpy(game["x_global"]).to(dev),
            torch.from_numpy(roster).to(dev))
        return torch.softmax(logits, dim=-1).cpu().numpy()  # [K,T,R]


def sticky_baseline(x_tracks: np.ndarray,
                    roster: np.ndarray) -> np.ndarray:
    """[K, T] roster-slot prediction = last jersey read seen on the track
    (-1 before any read). The propagation-free control."""
    jersey_to_slot = {int(j): s for s, j in enumerate(roster)}
    k_n, t_n = x_tracks.shape[0], x_tracks.shape[1]
    pred = np.full((k_n, t_n), -1, dtype=np.int64)
    for k in range(k_n):
        current = -1
        for b in range(t_n):
            if x_tracks[k, b, F_READ] > 0:
                value = int(round(x_tracks[k, b, F_READ_VALUE] * MAX_JERSEY))
                current = jersey_to_slot.get(value, current)
            pred[k, b] = current
    return pred


def eval_sim_game(game_dir: Path, weights: Path,
                  device: str | None = None) -> dict:
    game = load_slot_game(game_dir)
    assert game is not None, f"not slot-evaluable: {game_dir}"
    probs = _predict(weights, game, game["roster"], device)
    labels = game["labels"]
    sup = labels != LABEL_IGNORE
    top1 = probs.argmax(-1)
    conf = probs.max(-1)

    result: dict = {"game": Path(game_dir).name, "mode": "sim",
                    "tracks": len(game["tracks"]),
                    "bins": int(labels.shape[1]),
                    "supervised_frac": round(float(sup.mean()), 3),
                    "top1_acc": round(float((top1 == labels)[sup].mean()), 4),
                    "at_coverage": {}, "baseline": {}}
    for thr in THRESHOLDS:
        covered = sup & (conf >= thr)
        result["at_coverage"][str(thr)] = {
            "coverage": round(float(covered.sum() / sup.sum()), 4),
            "acc": (round(float((top1 == labels)[covered].mean()), 4)
                    if covered.any() else None),
        }
    base = sticky_baseline(game["x_tracks"], game["roster"])
    base_cov = sup & (base >= 0)
    result["baseline"] = {
        "policy": "sticky-last-jersey-read",
        "coverage": round(float(base_cov.sum() / sup.sum()), 4),
        "acc": (round(float((base == labels)[base_cov].mean()), 4)
                if base_cov.any() else None),
    }

    # swap recovery: bins from each mid-game swap until the affected
    # track's top1 next equals its new truth
    rows = read_stage(Path(game_dir) / "sim_identity").to_pylist()
    idx = {e: k for k, e in enumerate(game["tracks"])}
    latencies = []
    for r in rows:
        if r["tick"] == 0:
            continue
        k = idx.get(r["observed_id"])
        b0 = min(r["t_ms"] // BIN_MS, labels.shape[1] - 1)
        if k is None:
            continue
        hit = np.flatnonzero(
            top1[k, b0:b0 + RECOVERY_CAP_BINS]
            == labels[k, b0:b0 + RECOVERY_CAP_BINS])
        latencies.append(int(hit[0]) if hit.size else RECOVERY_CAP_BINS)
    result["swap_recovery_bins"] = {
        "n": len(latencies),
        "median": float(np.median(latencies)) if latencies else None,
        "capped_frac": (round(sum(x >= RECOVERY_CAP_BINS for x in latencies)
                              / len(latencies), 3) if latencies else None),
    }
    return result


def _shot_anchors(tokens: list[dict]) -> list[tuple[int, int]]:
    """(bin, shooter_jersey) per shooting pbp anchor with readable truth."""
    out = []
    for t in tokens:
        if t["channel"] != CH_PBP_ANCHOR:
            continue
        payload = json.loads(t["payload_json"] or "{}")
        jersey = parse_jersey(payload.get("shooter_jersey"))
        if payload.get("shooting_play") and jersey is not None:
            out.append((t["t_ms"] // BIN_MS, jersey))
    return out


def _shooter_track(x_tracks: np.ndarray, b: int) -> int | None:
    """nearest-ball-v1: the present track with the best (lowest) ball
    proximity in the window; ball_control distance breaks the tie."""
    lo = max(0, b - SHOT_WINDOW_BINS)
    hi = min(x_tracks.shape[1], b + SHOT_WINDOW_BINS + 1)
    best_k, best_d = None, np.inf
    for k in range(x_tracks.shape[0]):
        window = x_tracks[k, lo:hi]
        if not (window[:, F_PRESENT] > 0).any():
            continue
        cands = [d for col in (F_BALL_DIST, F_CONTROL_DIST)
                 for d in window[:, col][window[:, col] > 0]]
        d = min(cands) if cands else np.inf
        if d < best_d:
            best_k, best_d = k, float(d)
    return best_k


def _tracks_near_anchor(tokens: list[dict], b: int,
                        k_max: int = 16) -> list[int]:
    """Track ids with player tokens inside the anchor window, by count.

    Real games churn through hundreds of tracklets, so a global
    top-K-by-count roster of tracks almost never contains the shooter —
    selection must be window-local (the zero-coverage bug, 2026-08-01)."""
    lo_ms = (b - SHOT_WINDOW_BINS) * BIN_MS
    hi_ms = (b + SHOT_WINDOW_BINS + 1) * BIN_MS
    counts: dict[int, int] = {}
    for t in tokens:
        if (t["channel"] == CH_PLAYER and t.get("entity_id") is not None
                and lo_ms <= t["t_ms"] < hi_ms):
            counts[t["entity_id"]] = counts.get(t["entity_id"], 0) + 1
    return sorted(counts, key=lambda e: (-counts[e], e))[:k_max]


def eval_real_game(game_dir: Path, weights: Path,
                   device: str | None = None) -> dict:
    game_dir = Path(game_dir)
    assert stage_complete(game_dir / "tokens"), f"no tokens stage: {game_dir}"
    tokens = read_stage(game_dir / "tokens").to_pylist()
    anchors = _shot_anchors(tokens)
    roster = np.array(sorted({j for _b, j in anchors}), dtype=np.int64)
    result: dict = {"game": game_dir.name, "mode": "real",
                    "policy": "nearest-ball-v1-local",
                    "truth_shots": len(anchors),
                    "roster_size": int(roster.size),
                    "at_coverage": {}, "baseline": {}}
    if not anchors:
        return result
    x_global = featurize(tokens)

    calls = []  # (conf, correct, base_slot, truth_slot)
    slot_of = {int(j): s for s, j in enumerate(roster)}
    for b, jersey in anchors:
        local = _tracks_near_anchor(tokens, b)
        if not local:
            calls.append((0.0, False, -1, slot_of[jersey]))
            continue
        x_tracks = featurize_tracks(tokens, local)
        b_eff = min(b, x_tracks.shape[1] - 1)
        k = _shooter_track(x_tracks, b_eff)
        if k is None:
            calls.append((0.0, False, -1, slot_of[jersey]))
            continue
        game = {"x_tracks": x_tracks,
                "x_global": x_global[:x_tracks.shape[1]]}
        probs = _predict(weights, game, roster, device)
        base = sticky_baseline(x_tracks, roster)
        calls.append((float(probs[k, b_eff].max()),
                      int(probs[k, b_eff].argmax()) == slot_of[jersey],
                      int(base[k, b_eff]), slot_of[jersey]))
    anchored = [c for c in calls if c[0] > 0.0]
    result["diagnostics"] = {
        "anchors_with_track": len(anchored),
        "anchors_without_track": len(calls) - len(anchored),
        "conf_max": round(max((c[0] for c in calls), default=0.0), 4),
        "conf_median": (round(float(np.median([c[0] for c in anchored])), 4)
                        if anchored else None),
        "acc_among_anchored": (round(sum(c[1] for c in anchored)
                                     / len(anchored), 4) if anchored else None),
    }
    for thr in THRESHOLDS:
        covered = [c for c in calls if c[0] >= thr]
        result["at_coverage"][str(thr)] = {
            "coverage": round(len(covered) / len(calls), 4),
            "precision": (round(sum(c[1] for c in covered) / len(covered), 4)
                          if covered else None),
        }
    base_cov = [c for c in calls if c[2] >= 0]
    result["baseline"] = {
        "policy": "sticky-last-jersey-read",
        "coverage": round(len(base_cov) / len(calls), 4),
        "precision": (round(sum(c[2] == c[3] for c in base_cov)
                            / len(base_cov), 4) if base_cov else None),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--game-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for game_dir in args.game_dirs:
        if stage_complete(game_dir / "sim_identity"):
            result = eval_sim_game(game_dir, args.weights, args.device)
        else:
            result = eval_real_game(game_dir, args.weights, args.device)
        (args.out / f"{game_dir.name}.json").write_text(
            json.dumps(result, indent=2))
        print(f"SLOTS-EVAL: {json.dumps(result)}", flush=True)
    print("SLOTS_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
