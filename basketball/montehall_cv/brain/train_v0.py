"""Brain v0: recurrent state estimator over token streams (event tier).

A 2-layer GRU over 500ms-bin features with three heads — shot logit,
made logit, ball-mode logits — trained multitask: real games supervise
events (pbp anchors as targets), sim games supervise events AND dense
ball-mode dynamics. Whole-game sequences, batch of one, gradient
accumulation. Hours-per-iteration on spark by construction (no pixels).

Architecture is deliberately second-order at v0 (ratified 07-18): the
GRU is a latent-state model with linear whole-game scaling; the SSM
upgrade rides the same dataset and heads.

CLI:
    python -m montehall_cv.brain.train_v0 \
        --data-roots /work/sim /work/align --out /work/models/brain-v0 \
        --exclude-keys eid_401851175 eid_401851182 --epochs 12
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from montehall_cv.brain.dataset import (
    CHANNEL_SETS,
    MODE_IGNORE,
    MODES,
    N_FEATURES,
    discover_games,
    load_game,
)
from montehall_cv.eval.holdouts import is_sealed

HIDDEN = 128
LAYERS = 2
ACCUM = 4
SHOT_POS_WEIGHT = 20.0
MODE_LOSS_WEIGHT = 0.3
MADE_LOSS_WEIGHT = 0.5


def build_model():
    import torch.nn as nn

    class BrainV0(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(N_FEATURES, HIDDEN), nn.ReLU(),
            )
            self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=LAYERS,
                              batch_first=True)
            self.shot_head = nn.Linear(HIDDEN, 1)
            self.made_head = nn.Linear(HIDDEN, 1)
            self.mode_head = nn.Linear(HIDDEN, len(MODES))

        def forward(self, x):  # x: [B, T, F]
            h, _ = self.gru(self.proj(x))
            return (self.shot_head(h).squeeze(-1),
                    self.made_head(h).squeeze(-1),
                    self.mode_head(h))

    return BrainV0()


def _game_cache(cache_dir: Path, game_dir: Path,
                channels_name: str = "all") -> dict | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if channels_name == "all" else f".{channels_name}"
    cached = cache_dir / f"{game_dir.name}{tag}.npz"
    if cached.exists():
        z = np.load(cached)
        return {k: z[k] for k in ("x", "shot", "made", "mode")}
    game = load_game(game_dir, CHANNEL_SETS[channels_name])
    if game is None:
        return None
    np.savez_compressed(cached, x=game["x"], shot=game["shot"],
                        made=game["made"], mode=game["mode"])
    return {k: game[k] for k in ("x", "shot", "made", "mode")}


def _build_pool(dirs: list[Path], oversample: dict[str, int]) -> list[Path]:
    """Epoch sampling pool: a game whose path contains an oversample key
    appears K times (max K over matching keys); everything else once."""
    pool: list[Path] = []
    for d in dirs:
        ks = [k for key, k in oversample.items() if key in str(d)]
        pool.extend([d] * max(ks + [1]))
    return pool


def _dropout_channels(game_dir: Path, channels_name: str, p: float,
                      roots: list[Path], rng: random.Random) -> str:
    """Train-time align-channel dropout: with prob p a channel-complete
    game (under one of roots) loads its perception-only features, so the
    model cannot lean on clock/cut alone. Only applies from "all"."""
    if p <= 0 or channels_name != "all":
        return channels_name
    under = any(str(game_dir).startswith(str(r).rstrip("/") + "/")
                for r in roots)
    if not under:
        return channels_name
    return "perception" if rng.random() < p else channels_name


def _sealed_or_excluded(game_dir: Path, exclude_keys: list[str]) -> bool:
    name = game_dir.name
    if any(k in name for k in exclude_keys):
        return True
    if name.startswith("eid_") and is_sealed(name[len("eid_"):]):
        return True
    return False


def train(data_roots: list[Path], out_dir: Path, epochs: int,
          exclude_keys: list[str], lr: float = 1e-3,
          device: str | None = None, channels_name: str = "all",
          oversample: dict[str, int] | None = None,
          align_dropout: float = 0.0,
          dropout_roots: list[Path] | None = None) -> dict:
    import torch
    import torch.nn.functional as functional

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "feature_cache"

    dirs = [d for d in discover_games(data_roots)
            if not _sealed_or_excluded(d, exclude_keys)]
    if not dirs:
        raise RuntimeError("no training games discovered")

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pos_w = torch.tensor(SHOT_POS_WEIGHT, device=dev)
    rng = random.Random(0)
    pool = _build_pool(dirs, oversample or {})
    log_path = out_dir / "train_log.jsonl"
    print(f"BRAIN-V0: {len(dirs)} games (pool {len(pool)}), device={dev}, "
          f"channels={channels_name}, align_dropout={align_dropout}",
          flush=True)

    for epoch in range(1, epochs + 1):
        order = pool[:]
        rng.shuffle(order)
        model.train()
        totals = {"shot": 0.0, "made": 0.0, "mode": 0.0, "games": 0}
        opt.zero_grad()
        for gi, game_dir in enumerate(order):
            ch = _dropout_channels(game_dir, channels_name, align_dropout,
                                   dropout_roots or [], rng)
            arrays = _game_cache(cache, game_dir, ch)
            if arrays is None or len(arrays["x"]) < 8:
                continue
            x = torch.from_numpy(arrays["x"]).unsqueeze(0).to(dev)
            shot_t = torch.from_numpy(arrays["shot"]).to(dev)
            made_t = torch.from_numpy(arrays["made"]).to(dev)
            mode_t = torch.from_numpy(arrays["mode"]).to(dev)
            shot_l, made_l, mode_l = model(x)
            loss = functional.binary_cross_entropy_with_logits(
                shot_l[0], shot_t, pos_weight=pos_w)
            totals["shot"] += float(loss)
            shot_bins = shot_t > 0
            if shot_bins.any():
                made_loss = functional.binary_cross_entropy_with_logits(
                    made_l[0][shot_bins], made_t[shot_bins])
                loss = loss + MADE_LOSS_WEIGHT * made_loss
                totals["made"] += float(made_loss)
            if (mode_t != MODE_IGNORE).any():
                mode_loss = functional.cross_entropy(
                    mode_l[0], mode_t, ignore_index=MODE_IGNORE)
                loss = loss + MODE_LOSS_WEIGHT * mode_loss
                totals["mode"] += float(mode_loss)
            (loss / ACCUM).backward()
            totals["games"] += 1
            if (gi + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        n = max(totals["games"], 1)
        row = {"epoch": epoch, "games": totals["games"],
               "shot_loss": round(totals["shot"] / n, 4),
               "made_loss": round(totals["made"] / n, 4),
               "mode_loss": round(totals["mode"] / n, 4)}
        with log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"BRAIN-V0: epoch {epoch}/{epochs} {row}", flush=True)
        torch.save(model.state_dict(), out_dir / "last.pt")

    summary = {"games": len(dirs), "pool": len(pool), "epochs": epochs,
               "device": dev, "hidden": HIDDEN, "layers": LAYERS,
               "channels": channels_name, "oversample": oversample or {},
               "align_dropout": align_dropout}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print("BRAIN_V0_TRAIN_DONE", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-roots", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--exclude-keys", nargs="*", default=[])
    ap.add_argument("--device", default=None)
    ap.add_argument("--channels", choices=sorted(CHANNEL_SETS), default="all")
    ap.add_argument("--oversample", nargs="*", default=[],
                    metavar="PATHKEY=K",
                    help="games whose path contains PATHKEY appear K times "
                         "per epoch")
    ap.add_argument("--align-dropout", type=float, default=0.0,
                    help="train-time prob of masking clock/cut on games "
                         "under --dropout-roots")
    ap.add_argument("--dropout-roots", nargs="*", type=Path, default=[])
    args = ap.parse_args()
    oversample: dict[str, int] = {}
    for spec in args.oversample:
        key, _, k = spec.partition("=")
        if not key or not k.isdigit() or int(k) < 1:
            raise SystemExit(f"bad --oversample spec: {spec!r} "
                             "(want PATHKEY=K, K >= 1)")
        oversample[key] = int(k)
    train(args.data_roots, args.out, args.epochs, args.exclude_keys,
          lr=args.lr, device=args.device, channels_name=args.channels,
          oversample=oversample, align_dropout=args.align_dropout,
          dropout_roots=args.dropout_roots)


if __name__ == "__main__":
    main()
