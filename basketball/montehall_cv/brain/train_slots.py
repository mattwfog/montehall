"""Slot estimator v1: per-track identity belief over the roster (§10).

One GRU shared across observed tracks (tracks are the batch dim), a
global-context GRU over the v0 feature stream, fused per bin; the head
scores each track embedding against roster jersey embeddings -> per
track per bin a distribution over roster slots. Trained with CE on the
sim_identity permutation truth. Abstention is an inference policy
(max-prob threshold -> the precision-at-coverage curve in eval_slots),
not a trained class.

CLI:
    python -m montehall_cv.brain.train_slots \
        --data-roots /work/sim --out /work/models/slots-v1 \
        --exclude-keys sim_0007_045 sim_0007_046 sim_0007_047 \
            sim_0007_048 sim_0007_049 --epochs 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from montehall_cv.brain.dataset import N_FEATURES, discover_games
from montehall_cv.brain.dataset_slots import (
    LABEL_IGNORE,
    N_TRACK_FEATURES,
    load_slot_game,
)

HIDDEN = 128
EMB = 32
LAYERS = 2


def build_model():
    import torch
    import torch.nn as nn

    class SlotBrain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.track_proj = nn.Sequential(
                nn.Linear(N_TRACK_FEATURES, HIDDEN), nn.ReLU())
            self.track_gru = nn.GRU(HIDDEN, HIDDEN, num_layers=LAYERS,
                                    batch_first=True)
            self.global_proj = nn.Sequential(
                nn.Linear(N_FEATURES, HIDDEN), nn.ReLU())
            self.global_gru = nn.GRU(HIDDEN, HIDDEN, batch_first=True)
            self.fuse = nn.Linear(2 * HIDDEN, EMB)
            self.jersey_emb = nn.Embedding(100, EMB)

        def forward(self, x_tracks, x_global, roster):
            # x_tracks [K,T,Ft] · x_global [T,Fg] · roster [R] jersey ints
            h_t, _ = self.track_gru(self.track_proj(x_tracks))
            h_g, _ = self.global_gru(self.global_proj(x_global.unsqueeze(0)))
            h_g = h_g.expand(h_t.shape[0], -1, -1)
            emb = self.fuse(torch.cat([h_t, h_g], dim=-1))
            logits = emb @ self.jersey_emb(roster).T / (EMB ** 0.5)
            return logits  # [K, T, R]

    return SlotBrain()


def train(data_roots: list[Path], out_dir: Path, epochs: int,
          exclude_keys: list[str], lr: float = 1e-3,
          device: str | None = None) -> dict:
    import torch
    import torch.nn.functional as functional

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = [d for d in discover_games(data_roots)
            if not any(k in d.name for k in exclude_keys)]
    games = [g for g in (load_slot_game(d) for d in dirs) if g is not None]
    if not games:
        raise RuntimeError("no slot-trainable games (need sim_identity + "
                           "jerseys meta — regenerate the sim corpus)")

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        order = np.random.permutation(len(games))
        total, seen = 0.0, 0
        for gi in order:
            g = games[gi]
            logits = model(
                torch.from_numpy(g["x_tracks"]).to(dev),
                torch.from_numpy(g["x_global"]).to(dev),
                torch.from_numpy(g["roster"]).to(dev))
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                torch.from_numpy(g["labels"]).to(dev).reshape(-1),
                ignore_index=LABEL_IGNORE)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
            seen += 1
        history.append(round(total / max(seen, 1), 4))
        print(f"SLOTS: epoch {epoch + 1}/{epochs} loss {history[-1]} "
              f"({seen} games) {time.strftime('%H:%M:%S')}", flush=True)
    torch.save(model.state_dict(), out_dir / "last.pt")
    meta = {"games": len(games), "epochs": epochs, "loss_history": history,
            "hidden": HIDDEN, "emb": EMB}
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-roots", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--exclude-keys", nargs="*", default=[])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    meta = train(args.data_roots, args.out, args.epochs, args.exclude_keys,
                 lr=args.lr, device=args.device)
    print(f"SLOTS_TRAIN_DONE {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main()
