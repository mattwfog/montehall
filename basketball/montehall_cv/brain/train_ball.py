"""Ball-state estimator: position + mode + holder from tokens (§1).

One GRU over the global+ball observation stream, one GRU shared across
observed tracks; per bin the global state is fused with pooled track
context. Heads: ball mode (4-way CE on sim truth), ball position
(masked MSE in normalized court space), holder (attention over track
embeddings + a learned "none" logit, CE on the permutation-mapped
holder). The held-mode design intent: the holder head lets the
estimator ride player tracks through the ~90% of held bins where the
ball is undetected.

v1.1 knobs (2026-08-03, after the v1 measurement):
  --bidirectional  the product is post-game analysis — offline
                   estimation is legitimate, and v1's causal GRU lost
                   to offline linear interp (8.3-9.0ft invisible bins);
                   bidirectional GRUs are the smallest change that can
                   use future observations the way interp does.
  --ball-dropout P train-time random drop of ball observations (and
                   the carry-forward recomputed after the drop), pulling
                   sim's 32-35% observed-bin density toward the real
                   dirs' 2-6% so the estimator trains at deployment
                   sparsity. Same lesson as slots' --align-dropout.

CLI:
    python -m montehall_cv.brain.train_ball \
        --data-roots /work/sim --out /work/models/ball-v1.1 \
        --bidirectional --ball-dropout 0.85 \
        --exclude-keys sim_0007_045 sim_0007_046 sim_0007_047 \
            sim_0007_048 sim_0007_049 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from montehall_cv.brain.dataset import MODE_IGNORE, N_FEATURES, discover_games
from montehall_cv.brain.dataset_ball import (
    HOLDER_IGNORE,
    N_BALL_FEATURES,
    load_ball_game,
)
from montehall_cv.brain.dataset_slots import N_TRACK_FEATURES

HIDDEN = 128
EMB = 32
LAYERS = 2
POS_LOSS_W = 25.0  # normalized-space MSE is tiny; keep gradients comparable


def build_model(bidirectional: bool = False):
    import torch
    import torch.nn as nn

    d = 2 if bidirectional else 1

    class BallBrain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.track_proj = nn.Sequential(
                nn.Linear(N_TRACK_FEATURES, HIDDEN), nn.ReLU())
            self.track_gru = nn.GRU(HIDDEN, HIDDEN, num_layers=LAYERS,
                                    batch_first=True,
                                    bidirectional=bidirectional)
            self.global_proj = nn.Sequential(
                nn.Linear(N_FEATURES + N_BALL_FEATURES, HIDDEN), nn.ReLU())
            self.global_gru = nn.GRU(HIDDEN, HIDDEN, batch_first=True,
                                     bidirectional=bidirectional)
            self.fuse = nn.Sequential(
                nn.Linear(2 * HIDDEN * d, HIDDEN), nn.ReLU())
            self.mode_head = nn.Linear(HIDDEN, 4)
            self.pos_head = nn.Linear(HIDDEN, 2)
            self.track_emb = nn.Linear(HIDDEN * d, EMB)
            self.holder_query = nn.Linear(HIDDEN, EMB)
            self.none_logit = nn.Linear(HIDDEN, 1)

        def forward(self, x_tracks, x_global, x_ball):
            # x_tracks [K,T,Ft] · x_global [T,Fg] · x_ball [T,Fb]
            h_t, _ = self.track_gru(self.track_proj(x_tracks))  # [K,T,H]
            g_in = torch.cat([x_global, x_ball], dim=-1).unsqueeze(0)
            h_g, _ = self.global_gru(self.global_proj(g_in))  # [1,T,H]
            ctx = h_t.mean(dim=0, keepdim=True) if h_t.shape[0] else (
                torch.zeros_like(h_g))
            state = self.fuse(torch.cat([h_g, ctx], dim=-1))[0]  # [T,H]
            emb = self.track_emb(h_t)                # [K,T,E]
            q = self.holder_query(state)             # [T,E]
            scores = torch.einsum("kte,te->tk", emb, q) / (EMB ** 0.5)
            holder = torch.cat([scores, self.none_logit(state)], dim=-1)
            return self.mode_head(state), self.pos_head(state), holder

    return BallBrain()


N_DROPOUT_VARIANTS = 3


def train(data_roots: list[Path], out_dir: Path, epochs: int,
          exclude_keys: list[str], lr: float = 1e-3,
          device: str | None = None, bidirectional: bool = False,
          ball_dropout: float = 0.0) -> dict:
    import torch
    import torch.nn.functional as functional

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = [d for d in discover_games(data_roots)
            if not any(k in d.name for k in exclude_keys)]
    # with dropout, precompute a few degraded variants per game and cycle
    # them by epoch (featurizing per epoch would dominate wall-clock)
    n_var = N_DROPOUT_VARIANTS if ball_dropout > 0 else 1
    games: list[list[dict]] = []
    for d in dirs:
        variants = [g for g in
                    (load_ball_game(d, ball_dropout, seed=v)
                     for v in range(n_var)) if g is not None]
        if variants:
            games.append(variants)
    if not games:
        raise RuntimeError("no ball-trainable games (need sim_states + "
                           "sim_identity — regenerate the sim corpus)")

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(bidirectional).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        order = np.random.permutation(len(games))
        total, seen = 0.0, 0
        for gi in order:
            g = games[gi][epoch % len(games[gi])]
            mode_logits, pos, holder_logits = model(
                torch.from_numpy(g["x_tracks"]).to(dev),
                torch.from_numpy(g["x_global"]).to(dev),
                torch.from_numpy(g["x_ball"]).to(dev))
            loss = functional.cross_entropy(
                mode_logits, torch.from_numpy(g["mode"]).to(dev),
                ignore_index=MODE_IGNORE)
            loss = loss + functional.cross_entropy(
                holder_logits, torch.from_numpy(g["holder"]).to(dev),
                ignore_index=HOLDER_IGNORE)
            mask = torch.from_numpy(g["pos_mask"]).to(dev)
            if bool(mask.any()):
                target = torch.from_numpy(
                    np.nan_to_num(g["ball_xy"])).to(dev)
                loss = loss + POS_LOSS_W * functional.mse_loss(
                    pos[mask], target[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
            seen += 1
        history.append(round(total / max(seen, 1), 4))
        print(f"BALL: epoch {epoch + 1}/{epochs} loss {history[-1]} "
              f"({seen} games) {time.strftime('%H:%M:%S')}", flush=True)
    torch.save(model.state_dict(), out_dir / "last.pt")
    meta = {"games": len(games), "epochs": epochs, "loss_history": history,
            "hidden": HIDDEN, "emb": EMB, "pos_loss_w": POS_LOSS_W,
            "bidirectional": bidirectional, "ball_dropout": ball_dropout}
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-roots", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--exclude-keys", nargs="*", default=[])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--ball-dropout", type=float, default=0.0)
    args = ap.parse_args()
    meta = train(args.data_roots, args.out, args.epochs, args.exclude_keys,
                 lr=args.lr, device=args.device,
                 bidirectional=args.bidirectional,
                 ball_dropout=args.ball_dropout)
    print(f"BALL_TRAIN_DONE {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main()
