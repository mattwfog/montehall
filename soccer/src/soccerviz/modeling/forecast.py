"""Small shared-player GRU: observed two-second history → three-second trajectory.

Run on Spark with the cached NVIDIA PyTorch container. This is a deterministic
player-motion experiment, not a counterfactual tactic simulator or ball model.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

HISTORY = 11
FUTURE = 15
HZ = 5


def windows(path: Path):
    with np.load(path, allow_pickle=False) as z:
        xy, ball, teams = z["xy"], z["ball"], z["teams"]
        periods, times = z["periods"], z["times"]
    inputs, labels, velocities, origins = [], [], [], []
    for i in range(HISTORY - 1, len(xy) - FUTURE, 10):
        start, end = i - HISTORY + 1, i + FUTURE + 1
        if periods[start] != periods[end - 1] or np.max(np.diff(times[start:end])) > 0.25:
            continue
        history = xy[start : i + 1]
        future = xy[i + 1 : end]
        for team in (0, 1):
            own = history[:, teams == team]
            opposing = history[:, teams != team]
            if (
                np.isfinite(own).all(axis=2).sum(axis=1).min() < 7
                or np.isfinite(opposing).all(axis=2).sum(axis=1).min() < 7
            ):
                continue
            own_center = np.nanmean(own, axis=1)
            opp_center = np.nanmean(opposing, axis=1)
            for player in np.flatnonzero(teams == team):
                h, f = history[:, player], future[:, player]
                if not np.isfinite(h).all() or not np.isfinite(f).all():
                    continue
                relative = h - h[-1]
                velocity = np.diff(h, axis=0, prepend=h[:1]) * HZ
                # Gaps in ball observations are explicit; no estimated ball positions.
                b = ball[start : i + 1] - h
                observed = np.isfinite(b).all(axis=1)
                b[~observed] = 0
                features = np.concatenate(
                    [
                        relative / 20,
                        velocity / 10,
                        b / 30,
                        observed[:, None],
                        (own_center - h) / 30,
                        (opp_center - h) / 30,
                    ],
                    axis=1,
                )
                inputs.append(features)
                labels.append(f - h[-1])
                velocities.append(h[-1] - h[-6])  # metres per second, past 1s
                origins.append((i, player))
    if not inputs:
        raise ValueError("No complete player windows")
    return tuple(np.asarray(x, dtype=np.float32) for x in (inputs, labels, velocities)), np.asarray(
        origins, dtype=np.int64
    )


def make_model(torch):
    class PlayerGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(11, 48, batch_first=True)
            self.head = torch.nn.Linear(48, FUTURE * 2)
            torch.nn.init.zeros_(self.head.weight)
            torch.nn.init.zeros_(self.head.bias)

        def forward(self, inputs, velocity):
            _, hidden = self.gru(inputs)
            residual = self.head(hidden[-1]).reshape(-1, FUTURE, 2)
            horizon = torch.arange(1, FUTURE + 1, device=inputs.device) / HZ
            return velocity[:, None] * horizon[None, :, None] + residual

    return PlayerGRU()


def displacement_metrics(prediction, target):
    error = np.linalg.norm(prediction - target, axis=-1)
    return {"ade_m": float(error.mean()), "fde_3s_m": float(error[:, -1].mean())}


def train_forecast(data: Path, out: Path, epochs: int = 8, device: str = "cuda") -> dict:
    import torch

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.manual_seed(20260907)
    np.random.seed(20260907)
    torch.set_num_threads(4)
    start = time.monotonic()
    train_path = data / "processed/game_1/tracking.npz"
    test_path = data / "processed/game_2/tracking.npz"
    train, _ = windows(train_path)
    test, origins = windows(test_path)
    dataset = torch.utils.data.TensorDataset(*(torch.from_numpy(a) for a in train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)
    model = make_model(torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    losses = []
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for inputs, target, velocity in loader:
            inputs, target, velocity = (a.to(device) for a in (inputs, target, velocity))
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(inputs, velocity), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(inputs)
        losses.append(total / len(dataset))
        print(f"Epoch {epoch + 1}/{epochs}: training MSE {losses[-1]:.4f}", flush=True)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for i in range(0, len(test[0]), 1024):
            inputs = torch.from_numpy(test[0][i : i + 1024]).to(device)
            velocity = torch.from_numpy(test[2][i : i + 1024]).to(device)
            predictions.append(model(inputs, velocity).cpu().numpy())
    predictions = np.concatenate(predictions)
    constant = test[2][:, None] * ((np.arange(1, FUTURE + 1) / HZ)[None, :, None])
    report = {
        "train_game": 1,
        "evaluation_game": 2,
        "epochs": epochs,
        "seed": 20260907,
        "device": device,
        "hardware": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "history_s": 2,
        "forecast_s": 3,
        "sample_hz": HZ,
        "training_player_windows": len(train[0]),
        "evaluation_player_windows": len(test[0]),
        "gru": displacement_metrics(predictions, test[1]),
        "constant_velocity": displacement_metrics(constant, test[1]),
        "stationary": displacement_metrics(np.zeros_like(test[1]), test[1]),
        "training_mse_by_epoch": losses,
        "elapsed_s": time.monotonic() - start,
        "input_sha256": {
            "train": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "eval": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        },
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "Deterministic player-only prediction; no ball forecast or tactic simulation",
            "One held-out development match; no hyperparameter selection on this match",
            "Overlapping windows and players are correlated; counts are not independent trials",
            "Complete-window selection excludes occluded or substituted player tracks",
            "Uses provider tracking; video detector and tracker are not implemented",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "report": report}, out / "forecast.pt")
    (out / "forecast-report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        out / "forecast-evaluation.npz",
        origins=origins,
        predicted_displacement=predictions,
        target_displacement=test[1],
    )
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(train_forecast(args.data, args.artifacts, args.epochs, args.device), indent=2))
