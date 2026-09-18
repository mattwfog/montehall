"""Heteroscedastic player-motion baseline with separate-half uncertainty calibration."""

import hashlib
import json
from pathlib import Path

import numpy as np

from soccerviz.modeling.forecast import FUTURE, HZ, displacement_metrics, windows


def gaussian_model(torch):
    class GaussianGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(11, 48, batch_first=True)
            self.head = torch.nn.Linear(48, FUTURE * 3)

        def forward(self, features, velocity):
            _, hidden = self.gru(features)
            output = self.head(hidden[-1]).reshape(-1, FUTURE, 3)
            horizon = torch.arange(1, FUTURE + 1, device=features.device) / HZ
            mean = velocity[:, None] * horizon[None, :, None] + output[..., :2]
            scale = torch.nn.functional.softplus(output[..., 2]) + 0.1
            return mean, scale

    return GaussianGRU()


def nll(target, mean, scale):
    return float(
        np.mean(
            np.sum((target - mean) ** 2, axis=-1) / (2 * scale**2)
            + 2 * np.log(scale)
            + np.log(2 * np.pi)
        )
    )


def train_probabilistic(data: Path, out: Path, epochs=8, device="cuda"):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    import torch

    torch.manual_seed(41)
    torch.set_num_threads(4)
    path1, path2 = [data / "processed" / f"game_{g}" / "tracking.npz" for g in (1, 2)]
    game1, origins1 = windows(path1)
    test, origins = windows(path2)
    with np.load(path1, allow_pickle=False) as z:
        periods = z["periods"][origins1[:, 0]]
    fit_mask, calibration_mask = periods == 1, periods == 2
    train = tuple(a[fit_mask] for a in game1)
    calibration = tuple(a[calibration_mask] for a in game1)
    model = gaussian_model(torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    dataset = torch.utils.data.TensorDataset(*(torch.from_numpy(a) for a in train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    losses = []
    for epoch in range(epochs):
        model.train()
        total = 0
        for x, y, velocity in loader:
            x, y, velocity = [a.to(device) for a in (x, y, velocity)]
            mean, sigma = model(x, velocity)
            loss = (((y - mean) ** 2).sum(dim=-1) / (2 * sigma**2) + 2 * sigma.log()).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(x)
        losses.append(total / len(dataset))
        print(
            f"Gaussian epoch {epoch + 1}/{epochs}: NLL without constant {losses[-1]:.4f}",
            flush=True,
        )

    def infer(table):
        model.eval()
        means, scales = [], []
        with torch.inference_mode():
            for i in range(0, len(table[0]), 1024):
                mean, sigma = model(
                    torch.from_numpy(table[0][i : i + 1024]).to(device),
                    torch.from_numpy(table[2][i : i + 1024]).to(device),
                )
                means.append(mean.cpu().numpy())
                scales.append(sigma.cpu().numpy())
        return np.concatenate(means), np.concatenate(scales)

    mean_cal, scale_cal = infer(calibration)
    normalized = np.linalg.norm(calibration[1] - mean_cal, axis=-1) / scale_cal
    factor = float(np.quantile(normalized, 0.9, method="higher"))
    means, scales = infer(test)
    radius = scales * factor
    error = np.linalg.norm(test[1] - means, axis=-1)
    horizon = (np.arange(1, FUTURE + 1) / HZ)[None, :, None]
    constant_train = train[2][:, None] * horizon
    baseline_scale = np.sqrt(np.mean((train[1] - constant_train) ** 2, axis=(0, 2)))
    constant_test = test[2][:, None] * horizon
    report = {
        "fit": "game 1 first half",
        "uncertainty_calibration": "game 1 second half",
        "evaluation": "game 2, both halves",
        "epochs": epochs,
        "seed": 41,
        "training_windows": len(train[0]),
        "calibration_windows": len(calibration[0]),
        "evaluation_windows": len(test[0]),
        "gaussian_gru": displacement_metrics(means, test[1]),
        "mean_nll": nll(test[1], means, scales),
        "constant_velocity_gaussian_nll": nll(test[1], constant_test, baseline_scale),
        "uncalibrated_90_point_coverage": float(
            (error <= scales * np.sqrt(-2 * np.log(0.1))).mean()
        ),
        "calibrated_90_point_coverage": float((error <= radius).mean()),
        "calibrated_90_endpoint_coverage": float((error[:, -1] <= radius[:, -1]).mean()),
        "mean_endpoint_radius_m": float(radius[:, -1].mean()),
        "calibration_factor": factor,
        "hardware": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "Single isotropic Gaussian per player and horizon; not multimodal joint tactics",
            "Empirical pointwise calibration; not simultaneous whole-trajectory or team coverage",
            "Correlated windows and distribution shift preclude a coverage guarantee",
            "Uses less training data than the earlier all-game-1 deterministic baseline",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "evaluation.npz",
        origins=origins,
        predicted_displacement=means,
        sigma_m=scales,
        empirical_90_radius_m=radius,
        target_displacement=test[1],
    )
    torch.save({"state_dict": model.cpu().state_dict(), "report": report}, out / "model.pt")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(train_probabilistic(Path("data"), Path("artifacts/probabilistic")), indent=2))
