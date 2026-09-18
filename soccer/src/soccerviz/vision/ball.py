"""One-second ball-motion baseline and conservative event-actor attribution."""

import json
from pathlib import Path

import numpy as np


def ball_windows(path):
    with np.load(path, allow_pickle=False) as z:
        ball, xy, teams, periods, times = [
            z[k] for k in ["ball", "xy", "teams", "periods", "times"]
        ]
    inputs, targets, velocities, origins = [], [], [], []
    for i in range(5, len(ball) - 5, 5):
        h, future = ball[i - 5 : i + 1], ball[i + 1 : i + 6]
        if periods[i - 5] != periods[i + 5] or np.max(np.diff(times[i - 5 : i + 6])) > 0.25:
            continue
        if not np.isfinite(h).all() or not np.isfinite(future).all():
            continue
        context = []
        valid = True
        for t in range(i - 5, i + 1):
            nearest = []
            for team in (0, 1):
                players = xy[t, teams == team]
                players = players[np.isfinite(players).all(axis=1)]
                if not len(players):
                    valid = False
                    break
                nearest.extend(
                    players[np.argmin(np.linalg.norm(players - ball[t], axis=1))] - ball[t]
                )
            context.append(nearest)
        if not valid:
            continue
        velocity = np.diff(h, axis=0, prepend=h[:1]) * 5
        inputs.append(
            np.concatenate([(h - h[-1]) / 20, velocity / 30, np.array(context) / 20], axis=1)
        )
        targets.append(future - h[-1])
        velocities.append((h[-1] - h[-3]) / 0.4)
        origins.append(i)
    return tuple(
        np.asarray(a, dtype=np.float32) for a in (inputs, targets, velocities)
    ), np.asarray(origins)


def train_ball(data: Path, out: Path, epochs=8, device="cuda"):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    import torch

    torch.manual_seed(73)
    torch.set_num_threads(4)
    train, _ = ball_windows(data / "processed/game_1/tracking.npz")
    test, origins = ball_windows(data / "processed/game_2/tracking.npz")

    class BallGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(8, 32, batch_first=True)
            self.head = torch.nn.Linear(32, 10)

        def forward(self, x, velocity):
            _, hidden = self.gru(x)
            horizon = torch.arange(1, 6, device=x.device) / 5
            return velocity[:, None] * horizon[None, :, None] + self.head(hidden[-1]).reshape(
                -1, 5, 2
            )

    model = BallGRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(torch.from_numpy(a) for a in train)),
        batch_size=256,
        shuffle=True,
    )
    for epoch in range(epochs):
        for x, y, velocity in loader:
            x, y, velocity = [a.to(device) for a in (x, y, velocity)]
            loss = torch.nn.functional.mse_loss(model(x, velocity), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        print(f"Ball epoch {epoch + 1}/{epochs}", flush=True)
    model.eval()
    with torch.inference_mode():
        predicted = np.concatenate(
            [
                model(
                    torch.from_numpy(test[0][i : i + 1024]).to(device),
                    torch.from_numpy(test[2][i : i + 1024]).to(device),
                )
                .cpu()
                .numpy()
                for i in range(0, len(test[0]), 1024)
            ]
        )
    constant = test[2][:, None] * (np.arange(1, 6) / 5)[None, :, None]

    def error(prediction):
        distance = np.linalg.norm(prediction - test[1], axis=-1)
        return {"ade_m": float(distance.mean()), "fde_1s_m": float(distance[:, -1].mean())}

    report = {
        "train_game": 1,
        "evaluation_game": 2,
        "history_s": 1,
        "horizon_s": 1,
        "epochs": epochs,
        "training_windows": len(train[0]),
        "evaluation_windows": len(test[0]),
        "gru": error(predicted),
        "constant_velocity": error(constant),
        "stationary": error(np.zeros_like(predicted)),
        "hardware": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "seed": 73,
        "limitations": [
            "Complete observed ball windows only; excludes missing or unobserved ball segments",
            "Two-dimensional deterministic motion; aerial height is unavailable",
            "No counterfactual pass or next-touch guarantee",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "report": report}, out / "model.pt")
    np.savez_compressed(
        out / "evaluation.npz",
        origins=origins,
        predicted_displacement=predicted,
        target_displacement=test[1],
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def actor_hypothesis(positions, ball, maximum_distance=2.5, minimum_margin=0.5):
    if not np.isfinite(ball).all():
        return None, None
    distance = np.linalg.norm(positions - ball, axis=1)
    distance[~np.isfinite(distance)] = np.inf
    order = np.argsort(distance)
    if (
        len(order) < 2
        or distance[order[0]] > maximum_distance
        or distance[order[1]] - distance[order[0]] < minimum_margin
    ):
        return None, float(distance[order[0]]) if np.isfinite(distance[order[0]]) else None
    return int(order[0]), float(distance[order[0]])


def evaluate_actors(data: Path, out: Path):
    import pandas as pd

    from soccerviz.core.data import load_match, write_json

    match = load_match(data, 2)
    predictions = [
        actor_hypothesis(xy, ball) for xy, ball in zip(match.xy, match.ball, strict=True)
    ]
    timeline = pd.DataFrame(
        {
            "time_s": match.times,
            "actor_id": [match.players[p] if p is not None else None for p, _ in predictions],
            "actor_distance_m": [d for _, d in predictions],
            "status": [
                "near_ball_hypothesis" if p is not None else "unknown_or_in_flight"
                for p, _ in predictions
            ],
        }
    )
    rows = []
    for event_id, event in match.events[
        match.events.Type.isin(["PASS", "RECOVERY", "SHOT"])
    ].iterrows():
        truth = str(event.Team) + ":" + str(event.From)
        if truth not in match.players:
            continue
        i = int(np.searchsorted(match.times, event["Start Time [s]"], side="right")) - 1
        if i < 0 or match.periods[i] != event.Period:
            continue
        actor = timeline.iloc[i].actor_id
        rows.append(
            {
                "event_id": f"g2-e{event_id}",
                "type": event.Type,
                "true_actor": truth,
                "predicted_actor": actor,
                "abstained": actor is None,
                "correct": actor == truth,
            }
        )
    table = pd.DataFrame(rows)
    selected = table[~table.abstained]
    report = {
        "evaluation_game": 2,
        "events": len(table),
        "attributed_events": len(selected),
        "event_coverage": len(selected) / len(table),
        "accuracy_when_attributed": float(selected.correct.mean()),
        "all_events_correct_fraction": float(table.correct.mean()),
        "thresholds": {"maximum_distance_m": 2.5, "minimum_nearest_margin_m": 0.5},
        "limitations": [
            "Uses provider positions and IDs, not video tracking",
            "Nearest-at-start attribution is not continuous possession truth",
            "Abstains during flight, missing ball, or ambiguous nearby players",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    timeline.to_parquet(out / "timeline.parquet", index=False)
    table.to_parquet(out / "evaluation.parquet", index=False)
    write_json(out / "report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(train_ball(Path("data"), Path("artifacts/ball")), indent=2))
