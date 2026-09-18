"""Train the permissive whole-crop jersey-number classifier at broadcast scale.

Fixed protocol, no search: timm ResNet-34 (ImageNet init), 101 classes (0..99 and
illegible), 128 px input, AdamW, cosine schedule, label smoothing 0.05. Every
training crop is degraded to a broadcast torso height (16 to 64 px) before it is
resized to the input, then jittered.

Sources per epoch, class-balanced so the number prior is flat: synthetic positives
rendered on the fly (a number drawn uniformly from 1..99 in a kit font on a real
broadcast torso from the `football-players-detection` pool), the same pool
un-rendered as the illegible class, and the real `taiseis` and Pusan number crops
capped per class so no class dominates. The first run (`jersey-resnet34-v1`)
learned "illegible" from number-free windows below volleyball digits and the
taiseis class frequencies, and on real broadcast crops called 91% of torsos
legible and collapsed onto classes 17 and 5; the matched real/rendered pairs make
the digits the only cue that separates the classes.

Validation is the real exports' valid splits plus a fixed synthetic/illegible
valid set from the pool's valid split, degraded once with a fixed seed. best.pt is
the highest valid number accuracy over legible tiles among epochs whose
illegible-class recall clears the floor. After training, one temperature is fitted
on the valid set by minimising negative log-likelihood and stored in the
checkpoint; inference divides logits by it. The SoccerNet crops are never opened.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np

from soccerviz.candidates.jersey_classifier import (
    BACKBONE,
    ILLEGIBLE,
    INPUT_SIZE,
    NUM_CLASSES,
    PRETRAINED_TAG,
    SYNTHETIC_NUMBERS,
    TORSO_HEIGHT_RANGE,
    build_network,
    degrade,
    folder_records,
    jitter_crop,
    load_crop,
    load_fonts,
    preprocess,
    pusan_records,
    render_number,
    softmax,
)
from soccerviz.core.assets import sha256

SEED = 20260908
TRAINING = {
    "epochs": 12,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "warmup_epochs": 1,
    "label_smoothing": 0.05,
    "synthetic_per_epoch": 24000,
    "illegible_per_epoch": 8000,
    "real_per_class_cap": 120,
    "real_negative_cap": 1000,
    "valid_synthetic": 2000,
    "valid_illegible": 881,
    "torso_height_range": list(TORSO_HEIGHT_RANGE),
    "illegible_recall_floor": 0.8,
    "temperature_grid": [0.5, 3.0, 26],
}


def pool_records(pool_dir, split):
    """The cut torso pool's tiles of one split as illegible records."""
    paths = sorted((Path(pool_dir) / split).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No torso tiles under {pool_dir}/{split}")
    return [{"image_path": str(p), "label": ILLEGIBLE, "source": "broadcast-torso"} for p in paths]


def synthetic_records(pool, count, rng, seed_base):
    """`count` records that render a uniform 1..99 number on a random pool torso at load time."""
    picks = rng.choice(len(pool), count, replace=True)
    numbers = rng.choice(SYNTHETIC_NUMBERS, count, replace=True)
    return [
        {**pool[i], "label": int(n), "source": "synthetic", "render_seed": int(seed_base + k)}
        for k, (i, n) in enumerate(zip(picks, numbers))
    ]


def capped_by_class(records, cap, rng):
    """At most `cap` records per label, chosen at random."""
    by_label = {}
    for record in records:
        by_label.setdefault(record["label"], []).append(record)
    kept = []
    for group in by_label.values():
        if len(group) > cap:
            group = [group[i] for i in rng.choice(len(group), cap, replace=False)]
        kept.extend(group)
    return kept


def epoch_records(taiseis, pusan, pool, config, rng, epoch):
    """One epoch's sample: synthetic positives, real illegible torsos, capped real numbers."""
    synthetic = synthetic_records(pool, config["synthetic_per_epoch"], rng, epoch * 10**6)
    illegible = [
        pool[i] for i in rng.choice(len(pool), config["illegible_per_epoch"], replace=True)
    ]
    real_numbers = capped_by_class(
        [r for r in taiseis + pusan if r["label"] != ILLEGIBLE], config["real_per_class_cap"], rng
    )
    real_negatives = [r for r in taiseis + pusan if r["label"] == ILLEGIBLE]
    if len(real_negatives) > config["real_negative_cap"]:
        real_negatives = [
            real_negatives[i]
            for i in rng.choice(len(real_negatives), config["real_negative_cap"], replace=False)
        ]
    records = synthetic + illegible + real_numbers + real_negatives
    order = rng.permutation(len(records))
    return [records[i] for i in order]


def make_dataset(records, train, seed, fonts):
    import torch

    class CropDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            record = records[index]
            crop = load_crop(record)
            # Training: fresh randomness per item; validation: fixed per item, so the
            # valid degradation is identical every epoch. Synthetic records render with
            # their own seed so a valid synthetic tile is the same every epoch too.
            rng = np.random.default_rng(
                random.getrandbits(32) if train else (seed * 1000003 + index) % (2**32)
            )
            if record.get("source") == "synthetic":
                crop = render_number(
                    crop, record["label"], fonts, np.random.default_rng(record["render_seed"])
                )
            crop = degrade(crop, rng)
            if train:
                crop = jitter_crop(crop, rng)
            return torch.from_numpy(preprocess(crop)), int(record["label"])

    return CropDataset()


def evaluate(net, loader, device):
    """Number accuracy over legible tiles, illegible recall and precision, and calibration bins."""
    import torch

    net.eval()
    correct_number, legible_total = 0, 0
    illegible_hit, illegible_total, predicted_illegible = 0, 0, 0
    confidences, hits = [], []
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = torch.softmax(net(images.to(device)).float(), dim=-1).cpu().numpy()
            labels = labels.numpy()
            predicted = probabilities.argmax(axis=1)
            legible = labels != ILLEGIBLE
            number_prediction = probabilities[:, :ILLEGIBLE].argmax(axis=1)
            correct_number += int((number_prediction[legible] == labels[legible]).sum())
            legible_total += int(legible.sum())
            illegible_total += int((~legible).sum())
            illegible_hit += int((predicted[~legible] == ILLEGIBLE).sum())
            predicted_illegible += int((predicted == ILLEGIBLE).sum())
            confidences.extend(probabilities[legible].max(axis=1))
            hits.extend(predicted[legible] == labels[legible])
    confidences, hits = np.asarray(confidences), np.asarray(hits, dtype=float)
    bins = np.clip((confidences * 10).astype(int), 0, 9)
    calibration = {
        f"{b / 10:.1f}": [float(hits[bins == b].mean()), int((bins == b).sum())]
        for b in range(10)
        if (bins == b).any()
    }
    return {
        "legible_tiles": legible_total,
        "number_accuracy": correct_number / max(1, legible_total),
        "illegible_tiles": illegible_total,
        "illegible_recall": illegible_hit / max(1, illegible_total),
        "illegible_precision": illegible_hit / max(1, predicted_illegible),
        "expected_calibration_error": float(
            sum(
                abs(acc - (b + 0.05)) * n
                for (b, (acc, n)) in ((float(k), v) for k, v in calibration.items())
            )
            / max(1, len(hits))
        ),
        "calibration_bins": calibration,
    }


def fit_temperature(net, loader, device, grid):
    """Temperature minimising valid negative log-likelihood over a grid; logits are divided by it."""
    import torch

    net.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits.append(net(images.to(device)).float().cpu().numpy())
            labels.append(targets.numpy())
    logits, labels = np.concatenate(logits), np.concatenate(labels)
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(grid[0], grid[1], int(grid[2])):
        p = softmax(logits / t)
        nll = float(-np.log(np.maximum(p[np.arange(len(labels)), labels], 1e-9)).mean())
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t, best_nll


def improves(candidate, best, floor):
    """best.pt: highest number accuracy among epochs whose illegible recall clears the floor."""
    candidate_ok, best_ok = (
        candidate["illegible_recall"] >= floor,
        best["illegible_recall"] >= floor,
    )
    if candidate_ok and not best_ok:
        return True
    if best_ok and not candidate_ok:
        return False
    return candidate["number_accuracy"] > best["number_accuracy"]


def run(args):
    import torch

    torch.manual_seed(SEED)
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    config = {**TRAINING, "epochs": args.epochs, "batch_size": args.batch_size}
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    taiseis_train, taiseis_valid = (
        folder_records(args.taiseis, "train"),
        folder_records(args.taiseis, "valid"),
    )
    pusan_train, pusan_valid = (
        pusan_records(args.pusan, "train"),
        pusan_records(args.pusan, "valid"),
    )
    pool_train = pool_records(args.torso_pool, "train")
    pool_valid = pool_records(args.torso_pool, "valid")
    fonts = load_fonts(args.fonts)
    valid_rng = np.random.default_rng(SEED + 1)
    valid_records = (
        taiseis_valid
        + capped_by_class(pusan_valid, config["real_per_class_cap"], valid_rng)
        + synthetic_records(pool_valid, config["valid_synthetic"], valid_rng, 9 * 10**8)
        + pool_valid[: config["valid_illegible"]]
    )
    if args.smoke_step:
        valid_records = [
            valid_records[i] for i in valid_rng.choice(len(valid_records), 256, replace=False)
        ]
    device = torch.device(args.device)
    net = build_network(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    per_epoch = (
        len(epoch_records(taiseis_train, pusan_train, pool_train, config, rng, 0))
        // config["batch_size"]
    )
    total_steps = max(1, config["epochs"] * per_epoch)
    warmup = config["warmup_epochs"] * per_epoch

    def lr_at(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    valid_loader = torch.utils.data.DataLoader(
        make_dataset(valid_records, False, SEED, fonts),
        batch_size=config["batch_size"],
        num_workers=args.workers,
    )
    evidence = {
        "schema": "jersey-classifier-training-evidence/v1",
        "completed": False,
        "sources": {
            "taiseis": {
                "export": str(args.taiseis),
                "train_tiles": len(taiseis_train),
                "valid_tiles": len(taiseis_valid),
                "source_sha256": sha256(Path(args.taiseis) / "source.json"),
            },
            "pusan": {
                "export": str(args.pusan),
                "train_records": len(pusan_train),
                "valid_records": len(pusan_valid),
                "annotations_sha256": {
                    s: sha256(Path(args.pusan) / s / "_annotations.coco.json")
                    for s in ("train", "valid")
                },
            },
            "torso_pool": {
                "path": str(args.torso_pool),
                "train_torsos": len(pool_train),
                "valid_torsos": len(pool_valid),
                "manifest_sha256": sha256(Path(args.torso_pool) / "manifest.json"),
            },
            "fonts": {
                "path": str(args.fonts),
                "count": len(fonts),
                "source_sha256": sha256(Path(args.fonts) / "source.json"),
            },
        },
        "valid_records": len(valid_records),
        "classes": NUM_CLASSES,
        "backbone": BACKBONE,
        "initialization": "random" if args.no_pretrained else PRETRAINED_TAG,
        "input_size": INPUT_SIZE,
        "training": config,
        "selection": "highest valid number_accuracy among epochs with illegible_recall >= floor",
        "seed": SEED,
        "smoke_step": args.smoke_step,
        "device": str(device),
        "platform": platform.platform(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "timm", "numpy", "opencv-python-headless")
        },
        "checkpoints": {},
    }
    (out / "training-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    metrics_path = out / "metrics.csv"
    best = {"number_accuracy": -1.0, "illegible_recall": -1.0}
    started, step = time.perf_counter(), 0
    for epoch in range(config["epochs"]):
        net.train()
        loader = torch.utils.data.DataLoader(
            make_dataset(
                epoch_records(taiseis_train, pusan_train, pool_train, config, rng, epoch),
                True,
                SEED,
                fonts,
            ),
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
        )
        losses = []
        for images, labels in loader:
            loss = criterion(net(images.to(device)), labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            step += 1
            if args.smoke_step:
                break
        valid = evaluate(net, valid_loader, device)
        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": float(np.mean(losses)),
            "lr": optimizer.param_groups[0]["lr"],
            **{k: v for k, v in valid.items() if k != "calibration_bins"},
            "elapsed_s": time.perf_counter() - started,
        }
        write_row(metrics_path, row)
        print(json.dumps({**row, "calibration_bins": valid["calibration_bins"]}), flush=True)
        state = {"model": net.state_dict(), "epoch": epoch, "step": step, "classes": NUM_CLASSES}
        torch.save(state, out / "last.pt")
        if improves(valid, best, config["illegible_recall_floor"]):
            best = {**row}
            torch.save(state, out / "best.pt")
        if args.smoke_step:
            break
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(state["model"])
    temperature, nll = fit_temperature(net, valid_loader, device, config["temperature_grid"])
    torch.save({**state, "temperature": temperature}, out / "best.pt")
    evidence.update(
        {
            "completed": True,
            "temperature": temperature,
            "valid_nll_at_temperature": nll,
            "optimizer_steps": step,
            "best_epoch": best.get("epoch"),
            "best_valid": {k: v for k, v in best.items() if k not in ("epoch", "step")},
            "elapsed_s": time.perf_counter() - started,
            "checkpoints": {
                name: {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size}
                for name in ("best.pt", "last.pt")
                if (out / name).exists()
            },
        }
    )
    (out / "training-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {k: evidence[k] for k in ("completed", "optimizer_steps", "best_epoch", "best_valid")},
            indent=2,
        )
    )


def write_row(path, row):
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--taiseis", type=Path, required=True, help="Roboflow classification folder export"
    )
    parser.add_argument("--pusan", type=Path, required=True, help="Pusan COCO digit-box export")
    parser.add_argument(
        "--torso-pool", type=Path, required=True, help="cut broadcast torsos (jersey-torso-pool)"
    )
    parser.add_argument("--fonts", type=Path, required=True, help="folder of OFL .ttf faces")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=TRAINING["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAINING["batch_size"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--smoke-step", action="store_true", help="One optimizer step, short validation"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
