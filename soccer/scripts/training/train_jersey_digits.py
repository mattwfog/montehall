"""Train an RF-DETR digit detector on a Roboflow jersey-number export (ten classes 0..9).

The permissive replacement for the Uncertainty-JNR jersey reader (CC BY-NC-SA, SoccerNet
weights): rfdetr (Apache-2.0) initialised from the pinned COCO Medium checkpoint, on the
CC BY 4.0 `pusan-national-university-aajlj/jersey-number-detection-8a55j` v1 export.
Fixed protocol, no search; best checkpoint by the export's own valid split. The
labelled SoccerNet crops are never opened here.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from soccerviz.core.assets import sha256

SEED = 20260908
DIGITS = tuple(str(d) for d in range(10))
INIT_SHA256 = "749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8"
# 2026-09-08: the smoke (576 px, batch 8 x 2 accumulation, 2 loader workers) reached
# valid mAP50 0.954 in one epoch at 4.8 crops/s, 84 min per 24,315-crop epoch. Jersey
# crops are a few hundred pixels, so 448 px keeps digits large; batch 16 without
# accumulation is the same effective batch; 8 workers match the container's CPUs.
TRAINING = {
    "epochs": 4,
    "batch_size": 16,
    "grad_accum_steps": 1,
    "num_workers": 8,
    "lr": 1e-4,
    "lr_encoder": 1.5e-4,
    "resolution": 448,
    "warmup_epochs": 0.5,
    "ema_decay": 0.993,
    "checkpoint_interval": 1,
}


def export_categories(export):
    data = json.loads((Path(export) / "train/_annotations.coco.json").read_text())
    names = [c["name"] for c in sorted(data["categories"], key=lambda c: c["id"])]
    # Roboflow puts the project-level placeholder class first; the digits follow it.
    digits = [n for n in names if n in DIGITS]
    if tuple(digits) != DIGITS:
        raise ValueError(f"Export digit classes are {digits}, expected {DIGITS}")
    return names


def run(args):
    import torch
    from rfdetr import RFDETRMedium

    torch.manual_seed(SEED)
    export = args.export.resolve()
    for split in ("train", "valid", "test"):
        if not (export / split / "_annotations.coco.json").exists():
            raise FileNotFoundError(
                f"{export / split}: rfdetr's roboflow loader needs all three splits"
            )
    if sha256(args.checkpoint) != INIT_SHA256:
        raise ValueError("Initialization checkpoint hash mismatch")
    if args.out.exists():
        raise FileExistsError(
            "Training output already exists; completed and partial runs are preserved"
        )
    args.out.mkdir(parents=True)
    names = export_categories(export)
    config = {**TRAINING, "epochs": 1 if args.smoke else args.epochs}
    evidence = {
        "schema": "jersey-digit-training-evidence/v1",
        "completed": False,
        "export": str(export),
        "export_annotations_sha256": {
            split: sha256(export / split / "_annotations.coco.json") for split in ("train", "valid")
        },
        "class_names": names,
        "initialization_sha256": INIT_SHA256,
        "training": config,
        "seed": SEED,
        "smoke": args.smoke,
        "device": args.device,
        "platform": platform.platform(),
        "versions": {p: importlib.metadata.version(p) for p in ("rfdetr", "torch", "torchvision")},
    }
    (args.out / "training-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    started = time.perf_counter()
    model = RFDETRMedium(
        pretrain_weights=str(args.checkpoint),
        device=args.device,
        resolution=config["resolution"],
        num_classes=len(names),
    )
    model.train(
        dataset_dir=str(export),
        dataset_file="roboflow",
        output_dir=str(args.out),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        grad_accum_steps=config["grad_accum_steps"],
        num_workers=config["num_workers"],
        lr=config["lr"],
        lr_encoder=config["lr_encoder"],
        warmup_epochs=config["warmup_epochs"],
        ema_decay=config["ema_decay"],
        checkpoint_interval=config["checkpoint_interval"],
        seed=SEED,
        class_names=names,
        early_stopping=False,
        tensorboard=False,
        wandb=False,
    )
    checkpoints = {
        p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(args.out.glob("*.pth"))
    }
    evidence.update(
        {
            "completed": True,
            "elapsed_s": time.perf_counter() - started,
            "checkpoints": checkpoints,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
            if args.device.startswith("cuda")
            else None,
        }
    )
    (args.out / "training-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: evidence[k] for k in ("completed", "elapsed_s", "checkpoints")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="pinned rf-detr-medium.pth")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=TRAINING["epochs"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="One epoch, separate output")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
