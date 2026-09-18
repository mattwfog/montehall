"""Train the permissive pitch calibrator (vertex, line and conic heatmaps) on a Roboflow export.

Fixed protocol, no search: HRNet-W32 (timm, ImageNet init) with a 32 + 20 channel
heatmap head at 960 px, AdamW, cosine schedule. Vertices: spatial softmax
cross-entropy against Gaussian heatmaps for visible keypoints and against the
uniform distribution for invisible ones. Lines and conics: weighted per-pixel binary
cross-entropy against polylines rendered from each image's own fitted homography
(`pitch_keypoints.attach_primitive_samples`), so they need no labels. Horizontal
flips go through the geometric mirror pairs of both vertices and primitives; mild
affine and colour jitter. Every epoch appends to metrics.csv with keypoint error,
visible/invisible confidence separation, primitive pixel precision/recall, and the
result of running the real fit (`fit_from_maps`) on every valid image against its
fitted truth; best.pt is the lowest valid calibration error among epochs that clear
the visibility and acceptance floors. The external calibration benchmark is never
opened here.
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

import cv2
import numpy as np

from soccerviz.candidates.pitch_keypoints import (
    BACKBONE,
    DEFAULT_PROTOCOL,
    HEATMAP_SIZE,
    INPUT_SIZE,
    INVISIBLE_WEIGHT,
    NUM_KEYPOINTS,
    NUM_PRIMITIVES,
    PRETRAINED_TAG,
    PRIMITIVE_LOSS_WEIGHT,
    PRIMITIVE_POSITIVE_WEIGHT,
    attach_primitive_samples,
    build_network,
    decode_heatmaps,
    encode_heatmaps,
    encode_primitive_heatmaps,
    fit_from_maps,
    heatmap_loss,
    load_split,
    mirror_pairs,
    preprocess,
    primitive_loss,
)
from soccerviz.core.assets import sha256
from soccerviz.core.geometry import pitch_landmarks, project
from soccerviz.core.pitch_template import mirror_primitive_pairs, primitive_names

SEED = 20260908
TRAINING = {
    "epochs": 60,
    "batch_size": 4,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 2,
    "flip_probability": 0.5,
    "scale_range": [0.85, 1.15],
    "translate_fraction": 0.05,
    "brightness_jitter": 0.2,
    "contrast_jitter": 0.2,
    "pck_threshold_px": 10.0,
    "invisible_weight": INVISIBLE_WEIGHT,
    "visibility_floor": 0.95,
    "primitive_loss_weight": PRIMITIVE_LOSS_WEIGHT,
    "primitive_positive_weight": PRIMITIVE_POSITIVE_WEIGHT,
    "calibration_accept_floor": 0.95,
}


def flip_primitives(samples, width):
    """Mirror primitive samples: x reflected, classes swapped through their mirror pairs."""
    if samples is None:
        return None
    flipped = samples.copy()
    flipped[:, 0] = width - 1 - flipped[:, 0]
    remap = np.arange(NUM_PRIMITIVES)
    for i, j in mirror_primitive_pairs():
        remap[i], remap[j] = j, i
    flipped[:, 2] = remap[flipped[:, 2].astype(int)]
    return flipped


def warp_primitives(samples, matrix, width, height):
    """Apply the affine to primitive samples and keep the in-frame ones."""
    if samples is None:
        return None
    xy = samples[:, :2] @ matrix[:, :2].T + matrix[:, 2]
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    return np.column_stack([xy[inside], samples[inside, 2]]).astype(np.float32)


def augment(image, points, samples, rng, config):
    """Flip (mirror pairs), random scale/translate about the centre, colour jitter."""
    height, width = image.shape[:2]
    points = points.copy()
    if rng.random() < config["flip_probability"]:
        image = image[:, ::-1]
        points[:, 0] = width - 1 - points[:, 0]
        for i, j in mirror_pairs():
            points[[i, j]] = points[[j, i]]
        samples = flip_primitives(samples, width)
    scale = rng.uniform(*config["scale_range"])
    tx, ty = (rng.uniform(-1, 1) * config["translate_fraction"] * s for s in (width, height))
    matrix = np.array(
        [[scale, 0, (1 - scale) * width / 2 + tx], [0, scale, (1 - scale) * height / 2 + ty]],
        dtype=np.float32,
    )
    image = cv2.warpAffine(
        np.ascontiguousarray(image), matrix, (width, height), flags=cv2.INTER_LINEAR
    )
    xy = points[:, :2] @ matrix[:, :2].T + matrix[:, 2]
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    points[:, :2] = xy
    points[~inside, 2] = 0
    samples = warp_primitives(samples, matrix, width, height)
    alpha = 1 + rng.uniform(-1, 1) * config["contrast_jitter"]
    beta = rng.uniform(-1, 1) * config["brightness_jitter"] * 255
    image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return image, points, samples


def make_dataset(records, config, train, seed):
    import torch

    class PitchDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            record = records[index]
            image = cv2.imread(record["image_path"])
            if image is None:
                raise FileNotFoundError(record["image_path"])
            points, samples = record["keypoints"].copy(), record["primitives"]
            if train:
                # The global RNG is seeded once per run and per DataLoader worker.
                image, points, samples = augment(image, points, samples, random, config)
            height, width = image.shape[:2]
            heatmaps, weights = encode_heatmaps(points, width, height)
            primitives = encode_primitive_heatmaps(samples, width, height)
            return (
                torch.from_numpy(preprocess(image)),
                torch.from_numpy(heatmaps),
                torch.from_numpy(weights),
                torch.from_numpy(points),
                torch.tensor([width, height], dtype=torch.float32),
                torch.from_numpy(primitives),
                torch.tensor(0.0 if samples is None else 1.0),
            )

    return PitchDataset()


def evaluate(net, loader, device, threshold_px):
    """Keypoint error over visible keypoints, plus confidence separation of visible vs not.

    A keypoint counts as visible when the export marks it so and it lies inside the
    image, matching `encode_heatmaps`. Confidence is judged at the protocol's landmark
    threshold, the same cut `HeatmapCalibrationModel.predict` applies.
    """
    import torch

    net.eval()
    threshold = DEFAULT_PROTOCOL["landmark_confidence"]
    cut = DEFAULT_PROTOCOL["primitive_threshold"]
    logit_cut = math.log(cut / (1 - cut))
    errors, seen, unseen = [], [], []
    hits, positives, predicted_positive = 0, 0, 0
    accepted, calibration_errors = 0, []
    with torch.inference_mode():
        for images, _, _, points, sizes, targets, mask in loader:
            heatmaps = net(images.to(device)).float().cpu().numpy()
            for k in range(len(images)):
                width, height = sizes[k].tolist()
                predicted, confidence = decode_heatmaps(heatmaps[k, :NUM_KEYPOINTS], width, height)
                truth = points[k].numpy()
                if float(mask[k]) > 0:
                    target = targets[k].numpy() > 0.5
                    prob = heatmaps[k, NUM_KEYPOINTS:] > logit_cut
                    hits += int((target & prob).sum())
                    positives += int(target.sum())
                    predicted_positive += int(prob.sum())
                    fit, _, _, _ = fit_from_maps(
                        heatmaps[k], width, height, DEFAULT_PROTOCOL, NUM_PRIMITIVES
                    )
                    if fit.accepted:
                        accepted += 1
                        visible = truth[:, 2] > 0
                        world = project(truth[visible, :2], fit.matrix)
                        calibration_errors.append(
                            float(
                                np.median(
                                    np.linalg.norm(world - pitch_landmarks()[visible], axis=1)
                                )
                            )
                        )
                inside = (
                    (truth[:, 0] >= 0)
                    & (truth[:, 0] < width)
                    & (truth[:, 1] >= 0)
                    & (truth[:, 1] < height)
                )
                visible = (truth[:, 2] > 0) & inside
                errors.extend(np.linalg.norm(predicted[visible] - truth[visible, :2], axis=1))
                seen.extend(confidence[visible])
                unseen.extend(confidence[~visible])
    errors, seen, unseen = (np.asarray(v, dtype=float) for v in (errors, seen, unseen))
    correct = int((seen >= threshold).sum() + (unseen < threshold).sum())
    return {
        "visible_keypoints": len(errors),
        "mean_error_px": float(errors.mean()) if len(errors) else None,
        "median_error_px": float(np.median(errors)) if len(errors) else None,
        f"pck_{int(threshold_px)}px": float((errors <= threshold_px).mean())
        if len(errors)
        else None,
        "invisible_keypoints": len(unseen),
        "visible_confidence_median": float(np.median(seen)) if len(seen) else None,
        "invisible_confidence_median": float(np.median(unseen)) if len(unseen) else None,
        "invisible_above_threshold": float((unseen >= threshold).mean()) if len(unseen) else None,
        "visibility_accuracy": correct / max(1, len(seen) + len(unseen)),
        "primitive_precision": hits / predicted_positive if predicted_positive else 0.0,
        "primitive_recall": hits / positives if positives else 0.0,
        "calibration_images": len(loader.dataset),
        "calibration_accepted": accepted,
        "calibration_accept_fraction": accepted / max(1, len(loader.dataset)),
        "calibration_error_m": float(np.median(calibration_errors)) if calibration_errors else None,
    }


def clears(row, visibility_floor, accept_floor):
    return (
        row["visibility_accuracy"] >= visibility_floor
        and row["calibration_accept_fraction"] >= accept_floor
    )


def improves(candidate, best, visibility_floor, accept_floor):
    """best.pt rule: lowest valid calibration error among epochs clearing both floors.

    The floors are the visibility accuracy (no phantom landmarks, the v18 failure)
    and the fraction of valid images the real fit accepts. Until an epoch clears
    both, the lowest calibration error outright, then mean keypoint error as the
    tie-break when no image is accepted at all.
    """
    if candidate["mean_error_px"] is None:
        return False
    candidate_ok, best_ok = (
        clears(candidate, visibility_floor, accept_floor),
        clears(best, visibility_floor, accept_floor),
    )
    if candidate_ok and not best_ok:
        return True
    if best_ok and not candidate_ok:
        return False
    if candidate["calibration_error_m"] is None:
        return best["calibration_error_m"] is None and (
            candidate["mean_error_px"] < best["mean_error_px"]
        )
    if best["calibration_error_m"] is None:
        return True
    return candidate["calibration_error_m"] < best["calibration_error_m"]


def run(args):
    import torch

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    config = {**TRAINING, "epochs": args.epochs, "batch_size": args.batch_size}
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    train_records, valid_records = (
        attach_primitive_samples(load_split(args.export, "train")),
        attach_primitive_samples(load_split(args.export, "valid")),
    )
    unfit = sum(r["primitives"] is None for r in train_records + valid_records)
    train_loader = torch.utils.data.DataLoader(
        make_dataset(train_records, config, True, SEED),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        make_dataset(valid_records, config, False, SEED),
        batch_size=config["batch_size"],
        num_workers=args.workers,
    )
    device = torch.device(args.device)
    net = build_network(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    steps_per_epoch = len(train_loader)
    total_steps = max(1, config["epochs"] * steps_per_epoch)
    warmup = config["warmup_epochs"] * steps_per_epoch

    def lr_at(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    evidence = {
        "schema": "pitch-keypoint-training-evidence/v2",
        "completed": False,
        "export": str(args.export),
        "export_annotations_sha256": {
            split: sha256(Path(args.export) / split / "_annotations.coco.json")
            for split in ("train", "valid")
        },
        "train_images": len(train_records),
        "valid_images": len(valid_records),
        "images_without_fitted_truth": unfit,
        "primitives": NUM_PRIMITIVES,
        "primitive_names": primitive_names(),
        "selection": "lowest valid calibration_error_m among epochs clearing "
        "visibility_floor and calibration_accept_floor",
        "backbone": BACKBONE,
        "initialization": "random" if args.no_pretrained else PRETRAINED_TAG,
        "input_size": INPUT_SIZE,
        "heatmap_size": HEATMAP_SIZE,
        "keypoints": NUM_KEYPOINTS,
        "training": config,
        "seed": SEED,
        "smoke_step": args.smoke_step,
        "optimizer_steps": 0,
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
    best = {
        "mean_error_px": math.inf,
        "visibility_accuracy": -1.0,
        "calibration_accept_fraction": -1.0,
        "calibration_error_m": None,
    }
    started = time.perf_counter()
    step = 0
    for epoch in range(config["epochs"]):
        net.train()
        losses = []
        for images, heatmaps, weights, _, _, targets, mask in train_loader:
            images, heatmaps, weights = images.to(device), heatmaps.to(device), weights.to(device)
            targets, mask = targets.to(device), mask.to(device)
            predicted = net(images)
            loss = heatmap_loss(
                predicted[:, :NUM_KEYPOINTS], heatmaps, weights, config["invisible_weight"]
            ) + config["primitive_loss_weight"] * primitive_loss(
                predicted[:, NUM_KEYPOINTS:], targets, mask, config["primitive_positive_weight"]
            )
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
        valid = evaluate(net, valid_loader, device, config["pck_threshold_px"])
        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": float(np.mean(losses)),
            "lr": optimizer.param_groups[0]["lr"],
            **valid,
            "elapsed_s": time.perf_counter() - started,
        }
        write_row(metrics_path, row)
        print(json.dumps(row), flush=True)
        state = {
            "model": net.state_dict(),
            "epoch": epoch,
            "step": step,
            "num_primitives": NUM_PRIMITIVES,
        }
        torch.save(state, out / "last.pt")
        if improves(valid, best, config["visibility_floor"], config["calibration_accept_floor"]):
            best = {**row}
            torch.save(state, out / "best.pt")
        if args.smoke_step:
            break
    evidence.update(
        {
            "completed": True,
            "optimizer_steps": step,
            "best_epoch": best.get("epoch"),
            "best_valid": {k: v for k, v in best.items() if k not in ("epoch", "step")},
            "elapsed_s": time.perf_counter() - started,
            "checkpoints": {
                name: {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size}
                for name in ("best.pt", "last.pt")
                if (out / name).exists()
            },
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
            if device.type == "cuda"
            else None,
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
        "--export", type=Path, required=True, help="Roboflow COCO keypoint export folder"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=TRAINING["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAINING["batch_size"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-pretrained", action="store_true", help="Random init (offline smoke)")
    parser.add_argument(
        "--smoke-step", action="store_true", help="One optimizer step, one evaluation"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
