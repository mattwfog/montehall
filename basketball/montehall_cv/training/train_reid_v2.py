"""OSNet ReID from scratch, recipe v2 (design D7/D9 iteration).

Same data contract and rank-1 protocol as train_reid.py (query = first crop
per held-out identity, gallery = the rest) so results stay comparable with
the recipe-v1 baseline. Adds the from-scratch ReID staples the baseline
lacks: RandomErasing, LR warmup before cosine decay, and optionally
batch-hard triplet loss over P identities x K instances (--triplet).
Extra readouts (mAP, flip-TTA rank-1) never drive checkpoint selection —
best.pt is still picked on the plain rank-1 the baseline reports.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

VAL_IDENTITY_FRACTION = 0.1
INPUT_HW = (256, 128)


def pinned_val_split(
    identities: list[Path], pinned_names: set[str]
) -> tuple[list[Path], list[Path]]:
    """(val_ids, train_ids) with val pinned to named identity dirs.

    Benchmark comparability guard: training on a merged pool with the
    default Random(0) split scrambles membership and leaks benchmark val
    ids into train (the 2026-07-09 reid-combined 0.8609 was contaminated
    54/61). Pinning val to the benchmark's own held-out ids keeps rank-1
    comparable across data mixes. Every pinned name must exist."""
    missing = pinned_names - {d.name for d in identities}
    if missing:
        raise ValueError(
            f"{len(missing)} pinned val ids absent from dataset, "
            f"e.g. {sorted(missing)[:3]}"
        )
    val_ids = [d for d in identities if d.name in pinned_names]
    train_ids = [d for d in identities if d.name not in pinned_names]
    return val_ids, train_ids


def train(dataset_dir: Path, out_dir: Path, epochs: int = 150, batch_size: int = 64,
          lr: float = 3e-4, arch: str = "osnet_x1_0", use_triplet: bool = True,
          warmup_epochs: int = 10, erase_p: float = 0.5, k_instances: int = 4,
          hflip: bool = True, soft_margin: bool = False, triplet_start: int = -1,
          val_ids_file: Path | None = None,
          smoke: bool = False) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, Sampler
    from torchreid.models import build_model
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    identities = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    if len(identities) < 20:
        raise ValueError(f"only {len(identities)} identities under {dataset_dir}")
    rng = random.Random(0)  # same seed as recipe v1 -> identical val split
    rng.shuffle(identities)
    if val_ids_file is not None:
        pinned = {
            ln.strip() for ln in val_ids_file.read_text().splitlines() if ln.strip()
        }
        val_ids, train_ids = pinned_val_split(identities, pinned)
    else:
        n_val = max(int(len(identities) * VAL_IDENTITY_FRACTION), 5)
        val_ids, train_ids = identities[:n_val], identities[n_val:]
    if smoke:
        train_ids, epochs, warmup_epochs = train_ids[:40], 2, 1

    train_items = [
        (p, idx) for idx, ident in enumerate(train_ids) for p in sorted(ident.glob("*.jpg"))
    ]
    val_groups = [sorted(ident.glob("*.jpg")) for ident in val_ids]
    val_groups = [g for g in val_groups if len(g) >= 2]

    # Jersey digits are chirality-sensitive (a mirrored "12" is not a "12"),
    # so horizontal flip may erase the strongest identity cue — flag-gated.
    aug: list = [transforms.Resize(INPUT_HW)]
    if hflip:
        aug.append(transforms.RandomHorizontalFlip())
    aug += [
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        transforms.RandomCrop(INPUT_HW, padding=8),
        transforms.ToTensor(),
        transforms.RandomErasing(p=erase_p, scale=(0.02, 0.2)),
    ]
    tf_train = transforms.Compose(aug)
    tf_eval = transforms.Compose([transforms.Resize(INPUT_HW), transforms.ToTensor()])

    class Crops(Dataset):
        def __init__(self, items, tf):
            self.items, self.tf = items, tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, label = self.items[i]
            return self.tf(Image.open(path).convert("RGB")), label

    class PKSampler(Sampler):
        """Each batch = P random identities x K crops (with replacement when
        an identity has fewer than K), so batch-hard triplets always exist."""

        def __init__(self, items, p_identities, k):
            self.by_label: dict[int, list[int]] = {}
            for i, (_, label) in enumerate(items):
                self.by_label.setdefault(label, []).append(i)
            self.labels = list(self.by_label)
            self.p, self.k = p_identities, k
            self.batches = max(len(items) // (p_identities * k), 1)

        def __len__(self):
            return self.batches * self.p * self.k

        def __iter__(self):
            for _ in range(self.batches):
                for label in random.sample(self.labels, min(self.p, len(self.labels))):
                    pool = self.by_label[label]
                    picks = (random.sample(pool, self.k) if len(pool) >= self.k
                             else random.choices(pool, k=self.k))
                    yield from picks

    def batch_hard_triplet(feats: "torch.Tensor", labels: "torch.Tensor",
                           margin: float = 0.3, soft: bool = False) -> "torch.Tensor":
        dist = torch.cdist(feats, feats)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        hardest_pos = dist.masked_fill(~same | eye, float("-inf")).amax(dim=1)
        hardest_neg = dist.masked_fill(same, float("inf")).amin(dim=1)
        valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
        if not valid.any():
            return feats.new_zeros(())
        gap = hardest_pos[valid] - hardest_neg[valid]
        if soft:
            # softplus(gap): nonzero gradient even at the collapsed fixed
            # point where relu(gap+margin) plateaus at exactly `margin`
            # (rounds 4+5 both froze there — loss 6.635 = ln(565) CE + 0.3)
            return torch.nn.functional.softplus(gap).mean()
        return torch.relu(gap + margin).mean()

    if triplet_start < 0:
        triplet_start = warmup_epochs
    loss_name = "triplet" if use_triplet else "softmax"
    model = build_model(arch, num_classes=len(train_ids), loss=loss_name,
                        pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs - warmup_epochs, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [warmup, cosine], milestones=[warmup_epochs])
    ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    if use_triplet:
        p_identities = max(batch_size // k_instances, 2)
        sampler = PKSampler(train_items, p_identities, k_instances)
        loader = DataLoader(Crops(train_items, tf_train),
                            batch_size=p_identities * k_instances,
                            sampler=sampler, num_workers=4, drop_last=True)
    else:
        loader = DataLoader(Crops(train_items, tf_train), batch_size=batch_size,
                            shuffle=True, num_workers=4, drop_last=True)

    def embed(paths: list[Path], flip: bool = False) -> "np.ndarray":
        model.eval()
        feats = []
        with torch.no_grad():
            for start in range(0, len(paths), batch_size):
                batch = torch.stack([
                    tf_eval(Image.open(p).convert("RGB"))
                    for p in paths[start:start + batch_size]
                ]).to(device)
                f = model(batch)
                if flip:
                    f = f + model(torch.flip(batch, dims=[3]))
                feats.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        return np.concatenate(feats)

    def retrieval(flip: bool = False) -> dict:
        queries, gallery, q_labels, g_labels = [], [], [], []
        for label, group in enumerate(val_groups):
            queries.append(group[0])
            q_labels.append(label)
            gallery.extend(group[1:])
            g_labels.extend([label] * (len(group) - 1))
        qf, gf = embed(queries, flip=flip), embed(gallery, flip=flip)
        g_labels_arr = np.array(g_labels)
        hits, ap_sum = 0, 0.0
        for i in range(len(queries)):
            order = np.argsort(-(gf @ qf[i]))
            rel = g_labels_arr[order] == q_labels[i]
            hits += int(rel[0])
            n_rel = int(rel.sum())
            if n_rel:
                precision_at = np.cumsum(rel) / (np.arange(len(rel)) + 1)
                ap_sum += float((precision_at * rel).sum() / n_rel)
        return {"rank1": hits / len(queries), "mAP": ap_sum / len(queries)}

    started = time.monotonic()
    best = 0.0
    history = []
    # Self-abort on non-learning runs: rounds 4+5 burned 55+40 epochs pinned
    # at chance (rank-1 == 1/n_val_ids) before a human noticed. If the best
    # rank-1 is still <=2x chance well past warmup (and past any triplet
    # ramp-in), the run is dead — stop burning GPU so a queue advances.
    chance = 1.0 / max(len(val_groups), 1)
    abort_after = max(warmup_epochs, triplet_start if use_triplet else 0) + 10
    aborted = None
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            out = model(images)
            if use_triplet:
                logits, feats = out
                loss = ce(logits, labels)
                # CE-only warm start: let CE organize the embedding space
                # before batch-hard mining bites (mining random features is
                # the collapse trigger)
                if epoch >= triplet_start:
                    loss = loss + batch_hard_triplet(feats, labels, soft=soft_margin)
            else:
                loss = ce(out, labels)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        metrics = retrieval()
        history.append({"epoch": epoch + 1,
                        "loss": round(epoch_loss / max(len(loader), 1), 4),
                        "val_rank1": round(metrics["rank1"], 4),
                        "val_mAP": round(metrics["mAP"], 4),
                        "lr": round(opt.param_groups[0]["lr"], 6)})
        print(json.dumps(history[-1]), flush=True)
        if metrics["rank1"] > best:
            best = metrics["rank1"]
            torch.save(model.state_dict(), out_dir / f"{arch}_best.pt")
        if epoch + 1 >= abort_after and best <= 2 * chance:
            aborted = (f"rank-1 {best:.4f} <= 2x chance ({chance:.4f}) at epoch "
                       f"{epoch + 1} — model never learned, self-aborting")
            print(json.dumps({"ABORTED": aborted}), flush=True)
            break
    torch.save(model.state_dict(), out_dir / f"{arch}_last.pt")
    tta = retrieval(flip=True)
    summary = {
        "arch": arch,
        "recipe": {"triplet": use_triplet, "soft_margin": soft_margin,
                   "triplet_start": triplet_start if use_triplet else None,
                   "warmup_epochs": warmup_epochs,
                   "erase_p": erase_p, "hflip": hflip,
                   "k_instances": k_instances if use_triplet else None,
                   "val_ids_file": str(val_ids_file) if val_ids_file else None},
        "aborted": aborted,
        "train_identities": len(train_ids),
        "train_crops": len(train_items),
        "val_identities": len(val_groups),
        "best_val_rank1": round(best, 4),
        "final_val_mAP": round(history[-1]["val_mAP"], 4),
        "final_tta_rank1": round(tta["rank1"], 4),
        "final_tta_mAP": round(tta["mAP"], 4),
        "epochs": epochs,
        "wall_min": round((time.monotonic() - started) / 60, 1),
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="dir of <identity>/<crop>.jpg folders")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--arch", default="osnet_x1_0")
    parser.add_argument("--triplet", action="store_true",
                        help="CE + batch-hard triplet over PxK batches")
    parser.add_argument("--soft-margin", action="store_true",
                        help="softplus triplet (gradient survives the collapse plateau)")
    parser.add_argument("--triplet-start", type=int, default=-1,
                        help="epoch to enable the triplet term (default: after warmup)")
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--erase-p", type=float, default=0.5)
    parser.add_argument("--k-instances", type=int, default=4)
    parser.add_argument("--val-ids", type=Path, default=None,
                        help="file of identity dir names pinned as val — "
                             "keeps rank-1 comparable to a benchmark split "
                             "when training on a merged pool")
    parser.add_argument("--no-flip", action="store_true",
                        help="drop RandomHorizontalFlip (jersey digits are chiral)")
    parser.add_argument("--smoke", action="store_true",
                        help="2 epochs on 40 identities to validate the rig")
    args = parser.parse_args()
    summary = train(args.dataset, args.out, epochs=args.epochs,
                    batch_size=args.batch_size, arch=args.arch,
                    use_triplet=args.triplet, warmup_epochs=args.warmup_epochs,
                    erase_p=args.erase_p, k_instances=args.k_instances,
                    hflip=not args.no_flip, soft_margin=args.soft_margin,
                    triplet_start=args.triplet_start, val_ids_file=args.val_ids,
                    smoke=args.smoke)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
