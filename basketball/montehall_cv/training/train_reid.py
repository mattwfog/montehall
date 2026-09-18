"""OSNet ReID trained FROM SCRATCH on identity-crop folders (design D7).

torchreid (MIT) provides only the OSNet architecture; weights start random —
no published ReID checkpoint is license-clean for commercial use, which is
the whole reason this trains from scratch. Objective is identity softmax
(classification baseline); evaluation is rank-1 retrieval over held-out
identities (query = first crop per identity, gallery = the rest), since
classification accuracy on train identities says nothing about transfer.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

VAL_IDENTITY_FRACTION = 0.1
INPUT_HW = (256, 128)


def train(dataset_dir: Path, out_dir: Path, epochs: int = 60, batch_size: int = 64,
          lr: float = 3e-4, arch: str = "osnet_x1_0") -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchreid.models import build_model
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    identities = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    if len(identities) < 20:
        raise ValueError(f"only {len(identities)} identities under {dataset_dir}")
    rng = random.Random(0)
    rng.shuffle(identities)
    n_val = max(int(len(identities) * VAL_IDENTITY_FRACTION), 5)
    val_ids, train_ids = identities[:n_val], identities[n_val:]

    train_items = [
        (p, idx) for idx, ident in enumerate(train_ids) for p in sorted(ident.glob("*.jpg"))
    ]
    val_groups = [sorted(ident.glob("*.jpg")) for ident in val_ids]
    val_groups = [g for g in val_groups if len(g) >= 2]

    tf_train = transforms.Compose([
        transforms.Resize(INPUT_HW),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        transforms.RandomCrop(INPUT_HW, padding=8),
        transforms.ToTensor(),
    ])
    tf_eval = transforms.Compose([transforms.Resize(INPUT_HW), transforms.ToTensor()])

    class Crops(Dataset):
        def __init__(self, items, tf):
            self.items, self.tf = items, tf

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, label = self.items[i]
            return self.tf(Image.open(path).convert("RGB")), label

    model = build_model(arch, num_classes=len(train_ids), pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    loader = DataLoader(Crops(train_items, tf_train), batch_size=batch_size,
                        shuffle=True, num_workers=4, drop_last=True)

    def embed(paths: list[Path]) -> "np.ndarray":
        model.eval()
        feats = []
        with torch.no_grad():
            for start in range(0, len(paths), batch_size):
                batch = torch.stack([
                    tf_eval(Image.open(p).convert("RGB"))
                    for p in paths[start:start + batch_size]
                ]).to(device)
                f = model(batch)  # eval mode -> feature embeddings
                feats.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        return np.concatenate(feats)

    def rank1() -> float:
        queries, gallery, q_labels, g_labels = [], [], [], []
        for label, group in enumerate(val_groups):
            queries.append(group[0])
            q_labels.append(label)
            gallery.extend(group[1:])
            g_labels.extend([label] * (len(group) - 1))
        qf, gf = embed(queries), embed(gallery)
        hits = sum(
            int(g_labels[int(np.argmax(gf @ qf[i]))] == q_labels[i])
            for i in range(len(queries))
        )
        return hits / len(queries)

    started = time.monotonic()
    best = 0.0
    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            opt.step()
            epoch_loss += float(loss)
        sched.step()
        r1 = rank1()
        history.append({"epoch": epoch + 1, "loss": round(epoch_loss / max(len(loader), 1), 4),
                        "val_rank1": round(r1, 4)})
        print(json.dumps(history[-1]), flush=True)
        if r1 > best:
            best = r1
            torch.save(model.state_dict(), out_dir / f"{arch}_best.pt")
    torch.save(model.state_dict(), out_dir / f"{arch}_last.pt")
    summary = {
        "arch": arch,
        "train_identities": len(train_ids),
        "train_crops": len(train_items),
        "val_identities": len(val_groups),
        "best_val_rank1": round(best, 4),
        "epochs": epochs,
        "wall_min": round((time.monotonic() - started) / 60, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="dir of <identity>/<crop>.jpg folders")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--arch", default="osnet_x1_0")
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.out, epochs=args.epochs,
                           batch_size=args.batch_size, arch=args.arch), indent=2))


if __name__ == "__main__":
    main()
