"""Jersey models from the synthetic set: legibility gate + digit classifier.

Two small torchvision models (BSD weights, license-clean):
- legibility: ResNet34 binary (the Koshkina-recipe gate, re-implemented)
- digits: ResNet34 100-class (00-99) on legible crops only

Both train from the synth_jersey labels.jsonl. Small enough to train in
minutes on the GB10; checkpoints land beside the dataset.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def train(dataset_dir: Path, out_dir: Path, epochs: int = 6, batch_size: int = 128,
          scratch: bool = False) -> dict:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in (dataset_dir / "labels.jsonl").open()]
    split = int(len(rows) * 0.9)
    train_rows, val_rows = rows[:split], rows[split:]

    tf_train = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
        ]
    )
    tf_val = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    class CropSet(Dataset):
        def __init__(self, subset: list[dict], task: str, tf) -> None:
            self.task = task
            self.tf = tf
            if task == "legibility":
                self.items = [(r["file"], int(r["legible"])) for r in subset]
            else:
                self.items = [
                    (r["file"], int(r["label"])) for r in subset if r["legible"]
                ]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, i):
            name, label = self.items[i]
            img = Image.open(dataset_dir / name).convert("RGB")
            return self.tf(img), label

    results: dict[str, object] = {"init": "scratch" if scratch else "imagenet"}
    for task, n_classes in (("legibility", 2), ("digits", 100)):
        weights = None if scratch else models.ResNet34_Weights.IMAGENET1K_V1
        model = models.resnet34(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        loss_fn = torch.nn.CrossEntropyLoss()

        train_dl = DataLoader(
            CropSet(train_rows, task, tf_train),
            batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True,
        )
        val_dl = DataLoader(
            CropSet(val_rows, task, tf_val), batch_size=batch_size, num_workers=2
        )

        started = time.monotonic()
        best_acc = 0.0
        for epoch in range(epochs):
            model.train()
            for images, labels in train_dl:
                images, labels = images.to(device), labels.to(device)
                opt.zero_grad()
                loss = loss_fn(model(images), labels)
                loss.backward()
                opt.step()
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels in val_dl:
                    preds = model(images.to(device)).argmax(1).cpu()
                    correct += int((preds == labels).sum())
                    total += len(labels)
            acc = correct / max(total, 1)
            print(f"{task} epoch {epoch + 1}/{epochs} val_acc={acc:.4f}", flush=True)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), out_dir / f"{task}_resnet34.pt")
        results[task] = {
            "best_val_acc": round(best_acc, 4),
            "train_items": len(train_dl.dataset),
            "val_items": len(val_dl.dataset),
            "wall_min": round((time.monotonic() - started) / 60, 1),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--scratch", action="store_true",
                        help="random init instead of ImageNet weights (license-clean; "
                             "needs more epochs to converge)")
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.out, args.epochs, scratch=args.scratch),
                     indent=2))


if __name__ == "__main__":
    main()
