"""Cross-evaluate ReID checkpoints on one dataset's deterministic val split.

Settles "which checkpoint is champion" when candidates trained on different
datasets: rebuilds --dataset's val split exactly as train_reid_v2 does
(sorted identity dirs, Random(0) shuffle, 10% held out) and scores every
--weights checkpoint on that identical query/gallery protocol.

Leakage guard: a checkpoint trained on a SUPERSET dataset (e.g. 626+own
merged) may have seen some of this val split's identities in training —
--leak-check rebuilds that dataset's split the same way and reports which
eval-val identities sat in its train half, plus metrics restricted to the
clean subset. Identity matching is by folder name.

CPU-safe: OSNet is small; this runs on a busy box's CPUs while the GPU
trains (never touch CUDA next to a training run — pass --device cpu).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

VAL_IDENTITY_FRACTION = 0.1
INPUT_HW = (256, 128)


def val_split(dataset_dir: Path) -> tuple[list[Path], list[Path]]:
    """(val_ids, train_ids) exactly as train_reid_v2.train() builds them."""
    identities = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    rng = random.Random(0)
    rng.shuffle(identities)
    n_val = max(int(len(identities) * VAL_IDENTITY_FRACTION), 5)
    return identities[:n_val], identities[n_val:]


def evaluate(dataset: Path, weights: list[str], arch: str = "osnet_x1_0",
             device_name: str = "cpu", batch_size: int = 64,
             leak_check: Path | None = None) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from torchreid.models import build_model
    from torchvision import transforms

    device = torch.device(device_name)
    tf_eval = transforms.Compose([transforms.Resize(INPUT_HW), transforms.ToTensor()])

    val_ids, _ = val_split(dataset)
    val_groups = [(ident.name, sorted(ident.glob("*.jpg"))) for ident in val_ids]
    val_groups = [(name, g) for name, g in val_groups if len(g) >= 2]

    leaked: set[str] = set()
    if leak_check is not None:
        _, leak_train = val_split(leak_check)
        leak_train_names = {d.name for d in leak_train}
        leaked = {name for name, _ in val_groups if name in leak_train_names}

    def load_checkpoint(path: Path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        n_classes = state["classifier.weight"].shape[0]
        model = build_model(arch, num_classes=n_classes, loss="softmax",
                            pretrained=False)
        model.load_state_dict(state)
        return model.to(device).eval()

    def embed(model, paths):
        feats = []
        with torch.no_grad():
            for start in range(0, len(paths), batch_size):
                batch = torch.stack([
                    tf_eval(Image.open(p).convert("RGB"))
                    for p in paths[start:start + batch_size]
                ]).to(device)
                f = model(batch)
                feats.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
        return np.concatenate(feats)

    def retrieval(model, groups):
        queries, gallery, q_labels, g_labels = [], [], [], []
        for label, (_, group) in enumerate(groups):
            queries.append(group[0])
            q_labels.append(label)
            gallery.extend(group[1:])
            g_labels.extend([label] * (len(group) - 1))
        qf, gf = embed(model, queries), embed(model, gallery)
        g_arr = np.array(g_labels)
        hits, ap_sum = 0, 0.0
        for i in range(len(queries)):
            order = np.argsort(-(gf @ qf[i]))
            rel = g_arr[order] == q_labels[i]
            hits += int(rel[0])
            n_rel = int(rel.sum())
            if n_rel:
                precision_at = np.cumsum(rel) / (np.arange(len(rel)) + 1)
                ap_sum += float((precision_at * rel).sum() / n_rel)
        return {"rank1": round(hits / len(queries), 4),
                "mAP": round(ap_sum / len(queries), 4)}

    clean_groups = [(n, g) for n, g in val_groups if n not in leaked]
    results = {
        "dataset": str(dataset),
        "val_identities": len(val_groups),
        "leaked_identities": sorted(leaked),
        "clean_val_identities": len(clean_groups),
        "checkpoints": {},
    }
    for spec in weights:
        name, _, path = spec.partition("=")
        model = load_checkpoint(Path(path))
        entry = {"full_val": retrieval(model, val_groups)}
        if leaked and clean_groups:
            entry["leak_free_val"] = retrieval(model, clean_groups)
        results["checkpoints"][name] = entry
        print(json.dumps({name: entry}), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="identity-folder dataset whose val split is the benchmark")
    parser.add_argument("--weights", nargs="+", required=True,
                        help="name=path.pt pairs to score")
    parser.add_argument("--arch", default="osnet_x1_0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--leak-check", type=Path, default=None,
                        help="dataset a candidate trained on; reports val ids "
                             "that sat in its train split + leak-free metrics")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    results = evaluate(args.dataset, args.weights, args.arch, args.device,
                       args.batch_size, args.leak_check)
    print(json.dumps(results, indent=2))
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
