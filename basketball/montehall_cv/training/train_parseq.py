"""Fine-tune PARSeq on synthetic jersey crops (D9 recall lever).

Zero-shot PARSeq is precision-gated in the pipeline (per-char floors +
2-agreeing-reads) which leaves a recall gap: most legible crops never
clear the bars. This fine-tunes the same hub checkpoint the pipeline
loads (baudm/parseq, Apache-2.0) on synth_jersey.py output so the
confidence mass moves onto 0-99 digit strings.

Contract with pipeline/jersey.py (must never drift): identical hub model,
identical preprocessing (bilinear resize to hparams.img_size, /255,
(x-0.5)/0.5), identical tokenizer decode. The output state_dict loads
onto a hub-instantiated model as-is.

Standalone: torch + PIL + numpy, plus PARSeq's own import-time deps
(pytorch-lightning, timm, nltk). No lightning Trainer, no lmdb, no hydra —
plain loop calling the module's training_step with self.log muted.

--abstain trains the illegible synth crops as empty-string targets
(immediate EOS) so the OCR refuses garbage instead of hallucinating a
confident digit (the first fine-tune pushed illegible min-conf to 0.87,
above the pipeline's 0.85 trust bar). Empty decode -> pipeline _accept
rejects the read with no confidence bar involved.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_rows(
    dataset_dirs: list[Path], repeats: list[int] | None = None
) -> tuple[list[dict], list[dict]]:
    """labels.jsonl per dir -> (legible rows with digit labels, illegible rows).

    Each row gains an absolute "path" so multiple generator outputs (e.g.
    synthjersey-v0 + a harder v1) train as one pool, and a "_repeat" factor
    (per source dir) consumed AFTER the val split — tiny real-crop harvests
    (build_jersey_real) would otherwise dilute into tens of thousands of
    synth rows. Repeating before the split would leak copies of one crop
    into both train and val."""
    if repeats is not None and len(repeats) != len(dataset_dirs):
        raise SystemExit(
            f"--repeat needs one factor per dataset: "
            f"{len(repeats)} factors for {len(dataset_dirs)} datasets"
        )
    legible: list[dict] = []
    illegible: list[dict] = []
    for i, dataset_dir in enumerate(dataset_dirs):
        factor = repeats[i] if repeats else 1
        if factor < 1:
            raise SystemExit(f"--repeat factors must be >=1, got {factor}")
        with open(dataset_dir / "labels.jsonl") as f:
            for line in f:
                row = json.loads(line)
                row["path"] = str(dataset_dir / row["file"])
                row["_repeat"] = factor
                if row.get("legible") and row.get("label") is not None:
                    legible.append(row)
                else:
                    illegible.append(row)
    if not legible:
        raise SystemExit(f"no legible labeled rows in {dataset_dirs}")
    return legible, illegible


def expand_repeats(rows: list[dict]) -> list[dict]:
    """Train-side oversampling: each row appears _repeat times."""
    return [row for row in rows for _ in range(row.get("_repeat", 1))]


def train(dataset_dirs: list[Path], out_dir: Path, epochs: int = 20,
          batch_size: int = 256, lr: float = 1e-4, warmup_epochs: int = 2,
          val_frac: float = 0.1, abstain: bool = False, smoke: bool = False,
          repeats: list[int] | None = None) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True)
    model = model.to(device)
    model.log = lambda *a, **k: None  # training_step self.log()s; no Trainer attached
    img_h, img_w = tuple(model.hparams.img_size)

    legible, illegible = load_rows(dataset_dirs, repeats=repeats)
    rng = random.Random(0)
    rng.shuffle(legible)
    rng.shuffle(illegible)
    if smoke:
        legible, illegible, epochs, warmup_epochs = legible[:500], illegible[:100], 2, 1
    n_val = max(int(len(legible) * val_frac), 1)
    val_rows, train_rows = legible[:n_val], expand_repeats(legible[n_val:])
    # --abstain: illegible crops become training rows with an empty target
    # (tokenizer encodes "" as immediate EOS) so the OCR learns to read
    # nothing on garbage instead of confidently hallucinating a digit.
    if abstain:
        n_ival = max(int(len(illegible) * val_frac), 1)
        illeg_val = illegible[:n_ival]
        train_rows = train_rows + expand_repeats(illegible[n_ival:])
    else:
        illeg_val = illegible

    def load_batch(rows: list[dict]) -> tuple[torch.Tensor, list[str]]:
        imgs = []
        for row in rows:
            arr = np.asarray(Image.open(row["path"]).convert("RGB"))
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            t = torch.nn.functional.interpolate(
                t, size=(img_h, img_w), mode="bilinear", align_corners=False
            ).squeeze(0)
            imgs.append((t - 0.5) / 0.5)
        labels = ["" if r["label"] is None else str(r["label"]) for r in rows]
        return torch.stack(imgs), labels

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        hits, confs = 0, []
        for start in range(0, len(val_rows), batch_size):
            chunk = val_rows[start : start + batch_size]
            imgs, labels = load_batch(chunk)
            probs = model(imgs.to(device)).softmax(-1)
            preds, char_confs = model.tokenizer.decode(probs)
            for pred, cc, label in zip(preds, char_confs, labels, strict=True):
                hits += int(pred == label)
                confs.append(float(cc.min()) if len(cc) else 0.0)
        illeg_confs, illeg_empty = [], 0
        for start in range(0, len(illeg_val), batch_size):
            chunk = illeg_val[start : start + batch_size]
            imgs, _ = load_batch(chunk)
            probs = model(imgs.to(device)).softmax(-1)
            preds, char_confs = model.tokenizer.decode(probs)
            illeg_empty += sum(1 for p in preds if p == "")
            illeg_confs.extend(float(cc.min()) if len(cc) else 0.0 for cc in char_confs)
        model.train()
        return {
            "exact": round(hits / len(val_rows), 4),
            "legible_min_conf_mean": round(sum(confs) / len(confs), 4),
            "illegible_min_conf_mean": (
                round(sum(illeg_confs) / len(illeg_confs), 4) if illeg_confs else None
            ),
            "illegible_empty_rate": (
                round(illeg_empty / len(illeg_val), 4) if illeg_val else None
            ),
        }

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs - warmup_epochs, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [warmup, cosine], milestones=[warmup_epochs])

    out_dir.mkdir(parents=True, exist_ok=True)
    zero_shot = evaluate()
    print(json.dumps({"epoch": 0, "zero_shot": zero_shot}), flush=True)

    best, history = -1.0, []
    order = list(range(len(train_rows)))
    model.train()
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            rows = [train_rows[i] for i in order[start : start + batch_size]]
            imgs, labels = load_batch(rows)
            loss = model.training_step((imgs.to(device), labels), 0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            opt.step()
            losses.append(float(loss))
        sched.step()
        metrics = evaluate()
        history.append({"epoch": epoch, "loss": round(sum(losses) / len(losses), 4),
                        **metrics, "lr": round(opt.param_groups[0]["lr"], 8)})
        print(json.dumps(history[-1]), flush=True)
        # abstain runs are selected on read-legible AND refuse-illegible together
        score = metrics["exact"]
        if abstain:
            score += metrics["illegible_empty_rate"] or 0.0
        if score > best:
            best = score
            torch.save(model.state_dict(), out_dir / "parseq_jersey_best.pt")
    torch.save(model.state_dict(), out_dir / "parseq_jersey_last.pt")

    summary = {
        "recipe": {"lr": lr, "warmup_epochs": warmup_epochs, "batch_size": batch_size,
                   "epochs": epochs, "img_size": [img_h, img_w], "abstain": abstain,
                   "repeats": repeats},
        "train_rows": len(train_rows), "val_rows": len(val_rows),
        "illegible_val_rows": len(illeg_val),
        "zero_shot": zero_shot, "best_score": best, "history": history,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"best_score": best, "zero_shot_exact": zero_shot["exact"]}),
          flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, nargs="+",
                        help="synth_jersey.py output dir(s) (jpgs + labels.jsonl)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--abstain", action="store_true",
                        help="train illegible crops as empty-string abstentions")
    parser.add_argument("--repeat", type=int, nargs="+", default=None,
                        help="per-dataset train-side oversample factor "
                             "(one int per --dataset, applied after the "
                             "val split)")
    parser.add_argument("--smoke", action="store_true",
                        help="2 epochs on 500 rows to validate the rig")
    args = parser.parse_args()
    train(args.dataset, args.out, args.epochs, args.batch_size, args.lr,
          args.warmup_epochs, args.val_frac, args.abstain, args.smoke,
          repeats=args.repeat)


if __name__ == "__main__":
    main()
