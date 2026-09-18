"""RF-DETR Medium fine-tune on a merged COCO dataset (rfdetr layout).

The committed successor to the ad hoc spark train_v0.py — same invocation
shape, parameterized. Resumable: pass --resume with the last.ckpt rfdetr
wrote to continue an interrupted run.

Batching is tuned for the spark GB10 (128GB unified memory): a real batch of
16 with no gradient accumulation keeps the GPU fed, instead of stalling on
4-image micro-batches accumulated 4x — same effective batch (16), far fewer
sync/dataloader stalls, higher utilization. On a smaller GPU, lower
--batch-size and raise --grad-accum-steps to keep the product at 16.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def train(
    dataset_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int = 16,
    grad_accum_steps: int = 1,
    lr: float = 1e-4,
    resume: Path | None = None,
    pretrain_weights: Path | None = None,
) -> dict:
    from rfdetr import RFDETRMedium

    started = time.monotonic()
    kwargs: dict = {}
    if resume is not None:
        kwargs["resume"] = str(resume)
    # Warm-start from a prior fine-tune (e.g. v2 best) instead of COCO
    # pretrain — domain adaptation converges in a fraction of the epochs.
    model_kwargs: dict = {}
    if pretrain_weights is not None:
        model_kwargs["pretrain_weights"] = str(pretrain_weights)
    model = RFDETRMedium(**model_kwargs)
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        **kwargs,
    )
    return {"done": True, "wall_min": round((time.monotonic() - started) / 60, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--pretrain-weights", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(train(
        args.dataset_dir, args.output_dir, args.epochs,
        batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps,
        lr=args.lr, resume=args.resume, pretrain_weights=args.pretrain_weights,
    )))


if __name__ == "__main__":
    main()
