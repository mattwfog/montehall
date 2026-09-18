"""Run a fixed RF-DETR Medium soccer pilot or a one-optimizer-step smoke check.

Only the frozen official-TRAIN corpus is consumed. External benchmark labels,
images and metrics are not inputs to training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

from soccerviz.candidates.detector_training import (
    CLASSES,
    RF_TRAINING,
    SEED,
    verify_corpus,
    write_new_json,
)
from soccerviz.core.assets import sha256


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("artifacts/detector-training"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="Expected public COCO initialization hash, frozen before this run",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--epochs",
        type=int,
        default=RF_TRAINING["epochs"],
        help="Epoch count; the frozen protocol's five-epoch pilot is the default, more is a longer run "
        "on the same corpus, recorded in run-config.json",
    )
    parser.add_argument(
        "--smoke-step",
        action="store_true",
        help="Exactly one optimizer update after four microbatches; separate output",
    )
    args = parser.parse_args(argv)
    if args.device not in {"cpu", "cuda", "cuda:0"}:
        parser.error(
            "This single-device pilot supports cpu or cuda:0 only; other GPU indices are not remapped"
        )
    if args.epochs < 1:
        parser.error("epochs must be at least 1")
    if len(args.checkpoint_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in args.checkpoint_sha256
    ):
        parser.error("checkpoint-sha256 must be a lowercase SHA256 digest")
    return args


def train(args):
    started = time.perf_counter()
    args.execution_phase = "preflight"
    args.created_output = False
    if args.device not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("This pilot only supports cpu or the sole GPU cuda:0")
    if args.out.exists():
        raise FileExistsError(
            "Training output already exists; completed and partial runs are preserved"
        )
    corpus = args.corpus.resolve()
    manifest, protocol = verify_corpus(corpus)
    checkpoint = args.checkpoint.resolve()
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("Initialization checkpoint hash mismatch")
    if importlib.metadata.version("rfdetr") != "1.10.1":
        raise RuntimeError("This runner requires the frozen rfdetr1.10.1 training API")
    import torch
    from pytorch_lightning import Callback, seed_everything
    from rfdetr.config import RFDETRMediumConfig, TrainConfig
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

    seed_everything(SEED, workers=True)
    args.out.mkdir(parents=True)
    args.created_output = True
    args.execution_phase = "setup"
    initialization_hash = args.checkpoint_sha256
    config_values = {k: v for k, v in RF_TRAINING.items() if k != "resolution"}
    config_values.update(
        {
            "epochs": args.epochs,
            "dataset_dir": str(corpus / "coco"),
            "dataset_file": "roboflow",
            "output_dir": str(args.out.resolve()),
            "class_names": list(CLASSES),
            "eval_batch_size": 2,
            "eval_max_dets": 100,
            "log_per_class_metrics": True,
            "notes": {
                "corpus_manifest_sha256": sha256(corpus / "manifest.json"),
                "protocol_sha256": manifest["protocol_sha256"],
                "initialization_sha256": initialization_hash,
                "external_development_used_for_selection": False,
                "smoke_step": args.smoke_step,
                "epochs_override": args.epochs != RF_TRAINING["epochs"],
            },
        }
    )
    train_config = TrainConfig(**config_values)
    model_config = RFDETRMediumConfig(
        pretrain_weights=str(checkpoint),
        device=args.device,
        resolution=RF_TRAINING["resolution"],
        num_classes=len(CLASSES),
        model_name="RFDETRMedium",
    )
    configuration = {
        "schema": "detector-training-run/v1",
        "backend": "rf-detr-medium",
        "initialization_checkpoint": str(checkpoint),
        "initialization_sha256": initialization_hash,
        "corpus_manifest_sha256": sha256(corpus / "manifest.json"),
        "protocol_sha256": sha256(corpus / "protocol.json"),
        "training_config": train_config.model_dump(mode="json"),
        "model_config": model_config.model_dump(mode="json"),
        "run_kind": "one_optimizer_step_smoke"
        if args.smoke_step
        else ("five_epoch_pilot" if args.epochs == 5 else f"{args.epochs}_epoch_run"),
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "rfdetr",
                "torch",
                "torchvision",
                "pytorch-lightning",
                "transformers",
                "torchmetrics",
            )
        },
        "platform": platform.platform(),
        "source_protocol": protocol,
    }
    write_new_json(args.out / "run-config.json", configuration)

    class TrainingEvidence(Callback):
        def __init__(self):
            self.steps = []
            self.completed_epochs = []
            self.last_step = 0
            self.initial_head = None
            self.final_head = None
            self.gradients_finite = True
            self.last_loss = None

        def head_bytes(self, module):
            candidates = [
                (name, p)
                for name, p in module.named_parameters()
                if "class_embed" in name and name.endswith("weight") and p.requires_grad
            ]
            if not candidates:
                raise RuntimeError(
                    "Could not identify trainable classification head for update audit"
                )
            return candidates[0][1].detach().float().cpu().contiguous().numpy().tobytes()

        def on_train_start(self, trainer, pl_module):
            self.initial_head = hashlib.sha256(self.head_bytes(pl_module)).hexdigest()

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            gradients = [p.grad for p in pl_module.parameters() if p.grad is not None]
            if (
                not gradients
                or not torch.stack([torch.isfinite(g).all() for g in gradients]).all().item()
            ):
                self.gradients_finite = False
                raise FloatingPointError("Training has absent or nonfinite gradients")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if loss is not None:
                self.last_loss = float(loss.detach().cpu())
                if not math.isfinite(self.last_loss):
                    raise FloatingPointError("Nonfinite training loss")
            if trainer.global_step > self.last_step:
                self.last_step = int(trainer.global_step)
                self.final_head = hashlib.sha256(self.head_bytes(pl_module)).hexdigest()
                row = {
                    "optimizer_step": self.last_step,
                    "epoch": int(trainer.current_epoch),
                    "batch_index": batch_idx,
                    "loss": self.last_loss,
                }
                self.steps.append(row)
                with (args.out / "optimizer-steps.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")

        def on_train_epoch_end(self, trainer, pl_module):
            self.completed_epochs.append(int(trainer.current_epoch))

    evidence = TrainingEvidence()
    module = RFDETRModelModule(model_config, train_config)
    data = RFDETRDataModule(model_config, train_config)
    data.setup("fit")
    if list(data.class_names) != list(CLASSES):
        raise ValueError(
            "Native RF-DETR dataset class mapping differs from the four soccer classes"
        )
    accelerator = "gpu" if args.device.startswith("cuda") else "cpu"
    # RF1.10.1's wrapper inspects int/str devices before PTL handles the argument.
    # On the single GB10, devices=1 selects index0 without its list-handling bug.
    trainer_kwargs = {"devices": 1, "log_every_n_steps": 1}
    if args.smoke_step:
        trainer_kwargs.update({"max_steps": 1, "limit_val_batches": 1})
    trainer = build_trainer(train_config, model_config, accelerator=accelerator, **trainer_kwargs)
    # Append to, rather than replace, native EMA/checkpoint/metric callbacks.
    trainer.callbacks.append(evidence)
    failure = None
    try:
        args.execution_phase = "fit"
        trainer.fit(module, datamodule=data)
        if not evidence.steps or evidence.initial_head == evidence.final_head:
            raise RuntimeError("No verified optimizer update to the trainable classification head")
        if args.smoke_step and trainer.global_step != 1:
            raise RuntimeError("Smoke run did not perform exactly one optimizer update")
        if not args.smoke_step and evidence.completed_epochs != list(range(args.epochs)):
            raise RuntimeError(f"Run did not complete exactly {args.epochs} epochs")
        from rfdetr import RFDETRMedium

        args.execution_phase = "checkpoint_validation"
        best = args.out / "checkpoint_best_total.pth"
        if best.exists():
            reloaded = RFDETRMedium.from_checkpoint(str(best), device="cpu")
            if reloaded.class_names != list(CLASSES) or reloaded.model_config.num_classes != 4:
                raise ValueError("Saved checkpoint lost the four-class soccer label mapping")
            write_new_json(
                args.out / "checkpoint-class-roundtrip.json",
                {
                    "checkpoint": best.name,
                    "sha256": sha256(best),
                    "class_names": reloaded.class_names,
                    "num_classes": reloaded.model_config.num_classes,
                    "native_dataset_class_names": list(data.class_names),
                },
            )
            del reloaded
        elif not args.smoke_step:
            raise RuntimeError("Fit did not produce the expected native best checkpoint")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        checkpoints = [
            {"path": str(p.relative_to(args.out)), "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(args.out.rglob("*"))
            if p.suffix in {".pth", ".ckpt"}
        ]
        result = {
            "schema": "detector-training-evidence/v1",
            "completed": failure is None,
            "run_kind": configuration["run_kind"],
            "failure": failure,
            "optimizer_steps": int(trainer.global_step),
            "epochs_with_training_end_hook": evidence.completed_epochs,
            "initial_classification_head_sha256": evidence.initial_head,
            "final_classification_head_sha256": evidence.final_head,
            "classification_head_changed": evidence.initial_head != evidence.final_head,
            "finite_gradients": evidence.gradients_finite,
            "last_training_loss": evidence.last_loss,
            "checkpoints": checkpoints,
            "elapsed_s": time.perf_counter() - started,
            "corpus_manifest_sha256": configuration["corpus_manifest_sha256"],
            "selection": "Native best checkpoint by internal-development box mAP; external450 unused",
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
            if accelerator == "gpu"
            else None,
            "limitations": protocol["limitations"],
        }
        write_new_json(args.out / "training-evidence.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        return train(args)
    except Exception as error:
        if (
            getattr(args, "created_output", False)
            and not (args.out / "training-evidence.json").exists()
        ):
            write_new_json(
                args.out / "failure.json",
                {
                    "schema": "detector-training-failure/v1",
                    "completed": False,
                    "phase": getattr(args, "execution_phase", "unknown"),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "optimizer_update_verified": False,
                    "note": "No completed training evidence was produced; do not count this as a trained model.",
                },
            )
        raise


if __name__ == "__main__":
    main()
