"""Fixed YOLO26 Medium soccer training pilot on the shared TRAIN-only corpus.

Only the 75-frame internal development game selects best.pt. No external
benchmark images, labels, or metrics are opened by this worker.
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

from soccerviz.candidates.detector_training import CLASSES, SEED, verify_corpus, write_new_json
from soccerviz.core.assets import sha256

ULTRALYTICS_VERSION = "8.4.143"
INITIALIZATION_SHA256 = "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"
INITIALIZATION_SOURCE = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt"
TRAINING = {
    "epochs": 5,
    "batch": 2,
    "imgsz": 576,
    "seed": SEED,
    "workers": 2,
    "optimizer": "AdamW",
    "lr0": 1e-4,
    "lrf": 1.0,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "nbs": 8,
    "warmup_epochs": 0.0,
    "warmup_momentum": 0.9,
    "warmup_bias_lr": 0.0,
    "amp": False,
    "deterministic": True,
    "cos_lr": False,
    "patience": 0,
    "time": None,
    "resume": False,
    "val": True,
    "split": "val",
    "fraction": 1.0,
    "single_cls": False,
    "rect": False,
    "multi_scale": 0.0,
    "close_mosaic": 0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
    "auto_augment": None,
    "erasing": 0.0,
    "box": 7.5,
    "cls": 0.5,
    "cls_pw": 0.0,
    "dfl": 1.5,
    "cls_remap": False,
    "pretrained": True,
    "freeze": None,
    "save": True,
    "save_period": 1,
    "plots": False,
    "save_json": False,
    "cache": False,
    "profile": False,
    "compile": False,
    "exist_ok": False,
    "verbose": True,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("artifacts/detector-training"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=INITIALIZATION_SHA256)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cpu", "cuda", "cuda:0"))
    args = parser.parse_args(argv)
    if args.checkpoint_sha256 != INITIALIZATION_SHA256:
        parser.error("This frozen pilot requires the recorded public YOLO26M initialization SHA256")
    return args


def training_arguments(corpus, out, device):
    if device not in ("cpu", "cuda", "cuda:0"):
        raise ValueError("Only CPU or the sole GPU cuda:0 is supported")
    return {
        **TRAINING,
        "data": str((Path(corpus) / "yolo/data.yaml").resolve()),
        "project": str(Path(out).resolve()),
        "name": "native",
        "device": "cpu" if device == "cpu" else 0,
    }


def validate_native_data(data, corpus):
    """Validate the actual Ultralytics-resolved paths, not a guessed YAML base."""
    root = Path(corpus).resolve()
    names = data["names"]
    ordered = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    if ordered != list(CLASSES) or data.get("nc", len(ordered)) != 4:
        raise ValueError("Native YOLO class names differ from the frozen four soccer classes")
    for name, partition in (("train", "train"), ("val", "valid")):
        value = data[name]
        if not isinstance(value, str) or Path(value).resolve() != root / "yolo/images" / partition:
            raise ValueError(f"Native YOLO {name} path differs from the frozen internal partition")
    if data.get("test") or data.get("minival") or data.get("download"):
        raise ValueError("Test/minival/download inputs are forbidden in the fixed pilot")
    return {"train": data["train"], "val": data["val"], "class_names": ordered}


def expected_image_paths(manifest, corpus, partition):
    root = Path(corpus).resolve()
    return {
        str(root / "yolo/images" / partition / Path(frame["image_path"]).name)
        for frame in manifest["frames"]
        if frame["partition"] == partition
    }


def finite_loss_values(loss):
    if hasattr(loss, "detach"):
        values = loss.detach().float().cpu().reshape(-1).tolist()
    elif isinstance(loss, (list, tuple)):
        values = [float(value) for value in loss]
    else:
        values = [float(loss)]
    if not values or not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Nonfinite or absent native training loss")
    return values


def training_audit_summary(epochs, steps, initial_head, final_head):
    if epochs != list(range(5)):
        raise RuntimeError("Pilot did not complete exactly five training epochs")
    if steps < 1 or not initial_head or not final_head or initial_head == final_head:
        raise RuntimeError("No verified optimizer update to the four-class head")
    return {
        "epochs_completed": len(epochs),
        "optimizer_steps": steps,
        "classification_head_changed": True,
    }


def train(args):
    started = time.perf_counter()
    args.created_output = False
    args.execution_phase = "preflight"
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("Preserve existing completed or partial training output")
    corpus = args.corpus.resolve()
    manifest, protocol = verify_corpus(corpus)
    checkpoint = args.checkpoint.resolve()
    if (
        args.checkpoint_sha256 != INITIALIZATION_SHA256
        or sha256(checkpoint) != INITIALIZATION_SHA256
    ):
        raise ValueError("Public YOLO26M initialization checkpoint SHA256 mismatch")
    if importlib.metadata.version("ultralytics") != ULTRALYTICS_VERSION:
        raise RuntimeError("The frozen pilot requires ultralytics==8.4.143")
    out.mkdir(parents=True)
    args.created_output = True
    args.execution_phase = "setup"
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    data_yaml = corpus / "yolo/data.yaml"
    resolved = check_det_dataset(str(data_yaml), autodownload=False)
    native_data = validate_native_data(resolved, corpus)
    arguments = training_arguments(corpus, out, args.device)
    source_dir = Path(ultralytics.__file__).parent
    config = {
        "schema": "yolo-candidate-training-run/v1",
        "backend": "YOLO26-Medium",
        "run_kind": "five_epoch_pilot",
        "initialization_checkpoint": str(checkpoint),
        "initialization_source": INITIALIZATION_SOURCE,
        "initialization_sha256": INITIALIZATION_SHA256,
        "corpus_manifest_sha256": sha256(corpus / "manifest.json"),
        "corpus_protocol_sha256": sha256(corpus / "protocol.json"),
        "native_data_yaml_sha256": sha256(data_yaml),
        "native_resolved_data": native_data,
        "training_arguments": arguments,
        "effective_batch_size": 8,
        "gradient_accumulation": "Native nbs8/batch2 gives4; no warmup adjustment",
        "class_names": list(CLASSES),
        "source_game_partitions": protocol["sequences"],
        "excluded_external_game_ids": protocol["excluded_external_game_ids"],
        "external_development_used_for_selection": False,
        "selection": "Native best.pt by validation fitness on the75-frame internal-development partition only",
        "comparability": "Shared data, five epochs,576px,batch2/effective8 and AdamW1e-4; native augmentation, scheduler, head/loss/EMA, parameter grouping, and RF encoder LR differ. Not architecture-controlled training.",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python")
        },
        "upstream_source_sha256": {
            name: sha256(source_dir / name)
            for name in ("engine/trainer.py", "data/utils.py", "cfg/default.yaml")
        },
        "runner_sha256": sha256(__file__),
        "platform": platform.platform(),
    }
    write_new_json(out / "run-config.json", config)
    model = YOLO(str(checkpoint), task="detect")
    evidence = {
        "steps": 0,
        "epochs": [],
        "batches": 0,
        "initial_head": None,
        "final_head": None,
        "last_loss": None,
        "head_parameters": [],
        "hooks": [],
        "optimizer_object": None,
        "epoch_metrics": [],
    }

    def head_fingerprint(trainer):
        head = trainer.model.model[-1]
        candidates = [
            (name, parameter)
            for name, parameter in head.named_parameters()
            if "cv3" in name
            and name.endswith("weight")
            and parameter.requires_grad
            and parameter.ndim == 4
            and parameter.shape[0] == len(CLASSES)
        ]
        if not candidates:
            raise RuntimeError("Cannot audit native four-class classification output weights")
        evidence["head_parameters"] = [name for name, _ in candidates]
        digest = hashlib.sha256()
        for name, parameter in sorted(candidates):
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def runtime_contract(trainer):
        if trainer.batch_size != 2 or trainer.args.batch != 2 or trainer.accumulate != 4:
            raise RuntimeError("Native runtime changed the fixed batch/accumulation budget")
        if trainer.optimizer is not evidence["optimizer_object"]:
            raise RuntimeError("Native runtime replaced the audited optimizer")
        validate_native_data(trainer.data, corpus)

    def attach_optimizer_audit(trainer):
        validate_native_data(trainer.data, corpus)
        if Path(trainer.save_dir).resolve() != out / "native":
            raise RuntimeError("Native trainer changed the requested isolated output path")
        for loader, partition in ((trainer.train_loader, "train"), (trainer.test_loader, "valid")):
            actual = {str(Path(path).resolve()) for path in loader.dataset.im_files}
            expected = expected_image_paths(manifest, corpus, partition)
            if actual != expected or len(loader.dataset) != len(expected):
                raise ValueError(
                    "Native data loader dropped, added, or changed frozen training/dev images"
                )
        if not isinstance(trainer.optimizer, torch.optim.AdamW):
            raise TypeError("Native optimizer differs from explicit AdamW")
        evidence["optimizer_object"] = trainer.optimizer
        runtime_contract(trainer)
        evidence["initial_head"] = head_fingerprint(trainer)

        def before_step(optimizer, unused_args, unused_kwargs):
            runtime_contract(trainer)
            gradients = [
                p.grad
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            if (
                not gradients
                or not torch.stack([torch.isfinite(g).all() for g in gradients]).all().item()
            ):
                raise FloatingPointError("Absent or nonfinite native optimizer gradients")
            finite_loss_values(trainer.loss)

        def after_step(optimizer, unused_args, unused_kwargs):
            evidence["steps"] += 1
            evidence["final_head"] = head_fingerprint(trainer)
            row = {
                "optimizer_step": evidence["steps"],
                "epoch": int(trainer.epoch),
                "loss": finite_loss_values(trainer.loss),
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                "classification_head_sha256": evidence["final_head"],
            }
            with (out / "optimizer-steps.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")

        # Ultralytics' optimizer_step callback is reserved, not emitted by default.
        # PyTorch hooks count actual optimizer.step calls, after accumulation.
        evidence["hooks"] = [
            trainer.optimizer.register_step_pre_hook(before_step),
            trainer.optimizer.register_step_post_hook(after_step),
        ]

    def batch_start(trainer):
        runtime_contract(trainer)

    def batch_end(trainer):
        evidence["batches"] += 1
        evidence["last_loss"] = finite_loss_values(trainer.loss)

    def epoch_end(trainer):
        evidence["epochs"].append(int(trainer.epoch))

    def fit_epoch_end(trainer):
        metrics = {key: float(value) for key, value in trainer.metrics.items()}
        if any(not math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite internal-development metric")
        is_final_revalidation = len(evidence["epoch_metrics"]) >= TRAINING["epochs"]
        evidence["epoch_metrics"].append(
            {
                "epoch": int(trainer.epoch),
                "metrics": metrics,
                "phase": "final_best_checkpoint_validation"
                if is_final_revalidation
                else "epoch_validation",
                "fitness": None if is_final_revalidation else float(trainer.fitness),
                "best_fitness": float(trainer.best_fitness),
            }
        )

    for event, callback in (
        ("on_pretrain_routine_end", attach_optimizer_audit),
        ("on_train_batch_start", batch_start),
        ("on_train_batch_end", batch_end),
        ("on_train_epoch_end", epoch_end),
        ("on_fit_epoch_end", fit_epoch_end),
    ):
        model.add_callback(event, callback)
    args.execution_phase = "fit"
    model.train(**arguments)
    summary = training_audit_summary(
        evidence["epochs"], evidence["steps"], evidence["initial_head"], evidence["final_head"]
    )
    args.execution_phase = "checkpoint_validation"
    best = out / "native/weights/best.pt"
    last = out / "native/weights/last.pt"
    if not best.is_file() or not last.is_file():
        raise RuntimeError("Native pilot failed to emit both best.pt and last.pt")
    restored = YOLO(str(best), task="detect")
    names = restored.names
    if [names[i] for i in range(len(names))] != list(CLASSES):
        raise ValueError("Best checkpoint lost the four-class soccer mapping")
    validate_native_data(model.trainer.data, corpus)
    # Recheck immutable exported bytes after the complete training/validation flow.
    verify_corpus(corpus)
    result = {
        "schema": "yolo-candidate-training-evidence/v1",
        "completed": True,
        "backend": "YOLO26-Medium",
        "run_kind": "five_epoch_pilot",
        **summary,
        "epochs_with_training_end_hook": evidence["epochs"],
        "microbatches_completed": evidence["batches"],
        "initial_classification_head_sha256": evidence["initial_head"],
        "final_classification_head_sha256": evidence["final_head"],
        "audited_head_parameter_names": evidence["head_parameters"],
        "optimizer_audit": "PyTorch actual optimizer pre/post hooks; finite gradients and losses at every update",
        "finite_gradients": True,
        "finite_training_losses": True,
        "last_training_loss": evidence["last_loss"],
        "validation_events": evidence["epoch_metrics"],
        "class_names": list(CLASSES),
        "best_checkpoint": str(best.relative_to(out)),
        "checkpoints": [
            {
                "path": str(path.relative_to(out)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted((out / "native/weights").glob("*.pt"))
        ],
        "run_config_sha256": sha256(out / "run-config.json"),
        "corpus_manifest_sha256": config["corpus_manifest_sha256"],
        "native_data_yaml_sha256": config["native_data_yaml_sha256"],
        "best_class_mapping_verified_after_reload": True,
        "external_development_used_for_selection": False,
        "selection": config["selection"],
        "comparability": config["comparability"],
        "elapsed_s": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device != "cpu"
        else None,
        "limitations": protocol["limitations"],
    }
    write_new_json(out / "training-evidence.json", result)
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
                    "schema": "yolo-candidate-training-failure/v1",
                    "completed": False,
                    "phase": getattr(args, "execution_phase", "unknown"),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "note": "No completed five-epoch training evidence; partial checkpoints and optimizer logs may remain.",
                },
            )
        raise


if __name__ == "__main__":
    main()
