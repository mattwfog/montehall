"""Spark launcher for the RF-DETR jersey-digit detector: stage, train, retrieve, score.

The Pusan v1 export is staged separately (1.8 GB tar over ssh into
~/soccerviz-candidates/jersey-digits with test → valid linked); this script ships only
code and runs the detached training container in the detectors image. `score` reads
the 200 frozen SoccerNet crops with a trained checkpoint in the candidates workspace
(which already holds the crops), pulls the predictions back and evaluates them
locally against the same ground truth as the Uncertainty-JNR baseline.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

IMAGE = "soccerviz-detector-training:20260907"
REMOTE = "~/soccerviz-candidates/jersey-digits"
BENCH_WORK = "~/soccerviz-candidates/workspace"
EXPORT = "jersey-number-detection-8a55j-v1-coco"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "spark"]
SHIPPED = ["src/soccerviz", "scripts/training/train_jersey_digits.py"]
BENCH_SHIPPED = ["src/soccerviz", "scripts/benchmarks/run_jersey_benchmark.py"]
CROP_MANIFEST = "artifacts/integrations/jersey-model-crops/crop-manifest.json"
TRUTH = "artifacts/vision-benchmark/ground-truth.json"
CHECKPOINT = "checkpoint_best_ema.pth"  # the val/ema_* columns of metrics.csv are its numbers


def skip_pycache(member):
    return None if "__pycache__" in member.name else member


def remote(command, **kwargs):
    return subprocess.run(SSH + [command], check=True, **kwargs)


def ship(root, names, target):
    with tempfile.TemporaryFile() as archive:
        with tarfile.open(fileobj=archive, mode="w") as tar:
            for name in names:
                tar.add(root / name, arcname=name, filter=skip_pycache)
        archive.seek(0)
        remote(f"mkdir -p {target}/results && tar -xf - -C {target}", stdin=archive)


def stage(root):
    ship(root, SHIPPED, REMOTE)
    ship(root, BENCH_SHIPPED, BENCH_WORK)
    print(f"staged code under {REMOTE}; scoring code under {BENCH_WORK}")


def train(name, epochs, smoke):
    command = (
        f"python scripts/training/train_jersey_digits.py --export {EXPORT}"
        f" --checkpoint /models/rf-detr-medium.pth --out results/{name} --epochs {epochs}"
        + (" --smoke" if smoke else "")
    )
    docker = (
        f"docker run -d --rm --name train-digits-{name} --gpus all --cpus 8 --shm-size 8g"
        f' -v "$HOME/soccerviz-candidates/jersey-digits:/work" -w /work -e PYTHONPATH=/work/src'
        f' -v "$HOME/soccerviz-candidates/shared/weights:/models:ro" {IMAGE}'
        f" sh -c {shlex.quote(command + f' > results/{name}.log 2>&1; chmod -R a+rX results/{name} results/{name}.log')}"
    )
    remote(f"cd {REMOTE} && {docker}")
    print(f"started train-digits-{name}; log {REMOTE}/results/{name}.log")


def retrieve(root, name):
    """Bundle without the per-epoch checkpoints (checkpoint_best_total.pth and evidence only)."""
    out = root / "artifacts/models" / name
    if out.exists():
        raise FileExistsError(f"{out} exists; retrieved bundles are never overwritten")
    with tempfile.TemporaryFile() as archive:
        remote(
            f"cd {REMOTE}/results && tar -cf - --exclude='checkpoint[0-9]*.pth'"
            " --exclude='checkpoint.pth' --exclude='checkpoint_*.ckpt'"
            f" {shlex.quote(name)} {shlex.quote(name + '.log')}",
            stdout=archive,
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar:
            tar.extractall(out.parent, filter="data")
    (out.parent / f"{name}.log").rename(out / "spark-run.log")
    print(f"retrieved {out}")


def scored_dir(name):
    return f"artifacts/integrations/jersey-digits-{name}"


def infer(name):
    """Read the 200 frozen crops on Spark with results/<name>/checkpoint_best_ema.pth."""
    evidence = remote(
        f"cat {REMOTE}/results/{shlex.quote(name)}/training-evidence.json",
        capture_output=True,
        text=True,
    ).stdout
    resolution = json.loads(evidence)["training"]["resolution"]
    command = (
        "python scripts/benchmarks/run_jersey_benchmark.py infer-digits"
        f" --crop-manifest {CROP_MANIFEST} --checkpoint /trained/{CHECKPOINT}"
        f" --out {scored_dir(name)} --device cuda:0 --resolution {resolution}"
    )
    docker = (
        f"docker run --rm --name infer-digits-{name} --gpus all --cpus 8"
        f' -v "$HOME/soccerviz-candidates/workspace:/work" -w /work -e PYTHONPATH=/work/src'
        f' -v "$HOME/soccerviz-candidates/jersey-digits/results/{name}:/trained:ro" {IMAGE}'
        f" sh -c {shlex.quote(command)}"
    )
    remote(f"cd {BENCH_WORK} && {docker} | tail -25")
    print(f"predictions written to {BENCH_WORK}/{scored_dir(name)}/predictions.json")


def retrieve_predictions(root, name):
    target = root / scored_dir(name) / "predictions.json"
    if target.exists():
        raise FileExistsError(f"{target} exists; predictions are never overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    remote(f"cat {BENCH_WORK}/{scored_dir(name)}/predictions.json", stdout=target.open("wb"))
    print(f"retrieved {target}")


def evaluate(root, name):
    """Local evaluation against the frozen ground truth, same command as the JNR baseline."""
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/run_jersey_benchmark.py",
            "evaluate",
            "--crop-manifest",
            CROP_MANIFEST,
            "--predictions",
            f"{scored_dir(name)}/predictions.json",
            "--truth",
            TRUTH,
            "--out",
            f"{scored_dir(name)}/evaluation.json",
        ],
        check=True,
        cwd=root,
    )


def score(root, name):
    infer(name)
    retrieve_predictions(root, name)
    evaluate(root, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=(
            "stage",
            "train",
            "retrieve",
            "infer",
            "retrieve-predictions",
            "evaluate",
            "score",
        ),
    )
    parser.add_argument("--name", default="rf-digits-pusan-v1")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.operation == "stage":
        stage(root)
    elif args.operation == "train":
        train(args.name, args.epochs, args.smoke)
    elif args.operation == "retrieve":
        retrieve(root, args.name)
    elif args.operation == "infer":
        infer(args.name)
    elif args.operation == "retrieve-predictions":
        retrieve_predictions(root, args.name)
    elif args.operation == "evaluate":
        evaluate(root, args.name)
    else:
        score(root, args.name)


if __name__ == "__main__":
    main()
