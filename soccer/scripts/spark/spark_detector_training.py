"""Ship the RF-DETR trainer to Spark, run a training job detached, retrieve its bundle.

Mirrors the 2026-09-07 Roboflow v10 pilot launch (`~/soccerviz-candidates/training-roboflow`,
image soccerviz-detector-training:20260907, frozen corpus already staged there) so a longer run
uses exactly the same corpus, initialization checkpoint and mounts.
"""

import argparse
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path

# Manifests record absolute image paths from the machine that built them;
# the remote worker sees the same tree mounted at /work.
LOCAL_ROOT = Path(__file__).resolve().parents[2]

REMOTE = "~/soccerviz-candidates"
IMAGE = "soccerviz-detector-training:20260907"
BENCH_IMAGE = "soccerviz-detectors:20260907"
BENCH_WORK = f"{REMOTE}/workspace"  # frozen 450-frame manifest, sources and results live here
BENCH_SHIPPED = ["src/soccerviz", "scripts/benchmarks/run_detector_benchmark.py"]
WORK = f"{REMOTE}/training-roboflow"
CORPUS = "/work/artifacts/detector-training-roboflow-v10"
INIT_CHECKPOINT = "/models/rf-detr-medium.pth"
INIT_SHA256 = "749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "spark"]
SHIPPED = ["src/soccerviz", "scripts/training/train_detector_candidate.py"]


def skip_pycache(member):
    """Container runs leave root-owned __pycache__ on Spark; never ship or overwrite bytecode."""
    return None if "__pycache__" in member.name else member


def remote(command, **kwargs):
    return subprocess.run(SSH + [command], check=True, **kwargs)


def ship(root, names=SHIPPED, target=WORK):
    with tempfile.TemporaryFile() as archive:
        with tarfile.open(fileobj=archive, mode="w") as tar:
            for name in names:
                tar.add(root / name, arcname=name, filter=skip_pycache)
        archive.seek(0)
        remote(f"tar -xf - -C {target}", stdin=archive)


def train(name, epochs, smoke):
    """One detached container; outputs under ~/soccerviz-candidates/results/training/<name>."""
    command = (
        f"python scripts/training/train_detector_candidate.py --corpus {CORPUS}"
        f" --checkpoint {INIT_CHECKPOINT} --checkpoint-sha256 {INIT_SHA256}"
        f" --out /outputs/{name} --epochs {epochs}" + (" --smoke-step" if smoke else "")
    )
    docker = (
        f"docker run -d --rm --name train-{name} --gpus all --cpus 8 --shm-size 8g"
        f' -v "$HOME/soccerviz-candidates/training-roboflow:/work"'
        f' -v "$HOME/soccerviz-candidates/shared/weights:/models:ro"'
        f' -v "$HOME/soccerviz-candidates/results/training:/outputs" {IMAGE}'
        f" sh -c {shlex.quote(command + f' > /outputs/{name}.log 2>&1; ' + readable(name))}"
    )
    remote(f"cd {REMOTE} && {docker}")
    print(f"started train-{name}; log {REMOTE}/results/training/{name}.log")


def readable(name):
    """The container runs as root and RF-DETR writes checkpoint_best_total.pth mode 0600;
    without this the retrieve tar fails with Permission denied (2026-09-08)."""
    return f"chmod -R a+rX /outputs/{name} /outputs/{name}.log"


def retrieve(root, name):
    """Model bundle without the per-epoch .ckpt files (500 MB each); those stay on Spark."""
    out = root / "artifacts/models" / name
    if out.exists():
        raise FileExistsError(f"{out} exists; retrieved bundles are never overwritten")
    with tempfile.TemporaryFile() as archive:
        remote(
            f"cd {REMOTE}/results/training && tar -cf - --exclude='checkpoint_[0-9]*.ckpt'"
            f" {shlex.quote(name)} {shlex.quote(name + '.log')}",
            stdout=archive,
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar:
            tar.extractall(out.parent, filter="data")
    (out.parent / f"{name}.log").rename(out / "spark-run.log")
    print(f"retrieved {out}")


def benchmark(name, prefix):
    """Frozen 450-frame inference with the trained best checkpoint, same settings as the v10 pilot:
    576 px, capture floor 0.01 (evaluation applies 0.25 later), soccer class mode."""
    command = (
        "python scripts/benchmarks/run_detector_benchmark.py"
        " --manifest artifacts/vision-benchmark/manifest.json"
        f" --out results/{prefix}-predictions.json --backend rf-detr-medium --class-mode soccer"
        " --checkpoint /trained/checkpoint_best_total.pth --resolution 576 --threshold 0.01"
        f" --path-prefix {LOCAL_ROOT}=/work"
    )
    docker = (
        f"docker run -d --rm --name bench-{name} --gpus all --cpus 8 --shm-size 8g"
        f' -v "$HOME/soccerviz-candidates/workspace:/work"'
        f' -v "$HOME/soccerviz-candidates/results/training/{name}:/trained:ro" {BENCH_IMAGE}'
        f" sh -c {shlex.quote(command + f' > results/{prefix}-benchmark.log 2>&1')}"
    )
    remote(f"cd {REMOTE} && {docker}")
    print(f"started bench-{name}; log {BENCH_WORK}/results/{prefix}-benchmark.log")


def retrieve_benchmark(root, prefix):
    target = root / "artifacts/vision-benchmark" / f"{prefix}-predictions.json"
    if target.exists():
        raise FileExistsError(f"{target} exists; benchmark predictions are never overwritten")
    with tempfile.TemporaryFile() as archive:
        remote(
            f"cd {BENCH_WORK}/results && tar -cf - {shlex.quote(prefix + '-predictions.json')}"
            f" {shlex.quote(prefix + '-benchmark.log')}",
            stdout=archive,
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar:
            tar.extractall(target.parent, filter="data")
    (target.parent / f"{prefix}-benchmark.log").rename(root / "logs" / f"{prefix}-benchmark.log")
    print(f"retrieved {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--smoke-step", action="store_true")
    p = sub.add_parser("retrieve")
    p.add_argument("--name", required=True)
    p = sub.add_parser("benchmark")
    p.add_argument("--name", required=True, help="Training run name; its bundle stays on Spark")
    p.add_argument(
        "--prefix", required=True, help="Benchmark record prefix, e.g. rf-medium-roboflow-v10-e50"
    )
    p = sub.add_parser("retrieve-benchmark")
    p.add_argument("--prefix", required=True)
    args = parser.parse_args()
    for value in (getattr(args, "name", "x"), getattr(args, "prefix", "x")):
        if not value.replace("-", "").isalnum():
            parser.error("Use simple alphanumeric names and prefixes")
    root = Path(__file__).resolve().parents[2]
    if args.command == "train":
        ship(root)
        train(args.name, args.epochs, args.smoke_step)
    elif args.command == "retrieve":
        retrieve(root, args.name)
    elif args.command == "benchmark":
        ship(root, BENCH_SHIPPED, BENCH_WORK)
        benchmark(args.name, args.prefix)
    else:
        retrieve_benchmark(root, args.prefix)


if __name__ == "__main__":
    main()
