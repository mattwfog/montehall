"""Spark launcher for the jersey-number classifier: stage, train, retrieve, infer, evaluate.

Runs inside the workflow image (torch, timm, cv2 pinned there). Training writes
under ~/soccerviz-jersey-classifier/results/<name>. `chain` waits for a launched
run, retrieves it, runs the classifier on the oracle-track crops and on the 200
frozen crops on Spark, pulls both prediction sets back and evaluates locally.
`test` runs the module's pytest files inside the image, where torch is.
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

IMAGE = "soccerviz-workflow:20260908-e50"
REMOTE = "~/soccerviz-jersey-classifier"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "spark"]
TAISEIS = "artifacts/public-data/roboflow/jersey-number-ijbaq-v1-folder"
PUSAN = "artifacts/public-data/roboflow/jersey-number-detection-8a55j-v1-coco"
PUSAN_ON_SPARK = "$HOME/soccerviz-candidates/jersey-digits/jersey-number-detection-8a55j-v1-coco"
ORACLE = "artifacts/integrations/track-identity-oracle"
FROZEN = "artifacts/integrations/jersey-model-crops"
POOL = "artifacts/integrations/jersey-torso-pool"
FONTS = "artifacts/public-data/fonts"
SHIPPED = [
    "src/soccerviz",
    "scripts/training/train_jersey_classifier.py",
    "scripts/benchmarks/run_track_identity_benchmark.py",
    "scripts/benchmarks/run_jersey_benchmark.py",
    "tests/candidates/test_jersey_classifier.py",
    "tests/vision/test_track_identity.py",
]
TESTS = " ".join(name for name in SHIPPED if name.startswith("tests/"))


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
        remote(f"mkdir -p {target} && tar -xf - -C {target}", stdin=archive)


def stage(root, data=True):
    remote(f"mkdir -p {REMOTE}/results {REMOTE}/hf-cache")
    ship(root, SHIPPED + ([TAISEIS, ORACLE, FROZEN, POOL, FONTS] if data else []), REMOTE)
    print(f"staged code{' and data' if data else ''} under {REMOTE}")


def docker(name, command, extra=""):
    return (
        f"docker run --rm --name {name} --gpus all --cpus 8 --shm-size 8g"
        f' -v "$HOME/soccerviz-jersey-classifier:/work" -w /work -e PYTHONPATH=/work/src'
        f' -v "{PUSAN_ON_SPARK}:/work/{PUSAN}:ro"'  # the 1.8 GB export the digit lane already staged
        f" -e HF_HOME=/work/hf-cache {extra} {IMAGE} sh -c {shlex.quote(command)}"
    )


def train(name, epochs, smoke):
    command = (
        f"python scripts/training/train_jersey_classifier.py --taiseis {TAISEIS} --pusan {PUSAN}"
        f" --torso-pool {POOL} --fonts {FONTS}"
        f" --out results/{name} --epochs {epochs} --workers 8 --device cuda:0"
        + (" --smoke-step" if smoke else "")
        + f" > results/{name}.log 2>&1; chmod -R a+rX results/{name} results/{name}.log"
    )
    remote(f"cd {REMOTE} && {docker(f'train-jersey-{name}', command, '-d')}")
    print(f"started train-jersey-{name}; log {REMOTE}/results/{name}.log")


def test():
    remote(
        f"cd {REMOTE} && "
        + docker(
            "test-jersey-classifier",
            'python -c "import pytest" 2>/dev/null || pip install -q pytest;'
            f" python -m pytest -q {TESTS}",
        )
    )


def wait(name):
    remote(
        f'until [ -z "$(docker ps -q --filter name=train-jersey-{shlex.quote(name)})" ];'
        f" do sleep 30; done; tail -c 600 {REMOTE}/results/{shlex.quote(name)}.log"
    )


def retrieve(root, name):
    out = root / "artifacts/models" / name
    if out.exists():
        raise FileExistsError(f"{out} exists; retrieved bundles are never overwritten")
    with tempfile.TemporaryFile() as archive:
        remote(
            f"cd {REMOTE}/results && tar -cf - {shlex.quote(name)} {shlex.quote(name + '.log')}",
            stdout=archive,
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar:
            tar.extractall(out.parent, filter="data")
    (out.parent / f"{name}.log").rename(out / "spark-run.log")
    print(f"retrieved {out}")


def infer(name):
    """Classifier on the oracle-track crops and the 200 frozen crops, on Spark."""
    oracle = (
        "python scripts/benchmarks/run_track_identity_benchmark.py infer"
        f" --crop-manifest {ORACLE}/crop-manifest.json --checkpoint results/{name}/best.pt"
        f" --out results/{name}-oracle --device cuda:0"
    )
    frozen = (
        'python -c "from soccerviz.candidates.jersey_classifier import infer;'
        f" infer('{FROZEN}/crop-manifest.json', 'results/{name}/best.pt', 'results/{name}-frozen')\""
    )
    remote(f"cd {REMOTE} && {docker(f'infer-jersey-{name}', oracle + ' && ' + frozen)} | tail -5")
    print(f"inferred into {REMOTE}/results/{name}-oracle and {name}-frozen")


def retrieve_predictions(root, name):
    for suffix in ("oracle", "frozen"):
        target = root / "artifacts/integrations" / f"jersey-classifier-{name}-{suffix}"
        if target.exists():
            raise FileExistsError(f"{target} exists; prediction bundles are never overwritten")
        with tempfile.TemporaryFile() as archive:
            remote(
                f"cd {REMOTE}/results && tar -cf - {shlex.quote(name + '-' + suffix)}",
                stdout=archive,
            )
            archive.seek(0)
            with tarfile.open(fileobj=archive) as tar:
                tar.extractall(target.parent, filter="data")
        (target.parent / f"{name}-{suffix}").rename(target)
        print(f"retrieved {target}")


def evaluate(root, name):
    """Track-level evaluation on the oracle crops and the 200-crop evaluation, both local."""
    base = root / "artifacts/integrations"
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/run_track_identity_benchmark.py",
            "evaluate",
            "--benchmark",
            ORACLE,
            "--predictions",
            str(base / f"jersey-classifier-{name}-oracle"),
        ],
        check=True,
        cwd=root,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/run_jersey_benchmark.py",
            "evaluate",
            "--crop-manifest",
            f"{FROZEN}/crop-manifest.json",
            "--predictions",
            str(base / f"jersey-classifier-{name}-frozen" / "predictions.json"),
            "--truth",
            "artifacts/vision-benchmark/ground-truth.json",
            "--out",
            str(base / f"jersey-classifier-{name}-frozen" / "evaluation.json"),
        ],
        check=True,
        cwd=root,
    )


def chain(root, name):
    wait(name)
    retrieve(root, name)
    evidence = json.loads((root / "artifacts/models" / name / "training-evidence.json").read_text())
    if not evidence.get("completed"):
        raise RuntimeError(f"{name} did not complete: {evidence}")
    print("best epoch", evidence["best_epoch"], json.dumps(evidence["best_valid"]))
    infer(name)
    retrieve_predictions(root, name)
    evaluate(root, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=(
            "stage",
            "stage-code",
            "test",
            "train",
            "wait",
            "retrieve",
            "infer",
            "retrieve-predictions",
            "evaluate",
            "chain",
        ),
    )
    parser.add_argument("--name", default="jersey-resnet34-synth")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.operation == "stage":
        stage(root)
    elif args.operation == "stage-code":
        stage(root, data=False)
    elif args.operation == "test":
        test()
    elif args.operation == "train":
        train(args.name, args.epochs, args.smoke)
    elif args.operation == "wait":
        wait(args.name)
    elif args.operation == "retrieve":
        retrieve(root, args.name)
    elif args.operation == "infer":
        infer(args.name)
    elif args.operation == "retrieve-predictions":
        retrieve_predictions(root, args.name)
    elif args.operation == "evaluate":
        evaluate(root, args.name)
    else:
        chain(root, args.name)


if __name__ == "__main__":
    main()
