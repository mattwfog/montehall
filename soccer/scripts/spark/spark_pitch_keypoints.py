"""Spark launcher for the permissive pitch calibrator: stage, train, retrieve, benchmark.

Runs inside the workflow image (torch, timm, cv2 already pinned there). Training
writes under ~/soccerviz-pitch-keypoints/results/<name>; the 45-frame calibration
benchmark runs in the candidates workspace that already holds the frozen frames.
`chain` waits for a launched run, then retrieves, benchmarks and evaluates it
unattended; `test` runs the module's pytest file inside the image, where torch is.
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

# Manifests record absolute image paths from the machine that built them;
# the remote worker sees the same tree mounted at /work.
LOCAL_ROOT = Path(__file__).resolve().parents[2]

IMAGE = "soccerviz-workflow:20260908-e50"
REMOTE = "~/soccerviz-pitch-keypoints"
BENCH_WORK = "~/soccerviz-candidates/workspace"
EXPORT = "artifacts/public-data/roboflow/football-field-detection-f07vi-v18-coco"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "spark"]
SHIPPED = [
    "src/soccerviz",
    "scripts/training/train_pitch_keypoints.py",
    "scripts/benchmarks/run_calibration_benchmark.py",
    "tests/candidates/test_pitch_keypoints.py",
    "tests/core/test_pitch_template.py",
    "tests/core/test_primitive_calibration.py",
]
TESTS = " ".join(name for name in SHIPPED if name.startswith("tests/"))
CORPUS = "artifacts/vision-benchmark"
CALIBRATION = f"{CORPUS}/calibration"


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


def stage(root):
    remote(f"mkdir -p {REMOTE}/results {REMOTE}/hf-cache")
    ship(root, SHIPPED + [EXPORT], REMOTE)
    ship(root, SHIPPED, BENCH_WORK)
    print(f"staged code and the v18 export under {REMOTE}; benchmark code under {BENCH_WORK}")


def train(name, epochs, smoke):
    command = (
        f"python scripts/training/train_pitch_keypoints.py --export {EXPORT}"
        f" --out results/{name} --epochs {epochs} --workers 4 --device cuda:0"
        + (" --smoke-step" if smoke else "")
    )
    docker = (
        f"docker run -d --rm --name train-pitch-{name} --gpus all --cpus 8 --shm-size 8g"
        f' -v "$HOME/soccerviz-pitch-keypoints:/work" -w /work -e PYTHONPATH=/work/src'
        f" -e HF_HOME=/work/hf-cache {IMAGE}"
        f" sh -c {shlex.quote(command + f' > results/{name}.log 2>&1; chmod -R a+rX results/{name} results/{name}.log')}"
    )
    remote(f"cd {REMOTE} && {docker}")
    print(f"started train-pitch-{name}; log {REMOTE}/results/{name}.log")


def test():
    """The module's pytest files inside the image: the torch-only tests skip locally."""
    docker = (
        f'docker run --rm --gpus all -v "$HOME/soccerviz-pitch-keypoints:/work" -w /work'
        f" -e PYTHONPATH=/work/src {IMAGE}"
        ' sh -c \'python -c "import pytest" 2>/dev/null || pip install -q pytest;'
        f" python -m pytest -q {TESTS}'"
    )
    remote(f"cd {REMOTE} && {docker}")


def wait(name):
    """Block until the training container is gone, then print the log tail."""
    remote(
        f'until [ -z "$(docker ps -q --filter name=train-pitch-{shlex.quote(name)})" ];'
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


CALIBRATION_IMAGE = (
    "soccerviz-calibration:20260907"  # PnLCalib upstream + weights, built 2026-09-07
)


def corpus_paths(corpus):
    """Manifest, calibration protocol and calibration output directory of a corpus root
    (`artifacts/vision-benchmark` is the frozen development sample; the game-disjoint
    evaluation lives at `artifacts/calibration-benchmark`)."""
    corpus = corpus.rstrip("/")
    return f"{corpus}/manifest.json", f"{corpus}/calibration/protocol.json", f"{corpus}/calibration"


def benchmark(name, prefix, corpus=CORPUS):
    """Calibration benchmark with best.pt, same protocol file as PnLCalib and YOLO."""
    manifest, protocol, out_dir = corpus_paths(corpus)
    command = (
        "python scripts/benchmarks/run_calibration_benchmark.py run"
        f" --manifest {manifest} --protocol {protocol}"
        " --backend heatmap --checkpoint /trained/best.pt --device cuda:0"
        f" --path-prefix {LOCAL_ROOT}=/work"
        f" --out {out_dir}/{prefix}-predictions.json"
    )
    docker = (
        f"docker run --rm --name bench-pitch-{name} --gpus all --cpus 8"
        f' -v "$HOME/soccerviz-candidates/workspace:/work" -w /work -e PYTHONPATH=/work/src'
        f' -v "$HOME/soccerviz-pitch-keypoints/results/{name}:/trained:ro" {IMAGE}'
        f" sh -c {shlex.quote(command)}"
    )
    remote(f"cd {BENCH_WORK} && {docker} | tail -3")
    print(f"benchmark written to {BENCH_WORK}/{out_dir}/{prefix}-predictions.json")


def pnl_benchmark(prefix, corpus=CORPUS):
    """PnLCalib (GPL, evaluation only) on the same corpus, the incumbent to beat; the
    2026-09-07 run on the development sample was launched by hand with this command."""
    manifest, protocol, out_dir = corpus_paths(corpus)
    command = (
        "python scripts/benchmarks/run_calibration_benchmark.py run"
        f" --manifest {manifest} --protocol {protocol}"
        " --backend pnlcalib --upstream artifacts/upstreams/pnlcalib"
        " --weights-keypoints /models/pnl-keypoints.pt --weights-lines /models/pnl-lines.pt"
        f" --device cuda:0 --path-prefix {LOCAL_ROOT}=/work"
        f" --out {out_dir}/{prefix}-predictions.json"
    )
    docker = (
        f"docker run --rm --name bench-pnl-{prefix} --gpus all --cpus 8"
        f' -v "$HOME/soccerviz-candidates/workspace:/work" -w /work -e PYTHONPATH=/work/src'
        f' -v "$HOME/soccerviz-candidates/shared/weights:/models:ro" {CALIBRATION_IMAGE}'
        f" sh -c {shlex.quote(command)}"
    )
    remote(f"cd {BENCH_WORK} && {docker} | tail -3")
    print(f"benchmark written to {BENCH_WORK}/{out_dir}/{prefix}-predictions.json")


def retrieve_benchmark(root, prefix, corpus=CORPUS):
    _, _, out_dir = corpus_paths(corpus)
    target = root / out_dir / f"{prefix}-predictions.json"
    if target.exists():
        raise FileExistsError(f"{target} exists; benchmark outputs are never overwritten")
    remote(
        f"cat {BENCH_WORK}/{out_dir}/{shlex.quote(prefix + '-predictions.json')}",
        stdout=target.open("wb"),
    )
    print(f"retrieved {target}")


def evaluate(root, prefix, corpus=CORPUS):
    """Local evaluation against the corpus ground truth, same command as PnLCalib and YOLO."""
    manifest, protocol, out_dir = corpus_paths(corpus)
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/run_calibration_benchmark.py",
            "evaluate",
            "--manifest",
            manifest,
            "--protocol",
            protocol,
            "--predictions",
            f"{out_dir}/{prefix}-predictions.json",
            "--out",
            f"{out_dir}/{prefix}-results.json",
        ],
        check=True,
        cwd=root,
    )


def ship_corpus(root, corpus):
    """Copy a frozen corpus (manifest, protocol, sources, calibration protocol) to the
    Spark benchmark workspace; images are read there through --path-prefix."""
    remote(f"mkdir -p {BENCH_WORK}/{corpus}")
    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude",
            "*-predictions.json",
            "--exclude",
            "*-results*.json",
            "--exclude",
            "*-maps",
            f"{root / corpus}/",
            f"spark:{BENCH_WORK}/{corpus}/",
        ],
        check=True,
    )
    print(f"shipped {corpus} to {BENCH_WORK}/{corpus}")


def chain(root, name, prefix):
    """wait → retrieve → benchmark → retrieve-benchmark → evaluate, for an unattended run."""
    wait(name)
    retrieve(root, name)
    evidence = json.loads((root / "artifacts/models" / name / "training-evidence.json").read_text())
    if not evidence.get("completed"):
        raise RuntimeError(f"{name} did not complete: {evidence}")
    print("best epoch", evidence["best_epoch"], json.dumps(evidence["best_valid"]))
    benchmark(name, prefix)
    retrieve_benchmark(root, prefix)
    evaluate(root, prefix)
    overall = json.loads((root / CALIBRATION / f"{prefix}-results.json").read_text())["overall"]
    print(json.dumps(overall, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=(
            "stage",
            "test",
            "train",
            "wait",
            "retrieve",
            "benchmark",
            "pnl-benchmark",
            "retrieve-benchmark",
            "evaluate",
            "ship-corpus",
            "chain",
        ),
    )
    parser.add_argument("--name", default="pitch-hrnet-w32-v18-lines")
    parser.add_argument("--prefix", default="heatmap-hrnet-w32-v18-lines")
    parser.add_argument(
        "--corpus", default=CORPUS, help="corpus root, e.g. artifacts/calibration-benchmark"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.operation == "stage":
        stage(root)
    elif args.operation == "test":
        test()
    elif args.operation == "train":
        train(args.name, args.epochs, args.smoke)
    elif args.operation == "wait":
        wait(args.name)
    elif args.operation == "retrieve":
        retrieve(root, args.name)
    elif args.operation == "benchmark":
        benchmark(args.name, args.prefix, args.corpus)
    elif args.operation == "pnl-benchmark":
        pnl_benchmark(args.prefix, args.corpus)
    elif args.operation == "retrieve-benchmark":
        retrieve_benchmark(root, args.prefix, args.corpus)
    elif args.operation == "evaluate":
        evaluate(root, args.prefix, args.corpus)
    elif args.operation == "ship-corpus":
        ship_corpus(root, args.corpus)
    else:
        chain(root, args.name, args.prefix)


if __name__ == "__main__":
    main()
