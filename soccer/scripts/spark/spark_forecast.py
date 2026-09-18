"""Sync this experiment to ssh spark, train in a disposable container, retrieve results.

Usage: python scripts/spark/spark_forecast.py [--epochs 8]
Uses only ~/soccerviz on the remote host and does not modify existing containers.
"""

import argparse
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    root = Path(__file__).resolve().parents[2]
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "spark"]
    files = [
        "src/soccerviz/__init__.py",
        "src/soccerviz/modeling/__init__.py",
        "src/soccerviz/modeling/forecast.py",
        "data/processed/game_1/tracking.npz",
        "data/processed/game_2/tracking.npz",
    ]
    with tempfile.TemporaryFile() as archive:
        with tarfile.open(fileobj=archive, mode="w") as tar:
            for name in files:
                tar.add(root / name, arcname=name)
        archive.seek(0)
        subprocess.run(
            ssh + ["mkdir -p ~/soccerviz && tar -xf - -C ~/soccerviz"], stdin=archive, check=True
        )
    command = (
        "docker run --rm --gpus all --cpus 4 --shm-size 1g "
        "-e OMP_NUM_THREADS=4 -e PYTHONPATH=/work/src "
        '-v "$HOME/soccerviz:/work" -w /work nvcr.io/nvidia/pytorch:26.01-py3 '
        f"python -m soccerviz.modeling.forecast --epochs {args.epochs} --device cuda"
    )
    subprocess.run(ssh + [command], check=True)
    out = root / "artifacts"
    out.mkdir(exist_ok=True)
    for name in ["forecast.pt", "forecast-report.json", "forecast-evaluation.npz"]:
        temporary = out / (name + ".download")
        with temporary.open("wb") as dest:
            subprocess.run(ssh + [f"cat ~/soccerviz/artifacts/{name}"], stdout=dest, check=True)
        temporary.replace(out / name)
    print(f"Saved forecast model, report, and predictions to {out}")


if __name__ == "__main__":
    main()
