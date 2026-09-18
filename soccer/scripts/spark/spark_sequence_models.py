"""Run a probabilistic-player or ball forecast on Spark and retrieve its artifacts."""

import argparse
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["probabilistic", "ball"], default="probabilistic")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "spark"]
    files = [
        "src/soccerviz/__init__.py",
        "src/soccerviz/modeling/__init__.py",
        "src/soccerviz/vision/__init__.py",
        "src/soccerviz/modeling/forecast.py",
        "src/soccerviz/modeling/probabilistic.py",
        "src/soccerviz/vision/ball.py",
        "data/processed/game_1/tracking.npz",
        "data/processed/game_2/tracking.npz",
    ]
    with tempfile.TemporaryFile() as archive:
        with tarfile.open(fileobj=archive, mode="w") as tar:
            for file in files:
                tar.add(root / file, arcname=file)
        archive.seek(0)
        subprocess.run(
            ssh + ["mkdir -p ~/soccerviz && tar -xf - -C ~/soccerviz"], stdin=archive, check=True
        )
    subprocess.run(
        ssh
        + [
            (
                "docker run --rm --gpus all --cpus 4 --shm-size 1g "
                '-e OMP_NUM_THREADS=4 -e PYTHONPATH=/work/src -v "$HOME/soccerviz:/work" -w /work '
                f"nvcr.io/nvidia/pytorch:26.01-py3 python -m soccerviz.{args.model}"
            )
        ],
        check=True,
    )
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            ssh + [f"tar -cf - -C ~/soccerviz artifacts/{args.model}"], stdout=archive, check=True
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar:
            tar.extractall(root, filter="data")


if __name__ == "__main__":
    main()
