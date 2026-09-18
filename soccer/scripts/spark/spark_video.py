"""Build an isolated CV image on Spark, run the public demo, retrieve stage outputs."""

import argparse
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--run-name", default="demo-v1")
    args = parser.parse_args()
    if not args.run_name.replace("-", "").replace("_", "").isalnum() or args.seconds <= 0:
        parser.error("Use a simple alphanumeric run name and positive duration")
    root = Path(__file__).resolve().parents[2]
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "spark"]
    names = [
        *root.glob("src/soccerviz/**/*.py"),
        root / "envs/cv/Dockerfile",
        *root.glob("data/demo/*"),
    ]
    with tempfile.TemporaryFile() as archive:
        with tarfile.open(fileobj=archive, mode="w") as tar:
            for name in names:
                tar.add(name, arcname=str(name.relative_to(root)))
        archive.seek(0)
        subprocess.run(
            ssh + ["mkdir -p ~/soccerviz && tar -xf - -C ~/soccerviz"], stdin=archive, check=True
        )
    subprocess.run(
        ssh
        + [
            "docker build -t soccerviz-cv:0.2 -f ~/soccerviz/envs/cv/Dockerfile ~/soccerviz/envs/cv"
        ],
        check=True,
    )
    output = f"artifacts/video/{args.run_name}"
    command = (
        "docker run --rm --gpus all --cpus 4 --shm-size 1g "
        '-v "$HOME/soccerviz:/work" soccerviz-cv:0.2 '
        f"python -m soccerviz.vision.pipeline --seconds {args.seconds} --out {shlex.quote(output)}"
    )
    subprocess.run(ssh + [command], check=True)
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            ssh + [f"tar -cf - -C ~/soccerviz {shlex.quote(output)}"], stdout=archive, check=True
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar:
            tar.extractall(root, filter="data")
    print(f"Retrieved {output}")


if __name__ == "__main__":
    main()
