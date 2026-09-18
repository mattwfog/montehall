"""Stage, run and retrieve the SAM 3D Body mesh benchmark on Spark (docs/experiments/mesh-model.md)."""

import argparse
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path

# Manifests record absolute image paths from the machine that built them;
# the remote worker sees the same tree mounted at /work.
LOCAL_ROOT = Path(__file__).resolve().parents[2]

IMAGE = "soccerviz-mesh:20260908"
REMOTE = "~/soccerviz-mesh"
UPSTREAM_COMMIT = "b5c765a0d89d789985e186d396315e7590887b94"
# The upstream backbone calls torch.hub.load("facebookresearch/dinov3", ..., pretrained=False) at
# unpinned main; a checkout at this commit in the hub cache is what torch.hub then uses.
DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
HUB_CACHE = "artifacts/models/torch-hub/hub/facebookresearch_dinov3_main"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "spark"]
STAGED = [
    "src/soccerviz",
    "envs/mesh-model",
    "scripts/benchmarks/run_mesh_benchmark.py",
    "artifacts/vision-benchmark/manifest.json",
    "artifacts/vision-benchmark/protocol.json",
    "artifacts/vision-benchmark/yolov8x-soccer-predictions.json",
    "artifacts/vision-benchmark/sources",
]


def skip_pycache(member):
    """Container runs leave root-owned __pycache__ on Spark; never ship or overwrite bytecode."""
    return None if "__pycache__" in member.name else member


def remote(command, **kwargs):
    return subprocess.run(SSH + [command], check=True, **kwargs)


def stage(root):
    with tempfile.TemporaryFile() as archive:
        with tarfile.open(fileobj=archive, mode="w") as tar:
            for name in STAGED:
                tar.add(root / name, arcname=name, filter=skip_pycache)
        archive.seek(0)
        remote(f"mkdir -p {REMOTE} && tar -xf - -C {REMOTE}", stdin=archive)
    remote(
        f"set -e; cd {REMOTE}; mkdir -p artifacts/upstreams artifacts/models/hf artifacts/integrations logs;"
        " [ -d artifacts/upstreams/sam-3d-body/.git ] ||"
        " git clone -q https://github.com/facebookresearch/sam-3d-body.git artifacts/upstreams/sam-3d-body;"
        f" git -C artifacts/upstreams/sam-3d-body checkout -q {UPSTREAM_COMMIT};"
        f" [ -d {HUB_CACHE}/.git ] || git clone -q https://github.com/facebookresearch/dinov3 {HUB_CACHE};"
        f" git -C {HUB_CACHE} checkout -q {DINOV3_COMMIT};"
        ' python3 -c "from huggingface_hub import snapshot_download as s;'
        " print(s('facebook/sam-3d-body-dinov3', cache_dir='artifacts/models/hf'))\";"
        f" docker build -q -f envs/mesh-model/Dockerfile -t {IMAGE} ."
    )


def infer(name, limit):
    """Detached container; frames.jsonl resumes, so a rerun with the same name continues."""
    out = f"artifacts/integrations/{name}"
    command = (
        "python scripts/benchmarks/run_mesh_benchmark.py infer"
        " --manifest artifacts/vision-benchmark/manifest.json"
        " --detections artifacts/vision-benchmark/yolov8x-soccer-predictions.json"
        f" --out {out} --device cuda:0 --path-prefix {LOCAL_ROOT}=/work"
        + (f" --limit {limit}" if limit else "")
    )
    remote(
        f"cd {REMOTE} && docker run -d --rm --name mesh-{name} --gpus all --cpus 4 --shm-size 2g"
        " -e TORCH_HOME=/work/artifacts/models/torch-hub"
        f' -v "$HOME/soccerviz-mesh:/work" {IMAGE} sh -c {shlex.quote(command + f" > logs/{name}.log 2>&1")}'
    )
    print(f"started mesh-{name}; log {REMOTE}/logs/{name}.log")


def retrieve(root, name):
    out = f"artifacts/integrations/{name}"
    with tempfile.TemporaryFile() as archive:
        remote(f"tar -cf - -C {REMOTE} {shlex.quote(out)} logs/{name}.log", stdout=archive)
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar:
            tar.extractall(root, filter="data")
    print(f"retrieved {out} and logs/{name}.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stage")
    p = sub.add_parser("infer")
    p.add_argument("--name", default="mesh-sam-3d-body-dinov3")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("retrieve")
    p.add_argument("--name", default="mesh-sam-3d-body-dinov3")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.command == "stage":
        stage(root)
    elif args.command == "infer":
        if not args.name.replace("-", "").isalnum():
            parser.error("Use a simple alphanumeric run name")
        infer(args.name, args.limit)
    else:
        retrieve(root, args.name)


if __name__ == "__main__":
    main()
