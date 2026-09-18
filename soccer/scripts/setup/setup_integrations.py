"""Create reproducible optional worker environments without changing the core environment."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runtime", choices=["all", "analysis", "evaluation", "execution", "soccernet"]
    )
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    runtimes = (
        ["analysis", "evaluation", "execution", "soccernet"]
        if args.runtime == "all"
        else [args.runtime]
    )
    for name in runtimes:
        folder = root / ".venvs" / name
        python = folder / "bin/python"
        if not python.exists():
            subprocess.run([args.uv, "venv", "--python", "3.12", str(folder)], check=True)
        requirements = (
            root / "envs" / ("integrations" if name == "analysis" else name) / "requirements.txt"
        )
        subprocess.run(
            [args.uv, "pip", "install", "--python", str(python), "-r", str(requirements)],
            check=True,
        )
        if name == "analysis":
            subprocess.run(
                [
                    args.uv,
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--no-deps",
                    "-r",
                    str(root / "envs/tracklab-plugin/requirements.txt"),
                ],
                check=True,
            )
        print(f"{name}: {python}", flush=True)


if __name__ == "__main__":
    main()
