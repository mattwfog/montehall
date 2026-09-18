"""Durable Spark dispatch for the integrated video worker; no implicit dataset uploads."""

import json
from pathlib import Path

from soccerviz.providers.execution import JobSpec, SparkExecutor, build_spec
from soccerviz.vision.video_workflow import CALIBRATIONS, PRESETS


def executor(artifacts):
    config = json.loads((Path(artifacts) / "workflow-runtime.json").read_text())
    return SparkExecutor(
        Path(artifacts) / "workflow-jobs",
        host=config["host"],
        remote_root=config["remote_root"],
        max_concurrency=1,
    ), config


def submit(source, artifacts, preset="rf-soccer", calibration="pnl", seconds=20, jersey=True):
    source, artifacts = Path(source).resolve(), Path(artifacts).resolve()
    if not source.is_file() or source.stat().st_size > 500_000_000:
        raise ValueError("Choose a local video no larger than 500 MB")
    if preset not in PRESETS or calibration not in CALIBRATIONS or not 0 < seconds <= 600:
        raise ValueError("Invalid workflow settings")
    worker, config = executor(artifacts)
    root = Path(__file__).resolve().parents[3]
    files = {str(p.relative_to(root)): str(p) for p in (root / "src/soccerviz").rglob("*.py")}
    files["input/source.mp4"] = str(source)
    command = [
        "python",
        "-m",
        "soccerviz.vision.video_workflow",
        "--source",
        "input/source.mp4",
        "--out",
        "/output/run",
        "--preset",
        preset,
        "--calibration",
        calibration,
        "--seconds",
        str(float(seconds)),
    ]
    if not jersey:
        command.append("--no-jersey")
    spec = build_spec(
        config["image"],
        command,
        files,
        required_outputs=("run/report.json", "run/preview.mp4", "run/files.json"),
    )
    state = worker.check(spec)
    if state["status"] == "missing":
        worker.stage(spec, files)
    if state["status"] not in {"completed", "failed"}:
        state = worker.submit(spec)
    return spec.job_id, state


def inspect(job_id, artifacts):
    if len(job_id) != 64 or any(c not in "0123456789abcdef" for c in job_id):
        raise ValueError("Invalid workflow job ID")
    worker, _ = executor(artifacts)
    spec = JobSpec.from_dict(json.loads((worker.store / job_id / "spec.json").read_text()))
    status = worker.check(spec)
    if status["status"] == "completed":
        folder = worker.retrieve(spec) / "run"
        status["video_run"] = str(folder)
    return status


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["submit", "inspect"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--job-id")
    parser.add_argument("--preset", choices=PRESETS, default="rf-soccer")
    parser.add_argument("--calibration", choices=CALIBRATIONS, default="pnl")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--no-jersey", action="store_true")
    args = parser.parse_args()
    if args.operation == "submit":
        job, status = submit(
            args.source,
            args.artifacts,
            args.preset,
            args.calibration,
            args.seconds,
            not args.no_jersey,
        )
        print(json.dumps({"job_id": job, **status}, indent=2))
    else:
        print(json.dumps(inspect(args.job_id, args.artifacts), indent=2))


if __name__ == "__main__":
    main()
