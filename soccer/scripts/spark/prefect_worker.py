"""Submit, inspect, resume, or serve a durable Spark video flow.

Run from the repository root with PYTHONPATH=src. No remote services are installed.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from soccerviz.providers.execution import JobSpec, SparkExecutor, build_spec, prefect_flow

# A module-level entrypoint is required for Prefect to reload a served deployment
# in a fresh worker process; the configurable flow factory itself is a closure.
try:
    from prefect import flow
except ImportError:
    run_spark_job = None
else:

    @flow(name="soccerviz-spark-deployment", retries=0, persist_result=False)
    def run_spark_job(spec_data, files, options):
        return prefect_flow()(spec_data, files, options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("artifacts/execution"))
    parser.add_argument("--host", default="spark")
    parser.add_argument("--remote-root", default="~/soccerviz-execution")
    parser.add_argument("--max-concurrency", type=int, default=1)
    sub = parser.add_subparsers(dest="command", required=True)
    video = sub.add_parser("video")
    video.add_argument("--image", required=True, help="Pinned Docker image sha256 ID/digest")
    video.add_argument("--seconds", type=float, default=2)
    video.add_argument("--hz", type=float, default=2)
    video.add_argument("--attempt", type=int, default=0)
    video.add_argument("--source", type=Path, default=Path("data/demo/2e57b9_0.mp4"))
    video.add_argument("--models", type=Path, default=Path("data/demo"))
    for command in ("check", "resume", "retrieve"):
        sub.add_parser(command).add_argument("job_id")
    serve = sub.add_parser("serve")
    serve.add_argument("--name", default="soccerviz-spark")
    args = parser.parse_args()
    options = {
        "store": str(args.store.resolve()),
        "host": args.host,
        "remote_root": args.remote_root,
        "max_concurrency": args.max_concurrency,
    }
    executor = SparkExecutor(**options)
    if args.command == "serve":
        # Prefect API must be persistent; this is an explicit, foreground service.
        if run_spark_job is None:
            raise ImportError(
                "Run scripts/setup/setup_integrations.py execution and use .venvs/execution/bin/python"
            )
        run_spark_job.serve(name=args.name, limit=args.max_concurrency)
        return
    if args.command == "video":
        if args.seconds <= 0 or args.hz <= 0:
            parser.error("Seconds and sampling rate must be positive")
        root = Path(__file__).resolve().parents[2]
        files = {
            p.relative_to(root).as_posix(): str(p) for p in (root / "src/soccerviz").rglob("*.py")
        }
        files["data/demo/source.mp4"] = str(args.source.resolve())
        for name in (
            "football-player-detection.pt",
            "football-pitch-detection.pt",
            "football-ball-detection.pt",
        ):
            files["data/demo/" + name] = str((args.models / name).resolve())
        spec = build_spec(
            args.image,
            [
                "python",
                "-m",
                "soccerviz.vision.pipeline",
                "--source",
                "data/demo/source.mp4",
                "--models",
                "data/demo",
                "--out",
                "/output",
                "--seconds",
                str(args.seconds),
                "--hz",
                str(args.hz),
            ],
            files,
            required_outputs=(
                "report.json",
                "frames.parquet",
                "detections.parquet",
                "state.parquet",
                "calibration.parquet",
                "landmarks.parquet",
                "ball_candidates.parquet",
            ),
            attempt=args.attempt,
        )
        print(json.dumps({"job_id": spec.job_id}), flush=True)
        result = prefect_flow()(asdict(spec), files, options)
    else:
        if len(args.job_id) != 64 or any(c not in "0123456789abcdef" for c in args.job_id):
            parser.error("Expected a full 64-character job ID")
        spec = JobSpec.from_dict(json.loads((args.store / args.job_id / "spec.json").read_text()))
        if args.command == "check":
            result = executor.check(spec)
        elif args.command == "retrieve":
            result = str(executor.retrieve(spec))
        else:
            result = prefect_flow()(asdict(spec), {}, options)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
