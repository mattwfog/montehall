"""One command surface for annotation, evaluation and optional upstream workers."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from soccerviz.core.data import write_json
from soccerviz.harness.engine import Harness, digest


def read_json(path):
    return json.loads(Path(path).read_text())


def inventory():
    packages = {}
    for package in ("trackeval", "tracklab", "kloppy", "socceraction", "prefect"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages[package] = {
            "version": version,
            "import_available": importlib.util.find_spec(package) is not None,
        }
    return {
        "schema": "integration-inventory/v1",
        "python": sys.executable,
        "packages": packages,
        "note": "Package availability does not establish model accuracy or worker compatibility",
    }


def configure(parser):
    parser.add_argument("--store", type=Path, default=Path("artifacts/harness"))
    sub = parser.add_subparsers(dest="integration_command", required=True)
    sub.add_parser("inventory")
    export = sub.add_parser("cvat-export")
    export.add_argument("run_id")
    export.add_argument("--out", type=Path, required=True)
    load = sub.add_parser("cvat-import")
    load.add_argument("run_id")
    load.add_argument("--xml", type=Path, required=True)
    load.add_argument("--manifest", type=Path, required=True)
    load.add_argument("--review", type=Path, required=True)
    load.add_argument("--out", type=Path, required=True)
    load.add_argument(
        "--preview", action="store_true", help="Validate and export without applying corrections"
    )
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("run_id")
    evaluation.add_argument("--reviewed", type=Path, required=True)
    evaluation.add_argument(
        "--predictions", type=Path, help="Normalized tracker rows as JSON; defaults to frozen run"
    )
    evaluation.add_argument("--out", type=Path, required=True)
    evaluation.add_argument(
        "--python", type=Path, help="Optional isolated interpreter with official TrackEval"
    )
    tracking = sub.add_parser("track")
    tracking.add_argument("--video-run", type=Path, required=True)
    tracking.add_argument("--out", type=Path, required=True)
    tracking.add_argument("--max-age-s", type=float, default=0.8)
    tracking.add_argument("--python", type=Path)
    provider = sub.add_parser("provider")
    provider.add_argument("--home", type=Path, required=True)
    provider.add_argument("--away", type=Path, required=True)
    provider.add_argument("--out", type=Path, required=True)
    provider.add_argument("--sample-rate", type=float, default=0.2)
    provider.add_argument("--limit", type=int)
    provider.add_argument("--python", type=Path)
    actions = sub.add_parser("actions")
    actions.add_argument("--request", type=Path, required=True)
    actions.add_argument("--response", type=Path, required=True)
    actions.add_argument("--python", type=Path, required=True)
    remote = sub.add_parser("remote")
    remote.add_argument("--python", type=Path, required=True)
    remote.add_argument("--job-store", type=Path, default=Path("artifacts/execution"))
    remote.add_argument("--host", default="spark")
    remote.add_argument("--remote-root", default="~/soccerviz-execution")
    remote.add_argument("--max-concurrency", type=int, default=1)
    remote.add_argument("worker_args", nargs=argparse.REMAINDER)
    return parser


def add_parser(sub):
    configure(
        sub.add_parser("integrations", help="Upstream annotation, tracking and worker adapters")
    )


def worker_environment():
    return os.environ | {"PYTHONPATH": str(Path(__file__).resolve().parents[2])}


def run_worker(module, request, response, python=None):
    """Invoke a fixed adapter module; requests contain data, never executable shell fragments."""
    with tempfile.TemporaryDirectory(prefix="soccerviz-worker-") as folder:
        file = Path(folder) / "request.json"
        write_json(file, request)
        command = [
            str(Path(python).absolute()) if python else sys.executable,
            "-m",
            module,
            "--request",
            str(file),
            "--response",
            str(Path(response).resolve()),
        ]
        subprocess.run(command, check=True, env=worker_environment())
    return read_json(response)


def evaluate_payload(store, run_id, reviewed, predictions):
    from soccerviz.datasets.tracking_evaluation import evaluate_tracking, prediction_rows

    harness = Harness(store)
    context = harness.context(run_id)
    source = harness.get(context["input_id"])
    if reviewed["input_id"] != context["input_id"]:
        raise ValueError("Reviewed annotations belong to another frozen input")
    if predictions is None:
        rows = prediction_rows(source)
    else:
        if (
            not isinstance(predictions, dict)
            or predictions.get("source_sha256") != source["source_sha256"]
        ):
            raise ValueError("Alternative tracker predictions belong to another video")
        rows = predictions["predictions"]
    return evaluate_tracking(reviewed, rows, source_sha256=source["source_sha256"])


def dispatch(args):
    cmd = args.integration_command
    if cmd == "inventory":
        result = inventory()
    elif cmd == "cvat-export":
        from soccerviz.providers.annotation import export_cvat

        h = Harness(args.store)
        result = export_cvat(h.get(h.context(args.run_id)["input_id"]), args.out)
    elif cmd == "cvat-import":
        from soccerviz.providers.annotation import import_cvat
        from soccerviz.vision.specialists import specialists

        h = Harness(args.store)
        context = h.context(args.run_id)
        imported = import_cvat(
            args.xml, read_json(args.manifest), read_json(args.review), h.get(context["input_id"])
        )
        write_json(args.out, imported)
        result = {
            "imported": str(args.out.resolve()),
            "reviewed_boxes": len(imported["ground_truth"]),
            "contextual_corrections": len(imported["harness_reviews"]),
            "preview": args.preview,
        }
        if not args.preview:
            if imported["harness_reviews"]:
                receipt = h.review_batch(
                    args.run_id,
                    imported["harness_reviews"],
                    source_key=digest(imported),
                    expected_revision=context["revision"],
                )
                result["receipt"] = receipt
                result["execution"] = h.execute(
                    args.run_id, specialists(), revision=receipt["revision"]
                )
                revision = receipt["revision"]
            else:
                revision = context["revision"]
            result["attachment"] = h.attach(args.run_id, "reviewed-tracking", imported, revision)
    elif cmd == "evaluate":
        if args.python is not None:
            command = [
                str(args.python.absolute()),
                "-m",
                "soccerviz.cli.integrations",
                "--store",
                str(args.store.resolve()),
                "evaluate",
                args.run_id,
                "--reviewed",
                str(args.reviewed.resolve()),
                "--out",
                str(args.out.resolve()),
            ]
            if args.predictions:
                command.extend(["--predictions", str(args.predictions.resolve())])
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True, env=worker_environment()
            )
            print(completed.stdout, end="")
            return
        reviewed = read_json(args.reviewed)
        result = evaluate_payload(
            args.store,
            args.run_id,
            reviewed,
            read_json(args.predictions) if args.predictions else None,
        )
        write_json(args.out, result)
        Harness(args.store).attach(args.run_id, "tracking-evaluation", result)
    elif cmd == "track":
        result = run_worker(
            "soccerviz.providers.tracking",
            {
                "video_dir": str(args.video_run.resolve()),
                "out": str(args.out.resolve()),
                "max_age_s": args.max_age_s,
            },
            args.out / "worker-response.json",
            args.python,
        )
    elif cmd == "provider":
        result = run_worker(
            "soccerviz.providers.provider",
            {
                "home": str(args.home.resolve()),
                "away": str(args.away.resolve()),
                "out": str(args.out.resolve()),
                "sample_rate": args.sample_rate,
                "limit": args.limit,
            },
            args.out / "worker-response.json",
            args.python,
        )
    elif cmd == "actions":
        result = run_worker(
            "soccerviz.providers.action_value", read_json(args.request), args.response, args.python
        )
    elif cmd == "remote":
        if not args.worker_args:
            raise ValueError("Choose a remote operation: video, check, resume, retrieve or serve")
        script = Path(__file__).resolve().parents[3] / "scripts/spark/prefect_worker.py"
        command = [
            str(args.python.absolute()),
            str(script),
            "--store",
            str(args.job_store.resolve()),
            "--host",
            args.host,
            "--remote-root",
            args.remote_root,
            "--max-concurrency",
            str(args.max_concurrency),
            *args.worker_args,
        ]
        subprocess.run(command, check=True, env=worker_environment())
        return
    else:
        raise ValueError("Unknown integration operation")
    print(json.dumps(result, indent=2))


def main():
    dispatch(configure(argparse.ArgumentParser(description=__doc__)).parse_args())


if __name__ == "__main__":
    main()
