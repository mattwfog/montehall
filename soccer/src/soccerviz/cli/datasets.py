"""Public-data worker dispatch and shared experiment catalog."""

import argparse
import json
from pathlib import Path

from soccerviz.cli.integrations import read_json, run_worker
from soccerviz.datasets.dataset_catalog import DatasetCatalog

WORKERS = {
    "soccernet": "soccerviz.datasets.soccernet_adapter",
    "soccernet-evaluate": "soccerviz.datasets.soccernet_evaluation",
    "roboflow": "soccerviz.datasets.roboflow_universe",
    "skillcorner": "soccerviz.providers.skillcorner",
    "statsbomb": "soccerviz.datasets.statsbomb_dataset",
    "train-actions": "soccerviz.modeling.action_training",
}


def configure(parser):
    parser.add_argument("--store", type=Path, default=Path("artifacts/harness"))
    sub = parser.add_subparsers(dest="dataset_command", required=True)
    sub.add_parser("list", help="List immutable dataset experiment records")
    for name in ("inspect", "verify"):
        command = sub.add_parser(name)
        command.add_argument("record_id")
    register = sub.add_parser("register", help="Snapshot a report in the shared evidence store")
    register.add_argument("--name", required=True)
    register.add_argument("--kind", required=True)
    register.add_argument("--split", required=True)
    register.add_argument("--report", type=Path, required=True)
    register.add_argument("--artifact", type=Path, action="append", default=[])
    run = sub.add_parser(
        "run", help="Dispatch a pinned-data specialist through an isolated interpreter"
    )
    run.add_argument("worker", choices=sorted(WORKERS))
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--response", type=Path, required=True)
    run.add_argument("--python", type=Path)
    run.add_argument("--name", help="Register successful worker report with this display name")
    run.add_argument("--split", help="Required when registering; must describe actual source split")
    run.add_argument(
        "--artifact",
        type=Path,
        action="append",
        default=[],
        help="Produced file to verify; repeat for models/Parquet. Without this, registration verifies the report only.",
    )
    return parser


def add_parser(sub):
    configure(
        sub.add_parser("datasets", help="Public datasets, benchmarks and training experiments")
    )


def dispatch(args):
    catalog = DatasetCatalog(args.store)
    if args.dataset_command == "list":
        result = catalog.list()
    elif args.dataset_command == "inspect":
        result = catalog.get(args.record_id)
    elif args.dataset_command == "verify":
        result = catalog.verify(args.record_id)
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(1)
        return
    elif args.dataset_command == "register":
        result = catalog.register(args.name, args.kind, args.split, args.report, args.artifact)
    else:
        if args.name and not args.split:
            raise ValueError("--split is required with --name")
        if args.response.exists():
            raise ValueError("Choose a new response path; existing reports are retained")
        request = read_json(args.request)
        result = run_worker(WORKERS[args.worker], request, args.response, args.python)
        if args.name:
            result = catalog.register(
                args.name, args.worker, args.split, args.response, args.artifact
            )
    print(json.dumps(result, indent=2))


def main():
    dispatch(configure(argparse.ArgumentParser(description=__doc__)).parse_args())


if __name__ == "__main__":
    main()
