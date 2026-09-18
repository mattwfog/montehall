"""CLI for starting, resuming, inspecting, and correcting a harness run."""

import json
from pathlib import Path

from soccerviz.harness.colony_runtime import build_colony, execute_run
from soccerviz.harness.engine import Harness
from soccerviz.vision.specialists import import_video


def add_parser(sub):
    parser = sub.add_parser("harness", help="Unified soccer colony: video, tracking and review")
    parser.add_argument("--store", type=Path, default=Path("artifacts/harness"))
    commands = parser.add_subparsers(dest="harness_command", required=True)
    start = commands.add_parser("start")
    source = start.add_mutually_exclusive_group(required=True)
    source.add_argument("--video-run", type=Path)
    source.add_argument("--provider-run", type=Path)
    start.add_argument("--start-s", type=float, required=True)
    start.add_argument("--end-s", type=float, required=True)
    start.add_argument("--max-stages", type=int)
    for name in ("resume", "inspect", "export"):
        cmd = commands.add_parser(name)
        cmd.add_argument("run_id")
        cmd.add_argument("--revision", type=int)
        if name == "resume":
            cmd.add_argument("--max-stages", type=int)
    review = commands.add_parser("review")
    review.add_argument("run_id")
    review.add_argument("--file", type=Path, required=True)
    review.add_argument("--max-stages", type=int)
    registry = commands.add_parser(
        "registry", help="Show executable specialists and their dependencies"
    )
    registry.add_argument("--provider", action="store_true")
    assess = commands.add_parser(
        "assess", help="Check immutable measurements against explicit criteria"
    )
    assess.add_argument("run_id")
    assess.add_argument("--report", type=Path, required=True)
    assess.add_argument("--contract", type=Path)


def dispatch(args):
    harness = Harness(args.store)
    cmd = args.harness_command
    if cmd == "registry":
        result = build_colony({"schema": "provider-input/v1"} if args.provider else None).manifest()
    elif cmd == "assess":
        from soccerviz.harness.experiment_gates import AcceptanceContract, evaluate_acceptance

        report = json.loads(args.report.read_text())
        context = harness.context(args.run_id)
        source = harness.get(context["input_id"])
        if report.get("provenance", {}).get("source_sha256") != source["source_sha256"]:
            raise ValueError("Measurement source must match the selected run")
        with harness.connect() as db:
            report_id = harness.put(db, report)
        contract = (
            AcceptanceContract.from_dict(json.loads(args.contract.read_text()))
            if args.contract
            else None
        )
        completed = {r["stage"] for r in harness.status(args.run_id)["results"]}
        needed = {s.name for s in build_colony(source).execution_order()}
        result = evaluate_acceptance(
            harness.get(report_id),
            contract,
            runtime_status="completed" if needed <= completed else "incomplete",
        )
        harness.attach(args.run_id, "assessment_measurement", report)
        result["report_object_id"] = report_id
        result["assessment_object_id"] = harness.attach(
            args.run_id, "scientific_assessment", result
        )
    elif cmd == "start":
        if args.provider_run:
            from soccerviz.harness.provider_colony import import_provider

            source = import_provider(args.provider_run, args.start_s, args.end_s)
        else:
            source = import_video(args.video_run, args.start_s, args.end_s)
        run = harness.create(source)
        print(json.dumps({"created_run": run}), flush=True)
        result = execute_run(harness, run, max_stages=args.max_stages)
    elif cmd == "resume":
        result = execute_run(harness, args.run_id, args.revision, args.max_stages)
    elif cmd == "review":
        revision = harness.review(args.run_id, json.loads(args.file.read_text()))
        print(json.dumps({"created_revision": revision}), flush=True)
        result = execute_run(harness, args.run_id, revision, args.max_stages)
    elif cmd == "inspect":
        result = harness.status(args.run_id, args.revision)
    else:
        result = {"export": str(harness.export(args.run_id, args.revision).resolve())}
    print(json.dumps(result, indent=2))
