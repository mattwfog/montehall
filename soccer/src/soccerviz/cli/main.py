from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="SoccerViz research workbench")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    sub = parser.add_subparsers(dest="command", required=True)
    from soccerviz.cli.datasets import add_parser as add_datasets_parser
    from soccerviz.cli.harness import add_parser
    from soccerviz.cli.integrations import add_parser as add_integrations_parser

    add_parser(sub)
    add_integrations_parser(sub)
    add_datasets_parser(sub)
    for cmd in ("fetch", "prepare", "analyze"):
        item = sub.add_parser(cmd)
        item.add_argument("--games", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    train = sub.add_parser("train")
    train.add_argument("--train-game", type=int, default=1, choices=[1, 2])
    train.add_argument("--eval-game", type=int, default=2, choices=[1, 2])
    launch = sub.add_parser("workbench")
    launch.add_argument("--port", type=int, default=7865)
    forecast = sub.add_parser("train-forecast")
    forecast.add_argument("--epochs", type=int, default=8)
    forecast.add_argument("--device", default="cuda")
    for name in (
        "research",
        "tactics",
        "uncertainty",
        "manager",
        "inventory",
        "fetch-demo",
        "actors",
    ):
        sub.add_parser(name)
    lineup = sub.add_parser("lineup")
    lineup.add_argument("--context", type=Path, required=True)
    for name in ("train-ball", "train-probabilistic"):
        command = sub.add_parser(name)
        command.add_argument("--epochs", type=int, default=8)
        command.add_argument("--device", default="cuda")
    video = sub.add_parser("video")
    video.add_argument("--source", type=Path, required=True)
    video.add_argument("--models", type=Path, default=Path("data/demo"))
    video.add_argument("--out", type=Path, required=True)
    video.add_argument("--seconds", type=float, default=20)
    video.add_argument("--device", default="cuda:0")
    jersey = sub.add_parser("jersey")
    jersey.add_argument("--video-run", type=Path, default=Path("artifacts/video/demo-v2"))
    evaluation = sub.add_parser("evaluate-video")
    evaluation.add_argument("--video-run", type=Path, required=True)
    evaluation.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "datasets":
        from soccerviz.cli.datasets import dispatch

        dispatch(args)
    elif args.command == "integrations":
        from soccerviz.cli.integrations import dispatch

        dispatch(args)
    elif args.command == "harness":
        from soccerviz.cli.harness import dispatch

        dispatch(args)
    elif args.command == "fetch":
        from soccerviz.core.data import fetch

        fetch(args.data, args.games)
    elif args.command == "prepare":
        from soccerviz.core.data import prepare

        for game in args.games:
            print(json.dumps(prepare(args.data, game), indent=2))
    elif args.command == "analyze":
        from soccerviz.core.analysis import analyze

        for game in args.games:
            print(json.dumps(analyze(args.data, game), indent=2))
    elif args.command == "train":
        from soccerviz.core.model import train_baseline

        print(
            json.dumps(
                train_baseline(args.data, args.artifacts, args.train_game, args.eval_game), indent=2
            )
        )
    elif args.command == "train-forecast":
        from soccerviz.modeling.forecast import train_forecast

        print(
            json.dumps(
                train_forecast(args.data, args.artifacts, args.epochs, args.device), indent=2
            )
        )
    elif args.command == "lineup":
        from soccerviz.modeling.manager import lineup_assignment

        context = json.loads(args.context.read_text())
        print(json.dumps(lineup_assignment(context["profiles"], context["roles"]), indent=2))
    elif args.command == "actors":
        from soccerviz.vision.ball import evaluate_actors

        print(json.dumps(evaluate_actors(args.data, args.artifacts / "actor"), indent=2))
    elif args.command == "train-ball":
        from soccerviz.vision.ball import train_ball

        print(
            json.dumps(
                train_ball(args.data, args.artifacts / "ball", args.epochs, args.device), indent=2
            )
        )
    elif args.command == "train-probabilistic":
        from soccerviz.modeling.probabilistic import train_probabilistic

        print(
            json.dumps(
                train_probabilistic(
                    args.data, args.artifacts / "probabilistic", args.epochs, args.device
                ),
                indent=2,
            )
        )
    elif args.command == "research":
        from soccerviz.core.suite import run_suite

        print(json.dumps(run_suite(args.data, args.artifacts), indent=2))
    elif args.command == "inventory":
        from soccerviz.core.suite import collect_results

        print(json.dumps(collect_results(args.artifacts), indent=2))
    elif args.command == "tactics":
        from soccerviz.modeling.tactics import run_tactics

        print(json.dumps(run_tactics(args.data, args.artifacts / "tactics"), indent=2))
    elif args.command == "uncertainty":
        from soccerviz.modeling.uncertainty import run_uncertainty

        print(json.dumps(run_uncertainty(args.data, args.artifacts / "uncertainty"), indent=2))
    elif args.command == "manager":
        from soccerviz.modeling.manager import run_manager

        print(json.dumps(run_manager(args.data, args.artifacts), indent=2))
    elif args.command == "fetch-demo":
        from soccerviz.core.assets import fetch_demo

        print(json.dumps(fetch_demo(args.data / "demo"), indent=2))
    elif args.command == "video":
        from soccerviz.vision.pipeline import run_video

        print(
            json.dumps(
                run_video(args.source, args.models, args.out, args.seconds, device=args.device),
                indent=2,
            )
        )
    elif args.command == "jersey":
        from soccerviz.vision.jersey import run_jersey

        print(json.dumps(run_jersey(args.video_run, args.artifacts / "jersey"), indent=2))
    elif args.command == "evaluate-video":
        from soccerviz.datasets.evaluation import evaluate_annotations

        print(json.dumps(evaluate_annotations(args.video_run, args.annotations), indent=2))
    elif args.command == "workbench":
        from soccerviz.ui.workbench import build_app

        build_app(args.data, args.artifacts).launch(
            server_name="127.0.0.1", server_port=args.port, share=False
        )


if __name__ == "__main__":
    main()
