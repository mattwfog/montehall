"""Prepare predicted crops, execute JNR or the digit reader, and evaluate in separate commands."""

import argparse
import json
from pathlib import Path

from soccerviz.candidates import jersey_digits
from soccerviz.candidates.jersey_model import evaluate, infer, prepare, tesseract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--detections", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--limit", default=200, type=int)
    p.add_argument("--path-prefix", action="append", default=[])
    p = sub.add_parser("infer")
    p.add_argument("--crop-manifest", required=True, type=Path)
    p.add_argument("--upstream", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("infer-digits", help="RF-DETR digit reader, same prediction contract")
    p.add_argument("--crop-manifest", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resolution", required=True, type=int, help="training resolution")
    p = sub.add_parser("evaluate")
    p.add_argument("--crop-manifest", required=True, type=Path)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--truth", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--path-prefix", action="append", default=[])
    p = sub.add_parser("tesseract")
    p.add_argument("--crop-manifest", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prefixes = []
    for value in getattr(args, "path_prefix", []):
        old, sep, new = value.partition("=")
        if not sep or not old or not new:
            parser.error("path-prefix requires OLD=NEW")
        prefixes.append((Path(old), Path(new)))
    if args.command == "prepare":
        result = prepare(args.manifest, args.detections, args.out, args.limit, prefixes)
    elif args.command == "infer":
        result = infer(
            args.crop_manifest,
            args.upstream,
            args.checkpoint,
            args.out,
            args.batch_size,
            args.device,
            args.limit,
        )
    elif args.command == "infer-digits":
        result = jersey_digits.infer(
            args.crop_manifest,
            args.checkpoint,
            args.out,
            device=args.device,
            resolution=args.resolution,
        )
    elif args.command == "evaluate":
        result = evaluate(args.crop_manifest, args.predictions, args.truth, args.out, prefixes)
    else:
        result = tesseract(args.crop_manifest, args.out)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("crops", "predictions", "records")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
