"""Run the pinned SAM 3D Body backend on frozen detector boxes (CUDA), then summarize."""

import argparse
import json
from pathlib import Path

from soccerviz.candidates.mesh_model import BACKENDS, infer, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("infer")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--detections", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="Directory; resumes frames.jsonl")
    p.add_argument("--backend", default="sam-3d-body-dinov3", choices=sorted(BACKENDS))
    p.add_argument("--cache-dir", default=Path("artifacts/models/hf"), type=Path)
    p.add_argument("--upstream", default=Path("artifacts/upstreams/sam-3d-body"), type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, help="Smoke prefix only; output marked incomplete")
    p.add_argument("--save-vertices", action="store_true", help="float16 npz per frame")
    p.add_argument("--path-prefix", action="append", default=[], metavar="OLD=NEW")
    p = sub.add_parser("summarize")
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "infer":
        prefixes = []
        for value in args.path_prefix:
            old, sep, new = value.partition("=")
            if not sep or not old or not new:
                parser.error("path-prefix requires OLD=NEW")
            prefixes.append((Path(old), Path(new)))
        result = infer(
            args.manifest,
            args.detections,
            args.out,
            backend_name=args.backend,
            cache_dir=args.cache_dir,
            device=args.device,
            upstream=args.upstream,
            limit=args.limit,
            prefixes=prefixes,
            save_vertices=args.save_vertices,
        )
    else:
        result = summarize(args.predictions, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
