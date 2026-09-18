"""Track->person re-link — THIN WRAPPER over the person registry.

The v4 lock engine moved to its canonical pipeline home,
montehall_cv/pipeline/person_registry.py, when the
people-only identity design was ratified (2026-08-25: Frigate-style
persistent body IDs, no number identity). The engine there carries NO
number-based identity paths — no number attach, no veto, no
(team,number) exclusivity; a jersey number is a post-hoc ATTRIBUTE.

This wrapper keeps the render-facing contract: same CLI, same output
json (segments + person_meta) consumed by people_track_video
--relink-map. It builds (or reuses) the person_registry/person_meta
stages and emits the json view of them.

Usage (inside cvbench, GPU):
    python scripts/people_relink_smoke.py --video "<file>" \
        --game-dir /work/out-harvest/<job> \
        --reads-dir /work/out-harvest/<job> \
        --binding-stage /work/out-harvest/<job>/quark_binding \
        --reid-weights /work/models/current/reid.pt \
        --out /work/models/<x>/relink.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from montehall_cv.pipeline.person_registry import build
from montehall_cv.store.artifacts import read_stage


def _json_from_stages(game_dir: Path) -> dict:
    segs = read_stage(game_dir / "person_registry").to_pylist()
    meta = read_stage(game_dir / "person_meta").to_pylist()
    return {
        "segments": [[r["track_id"], r["t0_ms"], r["t1_ms"],
                      r["person_id"]] for r in segs],
        "person_meta": {
            str(r["person_id"]): {
                "team": r["team"], "number": r["number_attr"],
                "tracks": r["segments"], "gallery": r["gallery"],
                "lifetime_ms": r["lifetime_ms"],
            } for r in meta},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--game-dir", type=Path, required=True)
    ap.add_argument("--reads-dir", type=Path, default=None)
    ap.add_argument("--reid-weights", type=Path, required=True)
    ap.add_argument("--binding-stage", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=0.0)
    ap.add_argument("--sim-min", type=float, default=0.5)
    args = ap.parse_args()

    summary = build(args.video, args.game_dir.parent, args.game_dir.name,
                    args.reid_weights, binding_stage=args.binding_stage,
                    reads_dir=args.reads_dir, start_s=args.start_s,
                    duration_s=args.duration_s, sim_min=args.sim_min)
    if summary.get("skipped"):
        view = _json_from_stages(args.game_dir)
        out = {"version": 5, "from_existing_stage": True, **view}
    else:
        out = {
            "version": 5,
            "window_s": [args.start_s, args.start_s + args.duration_s],
            "sim_min": args.sim_min,
            **{k: v for k, v in summary.items()
               if k not in ("segments_rows", "person_meta")},
            "segments": summary["segments_rows"],
            "person_meta": {str(p): m
                            for p, m in summary["person_meta"].items()},
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print("RELINK_DONE " + json.dumps(
        {k: v for k, v in out.items()
         if k not in ("segments", "person_meta")}))


if __name__ == "__main__":
    main()
