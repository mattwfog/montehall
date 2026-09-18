"""Frame-detector probe over a ball-track hole with alternate weights.

The broadcast hole (Citadel-BC seg830, frames 11458-11586, 4.27s, fully
inside a live span) holds ZERO detector ball rows and WASB at noise floor
(ball-component census, 2026-08-26). The 08-26 "detector-swap" chain that
was meant to test the supply hypothesis compared models/current against
itself (current -> finetune-v3-broadcast/checkpoint_best_total.pth, whose
tensors are the v2 init; the genuine v3 fine-tune is checkpoint_best_
regular.pth, tensor hash 26dc9ff2 - the 08-01 checkpoint finding).

This probe answers the question directly and cheaply: decode the hole
window ONCE, run each --weights checkpoint over it at a low threshold,
and report per-frame ball supply per checkpoint at several thresholds.
A CONTROL window (the densest det-source span of the shipped ball_track,
same length as the hole) runs through the same checkpoints so a "sees
nothing" verdict discriminates weights from footage: a checkpoint that
is blind on the control too says nothing about the hole.

Read-only against stage artifacts; writes only --out.

Usage (inside cvbench, GPU):
    python scripts/ball_gap_detector_probe.py \
        --video /work/citadel_bc_seg830_20260826.mp4 \
        --game-dir /work/out-harvest/citadel_bc_seg830_20260826 \
        --frames 11390 11595 \
        --weights current=/work/models/current/detector.pth \
        --weights v3real=/work/models/finetune-v3-broadcast/checkpoint_best_regular.pth \
        --weights stock=none \
        --out /work/models/quark-v1/ball_gap_detector_citadelbc.json
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from montehall_cv.pipeline.detect import RFDetrDetector
from montehall_cv.pipeline.video import DecodedFrame, decode_frames
from montehall_cv.store.records import DetClass

PROBE_THRESHOLD = 0.05
REPORT_THRESHOLDS = (0.05, 0.2, 0.4)
CONTROL_SOURCES = ("det", "both")


def _parse_weights(specs: list[str]) -> list[tuple[str, Path | None]]:
    out: list[tuple[str, Path | None]] = []
    for spec in specs:
        label, _, path = spec.partition("=")
        if not label or not path:
            raise SystemExit(f"--weights needs label=path, got {spec!r}")
        out.append((label, None if path.lower() == "none" else Path(path)))
    return out


def _densest_det_window(game_dir: Path, length: int) -> tuple[int, int]:
    """Frame span [lo, lo+length] of ball_track with the most det-source rows."""
    files = sorted(glob.glob(str(game_dir / "ball_track" / "*.parquet")))
    if not files:
        raise SystemExit(f"no ball_track parquet under {game_dir}")
    rows = pq.read_table(files[0], columns=["frame_idx", "source"]).to_pylist()
    det_frames = sorted({r["frame_idx"] for r in rows if r["source"] in CONTROL_SOURCES})
    if not det_frames:
        raise SystemExit("ball_track has no det-source rows; no control window")
    best_lo, best_n, j = det_frames[0], 0, 0
    for i, lo in enumerate(det_frames):
        while j < len(det_frames) and det_frames[j] <= lo + length:
            j += 1
        if j - i > best_n:
            best_lo, best_n = lo, j - i
    return best_lo, best_lo + length


def _collect_frames(video: Path, spans: list[tuple[int, int]]) -> list[DecodedFrame]:
    hi_all = max(hi for _, hi in spans)
    keep = [f for f in _iter_until(video, hi_all) if any(lo <= f.frame_idx <= hi for lo, hi in spans)]
    return keep


def _iter_until(video: Path, hi: int):
    for frame in decode_frames(video):
        if frame.frame_idx > hi:
            return
        yield frame


def _run_weights(label: str, weights: Path | None, frames: list[DecodedFrame], batch: int) -> list[dict]:
    detector = RFDetrDetector(threshold=PROBE_THRESHOLD, batch_size=batch, weights=weights)
    rows: list[dict] = []
    for start in range(0, len(frames), batch):
        chunk = frames[start:start + batch]
        for frame, det in zip(chunk, detector.detect([f.image for f in chunk])):
            ball = det.conf[det.cls == int(DetClass.BALL)]
            rows.append({
                "weights": label,
                "frame_idx": frame.frame_idx,
                "ts_ms": frame.ts_ms,
                "n_ball": int(ball.size),
                "max_ball_conf": float(ball.max()) if ball.size else 0.0,
                "n_person": int((det.cls == int(DetClass.PERSON)).sum()),
                "n_rim": int((det.cls == int(DetClass.RIM)).sum()) if hasattr(DetClass, "RIM") else None,
            })
    return rows


def _summary(rows: list[dict], lo: int, hi: int) -> dict:
    inside = [r for r in rows if lo <= r["frame_idx"] <= hi]
    n = len(inside)
    confs = np.array([r["max_ball_conf"] for r in inside], dtype=np.float32)
    return {
        "frames": n,
        "frames_with_ball_at": {
            str(t): int((confs >= t).sum()) for t in REPORT_THRESHOLDS
        },
        "ball_frame_fraction_at": {
            str(t): round(float((confs >= t).sum()) / max(n, 1), 4) for t in REPORT_THRESHOLDS
        },
        "max_ball_conf_mean": round(float(confs.mean()), 4) if n else None,
        "max_ball_conf_max": round(float(confs.max()), 4) if n else None,
        "person_mean": round(float(np.mean([r["n_person"] for r in inside])), 2) if n else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--game-dir", type=Path, default=None,
                    help="shipped run dir; picks the densest det-source ball_track span as control")
    ap.add_argument("--frames", nargs=2, type=int, required=True, metavar=("LO", "HI"))
    ap.add_argument("--control", nargs=2, type=int, default=None, metavar=("LO", "HI"))
    ap.add_argument("--weights", action="append", required=True, metavar="LABEL=PATH|none")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    hole = (int(args.frames[0]), int(args.frames[1]))
    if args.control:
        control: tuple[int, int] | None = (int(args.control[0]), int(args.control[1]))
    elif args.game_dir:
        control = _densest_det_window(args.game_dir, hole[1] - hole[0])
    else:
        control = None
    spans = [hole] + ([control] if control else [])
    weights = _parse_weights(args.weights)

    t0 = time.monotonic()
    frames = _collect_frames(args.video, spans)
    decode_s = round(time.monotonic() - t0, 1)

    per_frame: list[dict] = []
    timings: dict[str, float] = {}
    for label, path in weights:
        t1 = time.monotonic()
        per_frame.extend(_run_weights(label, path, frames, args.batch))
        timings[label] = round(time.monotonic() - t1, 1)

    report = {
        "video": str(args.video),
        "hole": {"lo": hole[0], "hi": hole[1]},
        "control": {"lo": control[0], "hi": control[1]} if control else None,
        "probe_threshold": PROBE_THRESHOLD,
        "weights": {label: (str(p) if p else "stock-coco") for label, p in weights},
        "decode_s": decode_s,
        "detect_s": timings,
        "summary": {
            label: {
                "hole": _summary([r for r in per_frame if r["weights"] == label], *hole),
                "control": _summary([r for r in per_frame if r["weights"] == label], *control) if control else None,
            }
            for label, _ in weights
        },
        "per_frame": per_frame,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
