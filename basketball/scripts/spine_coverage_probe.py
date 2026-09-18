"""Spine event-coverage probe — why does a covered ball track yield no events?

Spine v3 (2026-08-25) closed the 114-131s track hole at TRACK level
(0 -> 307 confirmed rows, one segment via multi-peak WASB) yet NO
holds/releases/rim_arrivals derive in that span, so truth ev13/14 stay
uncovered and e2e sits at 1/6. This probe measures every event-formation
precondition per window, against a control window where events DID form,
so the failing condition is observed rather than guessed:

  track supply     confirmed-row density, sources, interp gaps, segments
  holder supply    non-coasted quark bodies per frame; nearest-body
                   distance from each confirmed ball row (HOLD_DIST_PX
                   is the gate holder_at applies)
  run formation    derive_events run on the window slice with the real
                   bodies/rims maps — contact runs formed, lengths,
                   possessions (>= HOLD_MIN_FRAMES)
  rim supply       frames with a rim detection (conf >= 0.3); confirmed
                   ball rows passing the at_rim test; nearest rim pass
                   around each truth ts inside the window

Read-only over existing stage artifacts; no inference.

Usage (inside cvbench):
    python scripts/spine_coverage_probe.py /work/out-harvest/hs_clip_20260824 \
        --window 110 135 --truth /work/hs_clip_truth_20260824.json \
        --out /work/models/quark-v1/spine_coverage_probe_hsclip.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from montehall_cv.pipeline.ball_track import (
    CLS_RIM,
    HOLD_DIST_PX,
    HOLD_MIN_FRAMES,
    RIM_ARRIVE_SCALE,
    derive_events,
)
from montehall_cv.store.artifacts import read_stage


def load_track(job_dir: Path) -> list[dict]:
    rows = read_stage(job_dir / "ball_track").to_pylist()
    rows.sort(key=lambda r: r["frame_idx"])
    return rows


def load_bodies(job_dir: Path) -> dict[int, list]:
    b = read_stage(job_dir / "quark_binding")
    cols = {c: b.column(c).to_numpy(zero_copy_only=False)
            for c in ("frame_idx", "track_id", "x1", "y1", "x2", "y2")}
    keep = np.ones(len(cols["frame_idx"]), dtype=bool)
    if "coasted" in b.column_names:
        keep = ~b.column("coasted").to_numpy(zero_copy_only=False)
    out: dict[int, list] = defaultdict(list)
    for i in np.nonzero(keep)[0]:
        out[int(cols["frame_idx"][i])].append(
            (int(cols["track_id"][i]),
             (cols["x1"][i], cols["y1"][i], cols["x2"][i], cols["y2"][i])))
    return out


def load_rims(job_dir: Path) -> dict[int, list]:
    t = read_stage(job_dir / "detections")
    out: dict[int, list] = defaultdict(list)
    for row in t.to_pylist():
        if row["cls"] == CLS_RIM and (row["conf"] or 0) >= 0.3:
            out[int(row["frame_idx"])].append(
                (row["x1"], row["y1"], row["x2"], row["y2"]))
    return out


def nearest_body_px(row: dict, bodies: dict[int, list]) -> float | None:
    best = None
    for _tid, (x1, y1, x2, y2) in bodies.get(row["frame_idx"], ()):
        dx = max(x1 - row["x"], 0.0, row["x"] - x2)
        dy = max(y1 - row["y"], 0.0, row["y"] - y2)
        d = float(np.hypot(dx, dy))
        if best is None or d < best:
            best = d
    return best


def nearest_rim_norm(row: dict, rims: dict[int, list]) -> float | None:
    """Distance to rim center in rim-box widths; at_rim passes <= RIM_ARRIVE_SCALE."""
    best = None
    for x1, y1, x2, y2 in rims.get(row["frame_idx"], ()):
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        d = max(abs(row["x"] - cx) / max(w, 1e-6),
                abs(row["y"] - cy) / max(h, 1e-6))
        if best is None or d < best:
            best = d
    return best


def measure_window(track: list[dict], events: list[dict],
                   bodies: dict[int, list], rims: dict[int, list],
                   t0_ms: int, t1_ms: int,
                   truth_ts: list[int]) -> dict:
    win = [r for r in track if t0_ms <= r["ts_ms"] <= t1_ms]
    conf = [r for r in win if r["source"] != "interp"]
    frames = [r["frame_idx"] for r in win]
    span = (frames[-1] - frames[0] + 1) if frames else 0

    longest_interp = cur = 0
    for r in win:
        cur = cur + 1 if r["source"] == "interp" else 0
        longest_interp = max(longest_interp, cur)

    near_body = [nearest_body_px(r, bodies) for r in conf]
    have_body = [d for d in near_body if d is not None]
    frames_in_win = list(range(frames[0], frames[-1] + 1)) if frames else []
    body_frames = sum(1 for f in frames_in_win if bodies.get(f))
    rim_frames = sum(1 for f in frames_in_win if rims.get(f))

    win_events = derive_events(win, bodies, rims)
    runs = [e for e in win_events if e["kind"] in ("contact", "contact_graze")]

    stored = defaultdict(int)
    for e in events:
        if t0_ms <= e["ts_ms"] <= t1_ms:
            stored[e["kind"]] += 1

    rim_near = [nearest_rim_norm(r, rims) for r in conf]
    rim_near = [d for d in rim_near if d is not None]
    truth_rim = {}
    for ts in truth_ts:
        if t0_ms <= ts <= t1_ms:
            around = [nearest_rim_norm(r, rims) for r in conf
                      if abs(r["ts_ms"] - ts) <= 2000]
            around = [d for d in around if d is not None]
            truth_rim[ts] = round(min(around), 2) if around else None

    return {
        "window_ms": [t0_ms, t1_ms],
        "track": {
            "rows": len(win), "confirmed": len(conf),
            "by_source": {s: sum(1 for r in win if r["source"] == s)
                          for s in ("det", "wasb", "both", "interp")},
            "segments": sorted({r["seg_id"] for r in win}),
            "span_coverage": round(len(conf) / max(span, 1), 3),
            "longest_interp_run": longest_interp,
        },
        "holder_supply": {
            "frames_with_any_body": body_frames,
            "frames_total": len(frames_in_win),
            "confirmed_rows_with_body_in_frame": len(have_body),
            "rows_within_hold_gate": sum(1 for d in have_body
                                         if d <= HOLD_DIST_PX),
            "rows_within_2x_gate": sum(1 for d in have_body
                                       if d <= 2 * HOLD_DIST_PX),
            "nearest_body_px": {
                "median": round(float(np.median(have_body)), 1)
                if have_body else None,
                "min": round(min(have_body), 1) if have_body else None,
            },
        },
        "run_formation": {
            "contact_runs": len(runs),
            "run_frame_counts": sorted(
                (e.get("n_track_frames") or 0 for e in runs), reverse=True)[:10],
            "possessions_formed": sum(
                1 for e in win_events if e["kind"] == "hold"),
            "hold_min_frames": HOLD_MIN_FRAMES,
        },
        "rim_supply": {
            "frames_with_rim_det": rim_frames,
            "confirmed_rows_at_rim": sum(
                1 for d in rim_near if d <= RIM_ARRIVE_SCALE),
            "min_rim_dist_boxwidths": round(min(rim_near), 2)
            if rim_near else None,
            "near_each_truth_ts": truth_rim,
        },
        "events_in_stage_output": dict(stored),
        "events_rederived_on_window": {
            k: sum(1 for e in win_events if e["kind"] == k)
            for k in ("hold", "release", "rim_arrival", "shot",
                      "contact", "contact_graze")},
    }


def auto_control(events: list[dict], t0_ms: int, t1_ms: int,
                 width_ms: int) -> tuple[int, int]:
    """Densest-hold window of the same width outside the probe window."""
    holds = [e["ts_ms"] for e in events if e["kind"] == "hold"
             and not (t0_ms <= e["ts_ms"] <= t1_ms)]
    if not holds:
        return 0, width_ms
    best = max(holds, key=lambda t: sum(
        1 for h in holds if t <= h <= t + width_ms))
    return best, best + width_ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game_dir", type=Path)
    ap.add_argument("--window", nargs=2, type=float, default=[110.0, 135.0],
                    help="probe window, seconds")
    ap.add_argument("--control", nargs=2, type=float, default=None,
                    help="control window, seconds (default: densest-hold "
                         "window of equal width elsewhere in the clip)")
    ap.add_argument("--truth", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    track = load_track(args.game_dir)
    events = read_stage(args.game_dir / "ball_events").to_pylist()
    bodies = load_bodies(args.game_dir)
    rims = load_rims(args.game_dir)
    truth_ts = []
    if args.truth:
        truth_ts = [int(e["ts_ms"]) for e in
                    json.loads(args.truth.read_text())["timed_events"]]

    t0, t1 = int(args.window[0] * 1000), int(args.window[1] * 1000)
    if args.control:
        c0, c1 = int(args.control[0] * 1000), int(args.control[1] * 1000)
    else:
        c0, c1 = auto_control(events, t0, t1, t1 - t0)

    report = {
        "game_dir": str(args.game_dir),
        "probe": measure_window(track, events, bodies, rims, t0, t1, truth_ts),
        "control": measure_window(track, events, bodies, rims, c0, c1,
                                  truth_ts),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
