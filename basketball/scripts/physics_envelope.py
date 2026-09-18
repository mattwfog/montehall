"""Physics envelope calibration: measured motion bounds for the quark
physics layer (design decision 08-05: an incorrect identity switch is
physically impossible motion, so physics can veto it).

The gates must come from DATA, not invented constants: this probe reads
a quark_binding stage, keeps only CLEAN segments (no contested /
disputed / coasted rows, >= MIN_CLEAN_S), estimates per-frame CAMERA
motion as the median displacement of all concurrently-observed bodies
(a pan makes everyone "teleport" — pixel physics without ego-motion
compensation gates nothing), and reports the percentile envelope of
ego-compensated speed and acceleration in body-heights/second. The
p99.9 of clean human motion becomes the engine's physics cap; anything
past it is a switch, not a sprint.

Usage (inside cvbench):
    python scripts/physics_envelope.py \
        /work/models/quark-v1/cal_full/quark_binding \
        /work/models/quark-v1/cd_full/quark_binding \
        --out /work/models/quark-v1/physics_envelope.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from montehall_cv.store.artifacts import read_stage

MIN_CLEAN_S = 3.0
SMOOTH_MS = 100.0  # velocity estimated over this horizon, not frame-to-
                   # frame (60fps jitter at 17ms baselines would swamp
                   # real motion)


def segment_rows(stage_dir: Path):
    t = read_stage(stage_dir)
    cols = {c: t.column(c).to_numpy() for c in
            ("ts_ms", "track_id", "x1", "y1", "x2", "y2",
             "contested", "coasted", "disputed")}
    by_track: dict[int, list[int]] = defaultdict(list)
    for i, tid in enumerate(cols["track_id"].tolist()):
        by_track[tid].append(i)
    return cols, by_track


def camera_motion(cols) -> dict[int, tuple[float, float]]:
    """ts_ms -> median (dx, dy) of bodies observed at both this instant
    and the previous one, per ms — the ego-motion estimate."""
    ts_sorted = np.unique(cols["ts_ms"])
    pos_at: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for i in range(len(cols["ts_ms"])):
        cx = (cols["x1"][i] + cols["x2"][i]) / 2
        cy = (cols["y1"][i] + cols["y2"][i]) / 2
        pos_at[int(cols["ts_ms"][i])][int(cols["track_id"][i])] = (cx, cy)
    cam: dict[int, tuple[float, float]] = {}
    for k in range(1, len(ts_sorted)):
        t0, t1 = int(ts_sorted[k - 1]), int(ts_sorted[k])
        dt = t1 - t0
        if dt <= 0 or dt > 100:
            cam[t1] = (0.0, 0.0)
            continue
        moves = [(pos_at[t1][tid][0] - pos_at[t0][tid][0],
                  pos_at[t1][tid][1] - pos_at[t0][tid][1])
                 for tid in pos_at[t1].keys() & pos_at[t0].keys()]
        if len(moves) >= 4:
            arr = np.array(moves) / dt
            cam[t1] = (float(np.median(arr[:, 0])),
                       float(np.median(arr[:, 1])))
        else:
            cam[t1] = (0.0, 0.0)
    return cam


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stages", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    speeds, accels = [], []
    n_clean = n_segments = 0
    for stage in args.stages:
        cols, by_track = segment_rows(stage)
        cam = camera_motion(cols)
        for _tid, idxs in by_track.items():
            n_segments += 1
            flagged = any(cols["contested"][i] or cols["coasted"][i]
                          or cols["disputed"][i] for i in idxs)
            ts = np.array([cols["ts_ms"][i] for i in idxs], dtype=np.float64)
            if flagged or ts[-1] - ts[0] < MIN_CLEAN_S * 1000:
                continue
            n_clean += 1
            cx = np.array([(cols["x1"][i] + cols["x2"][i]) / 2 for i in idxs])
            cy = np.array([(cols["y1"][i] + cols["y2"][i]) / 2 for i in idxs])
            h = float(np.median(
                [cols["y2"][i] - cols["y1"][i] for i in idxs]))
            # subtract integrated camera motion, then sample velocity on
            # the SMOOTH_MS horizon
            cam_x = np.zeros(len(idxs))
            cam_y = np.zeros(len(idxs))
            for k in range(1, len(idxs)):
                mx, my = cam.get(int(ts[k]), (0.0, 0.0))
                dt = ts[k] - ts[k - 1]
                cam_x[k] = cam_x[k - 1] + mx * dt
                cam_y[k] = cam_y[k - 1] + my * dt
            ex, ey = cx - cam_x, cy - cam_y
            step = max(1, int(round(SMOOTH_MS / max(
                np.median(np.diff(ts)), 1.0))))
            vx = (ex[step:] - ex[:-step]) / (ts[step:] - ts[:-step])
            vy = (ey[step:] - ey[:-step]) / (ts[step:] - ts[:-step])
            v = np.hypot(vx, vy) * 1000 / h  # body-heights per second
            speeds.extend(v.tolist())
            vt = ts[: len(v)]
            if len(v) > step:
                a = (v[step:] - v[:-step]) / ((vt[step:] - vt[:-step]) / 1000)
                accels.extend(np.abs(a).tolist())

    def pct(arr):
        if not arr:
            return None
        a = np.array(arr)
        return {p: round(float(np.percentile(a, p)), 3)
                for p in (50, 90, 99, 99.9)} | {"max": round(float(a.max()), 3)}

    out = {
        "stages": [str(s) for s in args.stages],
        "segments_total": n_segments,
        "segments_clean": n_clean,
        "speed_h_per_s": pct(speeds),
        "accel_h_per_s2": pct(accels),
        "n_speed_samples": len(speeds),
        "n_accel_samples": len(accels),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print("PHYSICS_ENVELOPE_DONE " + json.dumps(out))


if __name__ == "__main__":
    main()
