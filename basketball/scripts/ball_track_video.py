"""Ball-track video: yellow circle following the ball on real footage.

THE deliverable (2026-08-03): a video showing full ball tracking —
a yellow circle following the ball throughout. Two passes:

  A) decode (PyAV — AV1-safe), sample to ~30fps, run WASB on sliding
     3-frame triplets, keep confident heatmap peaks -> (t, x, y) track.
  B) decode again, draw the circle per frame — BRIGHT ring where the
     ball was detected, DIM ring where the position is interpolated
     across a short gap, nothing across long gaps (honest absence) —
     and encode h264 (render.py precedent: pts passthrough).

Scene + consistency gating (v3, Design decision 2026-08-03: the rim-only
gate failed at mid-court (no rim in frame during legitimate play).
Overlays now render when BOTH hold:
  - motion consistency: a peak belongs to a locally consistent ball
    track (>=MIN_NEIGHBORS peaks within +-NEIGH_S reachable at plausible
    ball speed) — cut-away false positives fail this discontinuity test;
  - scene signature: the frame looks like a court view from its person
    detections (>=SCENE_MIN_PERSONS boxes, median height <=
    SCENE_MAX_MEDIAN_H of frame) — kills static close-ups, works at
    mid-court where the rim doesn't. Rim presence is no longer a gate.

Usage (inside cvbench, GPU):
    python scripts/ball_track_video.py --video "<file>" \
        --game-dir /work/out-harvest/cal_fsu_v3real \
        --weights /work/weights/wasb/wasb_basketball_best.pth.tar \
        --out /work/models/ball-v1/ball_track_clip.mp4 \
        --start-s 600 --duration-s 180
Omit --start-s/--duration-s for the full game.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

SCORE_MIN = 0.4
GAP_MAX_S = 1.5
RIM_CONF_MIN = 0.3
RIM_NEAR_S = 0.4  # rim det within this of a frame (informational only)
# motion-consistency gate: a real ball track has reachable neighbors
NEIGH_S = 0.6
MAX_SPEED_PX_S = 1800.0  # full-court pass ~1100px/s at 720p; generous
MIN_NEIGHBORS = 3
# scene gate: court view = many smallish person boxes
SCENE_NEAR_MS = 200
SCENE_MIN_PERSONS = 5
SCENE_MAX_MEDIAN_H = 0.45  # of frame height
CLS_PERSON = 0  # DetClass space
SMOOTH_K = 5
RING_R = 16
DETECTED_W = 5
INTERP_W = 2
YELLOW = (255, 214, 0)
BATCH = 8
WASB_W, WASB_H = 512, 288


def load_rim_ok(game_dir: Path):
    """ts (s) -> bool: a confident rim detection sits within RIM_NEAR_S."""
    from montehall_cv.store.artifacts import read_stage

    det = read_stage(Path(game_dir) / "detections")
    ts = det.column("ts_ms").to_numpy()
    sel = (det.column("cls").to_numpy() == 3) & \
        (det.column("conf").to_numpy() >= RIM_CONF_MIN)
    rim_ms = np.sort(ts[sel])

    def rim_ok(t_s: float) -> bool:
        if rim_ms.size == 0:
            return False
        i = int(np.searchsorted(rim_ms, t_s * 1000))
        near = []
        if i > 0:
            near.append(abs(rim_ms[i - 1] - t_s * 1000))
        if i < rim_ms.size:
            near.append(abs(rim_ms[i] - t_s * 1000))
        return bool(near and min(near) <= RIM_NEAR_S * 1000)

    return rim_ok


def load_scene_ok(game_dir: Path, frame_h: float):
    """ts (s) -> bool: court-view signature from person detections —
    >=SCENE_MIN_PERSONS boxes within +-SCENE_NEAR_MS whose median height
    is <=SCENE_MAX_MEDIAN_H of the frame. Works at mid-court where the
    rim gate failed (2026-08-03 walk-back)."""
    from bisect import bisect_left

    from montehall_cv.store.artifacts import read_stage

    det = read_stage(Path(game_dir) / "detections")
    ts = det.column("ts_ms").to_numpy()
    sel = det.column("cls").to_numpy() == CLS_PERSON
    heights = (det.column("y2").to_numpy() - det.column("y1").to_numpy())
    p_ts = ts[sel]
    p_h = heights[sel]
    order = np.argsort(p_ts, kind="stable")
    p_ts, p_h = p_ts[order], p_h[order]

    def scene_ok(t_s: float) -> bool:
        lo = bisect_left(p_ts, t_s * 1000 - SCENE_NEAR_MS)
        hi = bisect_left(p_ts, t_s * 1000 + SCENE_NEAR_MS)
        if hi - lo < SCENE_MIN_PERSONS:
            return False
        return float(np.median(p_h[lo:hi])) <= SCENE_MAX_MEDIAN_H * frame_h

    return scene_ok


def consistent_peaks(peaks: list[tuple]) -> list[tuple]:
    """Keep peaks that belong to a locally consistent ball track: at
    least MIN_NEIGHBORS other peaks within +-NEIGH_S reachable at
    <=MAX_SPEED_PX_S. Cut-away false positives jump discontinuously at
    scene changes and die here."""
    if len(peaks) < MIN_NEIGHBORS + 1:
        return []
    t = np.array([p[0] for p in peaks])
    x = np.array([p[1] for p in peaks])
    y = np.array([p[2] for p in peaks])
    kept = []
    for i in range(t.size):
        sel = (np.abs(t - t[i]) <= NEIGH_S) & (np.abs(t - t[i]) > 1e-9)
        if not sel.any():
            continue
        d = np.hypot(x[sel] - x[i], y[sel] - y[i])
        dt = np.abs(t[sel] - t[i])
        if int((d / np.maximum(dt, 1e-3) <= MAX_SPEED_PX_S).sum()) \
                >= MIN_NEIGHBORS:
            kept.append(peaks[i])
    return kept


def build_track(peaks: list[tuple], out_times: list[float],
                view_ok=None) -> list:
    """Per output time: (x, y, 'det'|'interp') or None (long gap or
    gated non-court view)."""
    if not peaks:
        return [None] * len(out_times)
    pt = np.array([p[0] for p in peaks])
    px = np.array([p[1] for p in peaks])
    py = np.array([p[2] for p in peaks])
    if pt.size >= SMOOTH_K:  # median smoothing against single-frame jumps
        k = SMOOTH_K // 2
        px = np.array([np.median(px[max(0, i - k):i + k + 1])
                       for i in range(px.size)])
        py = np.array([np.median(py[max(0, i - k):i + k + 1])
                       for i in range(py.size)])
    out = []
    for t in out_times:
        if view_ok is not None and not view_ok(t):
            out.append(None)  # not a court view — never draw
            continue
        i = int(np.searchsorted(pt, t))
        exact = np.abs(pt - t).min() if pt.size else np.inf
        if exact <= 0.04:
            j = int(np.abs(pt - t).argmin())
            out.append((float(px[j]), float(py[j]), "det"))
            continue
        lo, hi = i - 1, i
        if lo < 0 or hi >= pt.size:
            out.append(None)
            continue
        if pt[hi] - pt[lo] > GAP_MAX_S:
            out.append(None)
            continue
        f = (t - pt[lo]) / max(pt[hi] - pt[lo], 1e-6)
        out.append((float(px[lo] + (px[hi] - px[lo]) * f),
                    float(py[lo] + (py[hi] - py[lo]) * f), "interp"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--game-dir", type=Path, default=None,
                    help="extract dir with rim detections (rim gate)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wasb-root", type=Path, default=Path("/work/wasb"))
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=0.0)
    args = ap.parse_args()

    import av
    import cv2
    import torch
    from PIL import Image, ImageDraw

    from wasb_shot_window_probe import (
        IMAGENET_MEAN, IMAGENET_STD, heatmap_of, load_wasb)

    model, dev, model_cfg = load_wasb(args.wasb_root, args.weights)
    wh = (model_cfg["inp_width"], model_cfg["inp_height"])
    started = time.monotonic()

    def frames(stride_only: bool):
        container = av.open(str(args.video))
        stream = container.streams.video[0]
        tb = stream.time_base
        fps = float(stream.average_rate or 30.0)
        stride = 2 if fps > 45 else 1
        if args.start_s > 0:
            container.seek(int(args.start_s / tb), stream=stream)
        end_s = args.start_s + args.duration_s if args.duration_s else None
        seen = 0
        for frame in container.decode(stream):
            stamp = frame.pts if frame.pts is not None else frame.dts
            if stamp is None:
                continue
            ts = float(stamp * tb)
            if ts < args.start_s - 0.02:
                continue
            if end_s and ts > end_s:
                break
            if seen % stride == 0:
                yield ts, frame
            seen += 1
        container.close()

    # ---- pass A: WASB peaks --------------------------------------------
    peaks: list[tuple] = []
    buf: list[tuple] = []  # (t, model_input)
    pend: list[tuple] = []  # (mid_t, triplet)
    native_scale = None

    def flush():
        if not pend:
            return
        batch = torch.from_numpy(
            np.stack([p[1] for p in pend])).to(dev)
        with torch.no_grad():
            preds = model(batch)
        for k in range(batch.shape[0]):
            hm = heatmap_of(
                preds[k:k + 1] if not isinstance(preds, dict)
                else {s: v[k:k + 1] for s, v in preds.items()})
            mid = hm[min(1, hm.shape[0] - 1)]
            score = float(mid.max())
            if score >= SCORE_MIN:
                yx = np.unravel_index(int(mid.argmax()), mid.shape)
                peaks.append((pend[k][0], float(yx[1]) * native_scale,
                              float(yx[0]) * native_scale, score))
        pend.clear()

    n_sampled = 0
    native_h = None
    out_times: list[float] = []
    for ts, frame in frames(stride_only=True):
        if native_scale is None:
            native_scale = frame.width / WASB_W
            native_h = frame.height
        out_times.append(ts)
        img = cv2.resize(frame.to_ndarray(format="rgb24"), wh)
        img = ((img.astype(np.float32) / 255 - IMAGENET_MEAN)
               / IMAGENET_STD).transpose(2, 0, 1)
        buf.append((ts, img))
        if len(buf) > 3:
            buf.pop(0)
        if len(buf) == 3:
            pend.append((buf[1][0],
                         np.concatenate([b[1] for b in buf], axis=0)))
            if len(pend) >= BATCH:
                flush()
        n_sampled += 1
        if n_sampled % 2000 == 0:
            print(f"BALLTRACK-A: {n_sampled} frames, {len(peaks)} peaks, "
                  f"{time.monotonic() - started:.0f}s", flush=True)
    flush()
    print(f"BALLTRACK-A-DONE frames={n_sampled} peaks={len(peaks)} "
          f"({len(peaks) / max(n_sampled, 1):.1%} of sampled)", flush=True)

    # ---- pass B: draw + encode (streamed — never buffer full-res) ------
    n_raw = len(peaks)
    peaks = consistent_peaks(peaks)
    scene_ok = (load_scene_ok(args.game_dir, native_h)
                if args.game_dir and native_h else None)
    print(f"BALLTRACK-GATES: peaks {n_raw} -> {len(peaks)} after "
          f"consistency; scene gate {'on' if scene_ok else 'off'}",
          flush=True)
    track = build_track(peaks, out_times, scene_ok)
    t0 = out_times[0] if out_times else 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".mp4.tmp")
    n_out = n_det = n_interp = 0
    with av.open(str(tmp), "w", format="mp4") as out_container:
        ostream = out_container.add_stream(
            "libx264", options={"crf": "23", "preset": "veryfast"})
        ostream.pix_fmt = "yuv420p"
        ostream.codec_context.time_base = Fraction(1, 1000)
        for i, (ts, frame) in enumerate(frames(stride_only=True)):
            if ostream.width == 0:
                ostream.width, ostream.height = frame.width, frame.height
            img = Image.fromarray(frame.to_ndarray(format="rgb24"))
            pos = track[i] if i < len(track) else None
            if pos is not None:
                x, y, kind = pos
                draw = ImageDraw.Draw(img)
                w = DETECTED_W if kind == "det" else INTERP_W
                draw.ellipse([x - RING_R, y - RING_R,
                              x + RING_R, y + RING_R],
                             outline=YELLOW, width=w)
                n_det += kind == "det"
                n_interp += kind == "interp"
            vf = av.VideoFrame.from_ndarray(np.asarray(img), format="rgb24")
            vf.pts = int((ts - t0) * 1000)
            out_container.mux(ostream.encode(vf))
            n_out += 1
            if n_out % 2000 == 0:
                print(f"BALLTRACK-B: {n_out} frames encoded", flush=True)
        out_container.mux(ostream.encode())
    tmp.rename(args.out)
    print("BALLTRACK_DONE " + json.dumps({
        "out": str(args.out), "frames": n_out,
        "ring_detected": n_det, "ring_interp": n_interp,
        "ring_coverage": round((n_det + n_interp) / max(n_out, 1), 4),
        "bytes": args.out.stat().st_size,
        "wall_s": round(time.monotonic() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
