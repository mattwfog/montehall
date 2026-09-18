"""THE BALL SPINE: one continuous per-game ball track + possession events.

Ball-first design (2026-08-03): estimate the ball's movement first — know
where it is even when invisible — and let player reasoning ride on it.
Built 2026-08-25 after the first e2e run showed every downstream step
re-deriving the ball badly and locally (per-window WASB refits at 2/15
coverage, ball-proximity rankers pointing at rebounders).

Two artifacts:

  ball_track/   one row per frame the spine covers: fused position from
                the frame detector's BALL class + a full-clip WASB pass
                (motion-based; sees the ball in flight where the frame
                detector goes blind), velocity-gated with bounded coasting.
                source: det | wasb | both | interp.

  ball_events/  what falls out of the track: HOLD segments (ball sits on
                a quark body), RELEASE (hold -> flight transition), and
                RIM_ARRIVAL (flight ends at a detector rim box). A
                release paired with a rim arrival IS a shot attempt with
                the shooter attached — selection stops being a ranking
                problem wherever the spine covers the shot.

Pixel space throughout: the court homography is dead in 84-92% of shot
windows (2026-08-03 probe) — the spine must not depend on it.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa

from montehall_cv.pipeline.flight import (
    CLS_RIM,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PEAK_MIN,
    WASB_W,
    load_wasb,
    heatmap_of,
)
from montehall_cv.store.artifacts import (
    ArtifactWriter,
    read_stage,
    stage_complete,
)
from montehall_cv.store.records import DetClass

BALL_TRACK_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("frame_idx", pa.int32()),
        pa.field("ts_ms", pa.int64()),
        pa.field("x", pa.float32()),
        pa.field("y", pa.float32()),
        pa.field("conf", pa.float32()),
        pa.field("source", pa.string()),  # det | wasb | both | interp
        pa.field("seg_id", pa.int32()),   # contiguous confirmed segment
    ]
)

BALL_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("kind", pa.string()),  # hold | release | rim_arrival | shot
        pa.field("ts_ms", pa.int64()),
        pa.field("ts_end_ms", pa.int64(), nullable=True),   # hold/shot spans
        pa.field("track_id", pa.int64(), nullable=True),    # quark holder
        pa.field("x", pa.float32(), nullable=True),
        pa.field("y", pa.float32(), nullable=True),
        pa.field("release_ts_ms", pa.int64(), nullable=True),  # shot rows
        pa.field("n_track_frames", pa.int32(), nullable=True),
        pa.field("made_geom", pa.bool_(), nullable=True),  # shot/putback:
        # trajectory-through-rim verdict (None = track can't judge)
    ]
)

# fusion
GATE_BASE_PX = 60.0        # match gate at gap=1 frame ...
GATE_GROW_PX = 40.0        # ... growing per coasted frame (ball is fast)
COAST_MAX_FRAMES = 12      # ~0.4s at 30fps: beyond this the segment ends
DET_CONF_MIN = 0.3
WASB_AGREE_PX = 40.0       # det+wasb within this = "both" (max confidence)
MIN_SEG_FRAMES = 4         # confirmed segments shorter than this are noise
WASB_LOW_MIN = 0.08        # track-before-detect: sub-PEAK_MIN peaks are
                           # usable ONLY on an existing track's prediction
                           # (never seed a segment) — continuity v2
WASB_TOPK = 3              # peaks per frame (spine v3): argmax-only lost
                           # the ball whenever a distractor out-scored it
                           # — the 114-131s track hole's supply side
WASB_PEAK_SEP_HM = 6       # heatmap px between extracted peaks
STITCH_MAX_FRAMES = 45     # ~1.5s: segments this close are bridge
                           # candidates (matches controls.py's court-space
                           # interp bound)
STITCH_MAX_SPEED = 90.0    # px/frame implied by the bridge; faster = a
                           # cut or a different ball story, no stitch
GAP_MIN_FRAMES = 30        # ~1s: uncovered spans this long get the
                           # tiled (2x-resolution) WASB second pass —
                           # the ev13 class: full-frame WASB squeezes
                           # 1920->512 and a distant ball vanishes
                           # (zoom probe 2026-08-25: 8/121 gap frames
                           # with any full-frame peak vs 70/121 tiled,
                           # 0 vs 22 at >=.5, tile peak 2px from an
                           # independent det row)
TILE_OVERLAP = 0.25        # fraction of tile size shared with neighbour

# events
HOLD_DIST_PX = 30.0        # ball center within box (+pad) = candidate hold
HOLD_MIN_FRAMES = 5        # ~0.17s persistent contact = a POSSESSION
                           # (contact runs of ANY length still bind shots)
HOLD_BREAK_FRAMES = 6      # this many consecutive off-body frames ends it
HOLD_MERGE_GAP_MS = 400    # same-body runs this close = one possession
                           # (dribbles; v1 spawned fake putbacks from them)
RIM_ARRIVE_SCALE = 2.0     # arrival = ball inside rim box scaled by this
RELEASE_RIM_MAX_MS = 4000  # a release pairs with an arrival this far out
MIN_FLIGHT_MS = 250        # release->arrival faster than this is a putback
                           # tap, not a shot we can attribute
PUTBACK_MERGE_MS = 2500    # shots arriving inside one rim scramble: first
                           # release is THE attempt, the rest are putbacks
                           # (same window run_shots uses to merge bursts)
RIM_PASS_WINDOW_MS = (-200, 1000)  # rows examined around a rim arrival
RIM_PASS_X_FRAC = 0.6      # |ball x - rim cx| within this fraction of rim
                           # width counts as "over the mouth"
RIM_PASS_MIN_ROWS = 3      # fewer rows near the rim = no geometric verdict
GRAZE_UP_VY = 10.0         # px/frame the ball must already be RISING when
                           # a sub-possession contact begins for it to be a
                           # graze (defender closeout on a launched ball,
                           # ev-9 class) — a catch arrives flat or falling;
                           # a dribble bounce rises slower than a launch
GRAZE_VEL_FRAMES = 4       # confirmed rows the incoming velocity spans


# ---------------------------------------------------------------- fusion --

def _det_balls(job_dir: Path) -> dict[int, list[tuple[float, float, float]]]:
    t = read_stage(job_dir / "detections")
    cls = t.column("cls").to_numpy(zero_copy_only=False)
    conf = t.column("conf").to_numpy(zero_copy_only=False)
    sel = (cls == int(DetClass.BALL)) & (conf >= DET_CONF_MIN)
    fi = t.column("frame_idx").to_numpy(zero_copy_only=False)[sel]
    cx = (t.column("x1").to_numpy(zero_copy_only=False)[sel]
          + t.column("x2").to_numpy(zero_copy_only=False)[sel]) / 2
    cy = (t.column("y1").to_numpy(zero_copy_only=False)[sel]
          + t.column("y2").to_numpy(zero_copy_only=False)[sel]) / 2
    cf = conf[sel]
    out: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for f, x, y, c in zip(fi, cx, cy, cf):
        out[int(f)].append((float(x), float(y), float(c)))
    return out


def _topk_peaks(hm: np.ndarray, native_scale: float,
                k: int = WASB_TOPK) -> list[tuple[float, float, float]]:
    """Up to k local maxima >= WASB_LOW_MIN, suppressed within
    WASB_PEAK_SEP_HM. The ball is often the SECOND peak (a bright head,
    an LED segment, a sneaker takes argmax) — one peak per frame starved
    fusion exactly where continuity mattered."""
    work = hm.copy()
    out: list[tuple[float, float, float]] = []
    for _ in range(k):
        score = float(work.max())
        if score < WASB_LOW_MIN:
            break
        yx = np.unravel_index(int(work.argmax()), work.shape)
        out.append((float(yx[1]) * native_scale,
                    float(yx[0]) * native_scale, score))
        y0 = max(int(yx[0]) - WASB_PEAK_SEP_HM, 0)
        x0 = max(int(yx[1]) - WASB_PEAK_SEP_HM, 0)
        work[y0:int(yx[0]) + WASB_PEAK_SEP_HM + 1,
             x0:int(yx[1]) + WASB_PEAK_SEP_HM + 1] = 0.0
    return out


def wasb_full_pass(video: Path, wasb_root: Path, weights: Path,
                   log=None) -> dict[int, list[tuple[float, float, float]]]:
    """frame_idx -> [(x, y, score)]: top-K WASB peaks per frame, whole clip.
    Sequential decode (PyAV — AV1 trap), triplets over consecutive frames.
    """
    import av
    import cv2
    import torch

    model, dev, model_cfg = load_wasb(wasb_root, weights)
    wh = (model_cfg["inp_width"], model_cfg["inp_height"])

    container = av.open(str(video))
    stream = container.streams.video[0]
    native_scale = stream.width / WASB_W

    peaks: dict[int, list[tuple[float, float, float]]] = {}
    buf: list[np.ndarray] = []
    buf_idx: list[int] = []
    batch_trip: list[np.ndarray] = []
    batch_mid: list[int] = []

    def flush() -> None:
        if not batch_trip:
            return
        arr = torch.from_numpy(np.stack(batch_trip)).to(dev)
        with torch.no_grad():
            preds = model(arr)
        for k in range(arr.shape[0]):
            hm = heatmap_of(
                preds[k:k + 1] if not isinstance(preds, dict)
                else {s: v[k:k + 1] for s, v in preds.items()})
            mid = hm[min(1, hm.shape[0] - 1)]
            found = _topk_peaks(mid, native_scale)
            if found:
                peaks[batch_mid[k]] = found
        batch_trip.clear()
        batch_mid.clear()

    fps = float(stream.average_rate or 30.0)
    n_done = 0
    for frame in container.decode(stream):
        if frame.time is None:
            continue
        idx = round(frame.time * fps)
        img = cv2.resize(frame.to_ndarray(format="rgb24"), wh)
        img = ((img.astype(np.float32) / 255 - IMAGENET_MEAN)
               / IMAGENET_STD).transpose(2, 0, 1)
        buf.append(img)
        buf_idx.append(idx)
        if len(buf) == 3:
            batch_trip.append(np.concatenate(buf, axis=0))
            batch_mid.append(buf_idx[1])
            buf.pop(0)
            buf_idx.pop(0)
            if len(batch_trip) == 16:
                flush()
        n_done += 1
        if log and n_done % 3000 == 0:
            log(f"wasb full pass: {n_done} frames, {len(peaks)} peaks")
    flush()
    container.close()
    return peaks


def uncovered_spans(track: list[dict], lo_frame: int,
                    hi_frame: int) -> list[tuple[int, int]]:
    """Frame spans >= GAP_MIN_FRAMES with no confirmed track row."""
    confirmed = sorted(r["frame_idx"] for r in track
                       if r["source"] != "interp")
    spans: list[tuple[int, int]] = []
    prev = lo_frame - 1
    for f in confirmed + [hi_frame + 1]:
        if f - prev - 1 >= GAP_MIN_FRAMES:
            spans.append((prev + 1, f - 1))
        prev = f
    return spans


def tiled_wasb_pass(video: Path, wasb_root: Path, weights: Path,
                    frames_needed: set[int],
                    log=None) -> dict[int, list[tuple[float, float, float]]]:
    """2x-effective-resolution WASB over selected frames only: a 2x2
    tile grid (TILE_OVERLAP shared), each tile resized to model input.
    Sequential decode, inference only where the fused track has holes."""
    import av
    import cv2
    import torch

    if not frames_needed:
        return {}
    model, dev, model_cfg = load_wasb(wasb_root, weights)
    wh = (model_cfg["inp_width"], model_cfg["inp_height"])

    container = av.open(str(video))
    stream = container.streams.video[0]
    fps = float(stream.average_rate or 30.0)
    hi_needed = max(frames_needed)

    regions: list[tuple[int, int, int, int]] | None = None
    peaks: dict[int, list[tuple[float, float, float]]] = {}
    buf: list[np.ndarray] = []
    buf_idx: list[int] = []
    n_done = 0
    for frame in container.decode(stream):
        if frame.time is None:
            continue
        idx = round(frame.time * fps)
        if idx > hi_needed + 1:
            break
        # keep a rolling raw buffer only near needed frames (a needed
        # mid m consumes frames m-1..m+1, each of which passes this test)
        if not any(f in frames_needed for f in (idx - 1, idx, idx + 1)):
            buf.clear()
            buf_idx.clear()
            continue
        rgb = frame.to_ndarray(format="rgb24")
        if regions is None:
            w, h = rgb.shape[1], rgb.shape[0]
            tw, th = int(w / (2 - TILE_OVERLAP)), int(h / (2 - TILE_OVERLAP))
            regions = [(x, y, x + tw, y + th)
                       for y in (0, h - th) for x in (0, w - tw)]
        buf.append(rgb)
        buf_idx.append(idx)
        if len(buf) < 3:
            continue
        mid = buf_idx[1]
        if mid in frames_needed:
            trips = []
            for x1, y1, x2, y2 in regions:
                tile = [cv2.resize(b[y1:y2, x1:x2], wh) for b in buf]
                tile = [((t.astype(np.float32) / 255 - IMAGENET_MEAN)
                         / IMAGENET_STD).transpose(2, 0, 1) for t in tile]
                trips.append(np.concatenate(tile, axis=0))
            arr = torch.from_numpy(np.stack(trips)).to(dev)
            with torch.no_grad():
                preds = model(arr)
            found: list[tuple[float, float, float]] = []
            for k, (x1, y1, x2, y2) in enumerate(regions):
                hm = heatmap_of(
                    preds[k:k + 1] if not isinstance(preds, dict)
                    else {s: v[k:k + 1] for s, v in preds.items()})
                mid_hm = hm[min(1, hm.shape[0] - 1)]
                sx = (x2 - x1) / mid_hm.shape[1]
                sy = (y2 - y1) / mid_hm.shape[0]
                for px, py, ps in _topk_peaks(mid_hm, 1.0):
                    found.append((x1 + px * sx, y1 + py * sy, ps))
            if found:
                peaks[mid] = found
            n_done += 1
            if log and n_done % 200 == 0:
                log(f"tiled wasb pass: {n_done} frames, {len(peaks)} hit")
        buf.pop(0)
        buf_idx.pop(0)
    container.close()
    return peaks


def fuse_track(det_by_frame: dict, wasb_by_frame: dict,
               ts_of: dict[int, int]) -> list[dict]:
    """Velocity-gated forward fusion with bounded coasting.

    Confirmed rows come from a det or wasb candidate inside the gate;
    coasted rows carry the prediction (source=interp) and only survive
    if the segment re-confirms within COAST_MAX_FRAMES.
    """
    frames = sorted(set(det_by_frame) | set(wasb_by_frame))
    if not frames:
        return []
    rows: list[dict] = []
    seg: list[dict] = []
    seg_id = 0
    pos = vel = None
    last_frame = None
    coasting: list[dict] = []

    def close_segment() -> None:
        nonlocal seg, seg_id, pos, vel, coasting
        if len([r for r in seg if r["source"] != "interp"]) >= MIN_SEG_FRAMES:
            rows.extend(seg)
            seg_id += 1
        seg = []
        coasting = []
        pos = vel = None

    for f in range(frames[0], frames[-1] + 1):
        cands = []
        low_cands = []
        for x, y, c in det_by_frame.get(f, ()):
            cands.append((x, y, c, "det"))
        wasb_here = wasb_by_frame.get(f)
        if wasb_here is not None:
            # v3: top-K peaks per frame; a bare tuple (v2 caches, tests)
            # still reads as a single peak
            if isinstance(wasb_here, tuple):
                wasb_here = [wasb_here]
            for wx, wy, ws in wasb_here:
                if ws < PEAK_MIN:
                    # track-before-detect: too weak to seed or free-match,
                    # but a live track's prediction can claim it
                    low_cands.append((wx, wy, ws, "wasb"))
                    continue
                merged = False
                for i, (x, y, c, s) in enumerate(cands):
                    if np.hypot(x - wx, y - wy) <= WASB_AGREE_PX:
                        cands[i] = ((x + wx) / 2, (y + wy) / 2,
                                    max(c, ws), "both")
                        merged = True
                        break
                if not merged:
                    cands.append((wx, wy, ws, "wasb"))
        if pos is None:
            if cands:
                x, y, c, s = max(cands, key=lambda t: t[2])
                pos, vel = np.array([x, y]), np.array([0.0, 0.0])
                last_frame = f
                seg.append({"frame_idx": f, "x": x, "y": y, "conf": c,
                            "source": s, "seg_id": seg_id})
            continue
        gap = f - last_frame
        if gap > COAST_MAX_FRAMES:
            close_segment()
            continue
        pred = pos + vel * gap
        gate = GATE_BASE_PX + GATE_GROW_PX * (gap - 1)
        best = None
        for x, y, c, s in cands:
            d = float(np.hypot(x - pred[0], y - pred[1]))
            if d <= gate and (best is None or d < best[0]):
                best = (d, x, y, c, s)
        if best is None:
            # continuity v2: a sub-threshold peak on the prediction keeps
            # the track alive through flight blur (tighter gate — weak
            # evidence must sit where physics says the ball is)
            for x, y, c, s in low_cands:
                d = float(np.hypot(x - pred[0], y - pred[1]))
                if d <= gate * 0.6 and (best is None or d < best[0]):
                    best = (d, x, y, c, s)
        if best is None:
            coasting.append({"frame_idx": f, "x": float(pred[0]),
                             "y": float(pred[1]), "conf": 0.0,
                             "source": "interp", "seg_id": seg_id})
            continue
        _d, x, y, c, s = best
        new = np.array([x, y])
        vel = (new - pos) / gap
        pos = new
        last_frame = f
        seg.extend(coasting)
        coasting = []
        seg.append({"frame_idx": f, "x": x, "y": y, "conf": c,
                    "source": s, "seg_id": seg_id})
    close_segment()
    rows = _stitch_segments(rows)
    for r in rows:
        r["ts_ms"] = ts_of.get(r["frame_idx"],
                               int(r["frame_idx"] * 1000 / 30))
    return rows


def _stitch_segments(rows: list[dict]) -> list[dict]:
    """Continuity v2: bridge nearby segments under a physics bound.

    "Know where it is even when invisible" (ball-first ruling): two
    confirmed segments separated by <= STITCH_MAX_FRAMES whose bridge
    implies a plausible ball speed become ONE segment with honest
    interp rows across the gap. A hold that spans the gap survives —
    the missed-release class of the v1 spine."""
    if not rows:
        return rows
    segs: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        segs[r["seg_id"]].append(r)
    ordered = sorted(segs.values(), key=lambda s: s[0]["frame_idx"])
    out: list[list[dict]] = [ordered[0]]
    for seg in ordered[1:]:
        tail = out[-1][-1]
        head = seg[0]
        gap = head["frame_idx"] - tail["frame_idx"]
        dist = float(np.hypot(head["x"] - tail["x"], head["y"] - tail["y"]))
        if 0 < gap <= STITCH_MAX_FRAMES and dist / gap <= STITCH_MAX_SPEED:
            for i in range(1, gap):
                f = i / gap
                out[-1].append({
                    "frame_idx": tail["frame_idx"] + i,
                    "x": tail["x"] + (head["x"] - tail["x"]) * f,
                    "y": tail["y"] + (head["y"] - tail["y"]) * f,
                    "conf": 0.0, "source": "interp",
                    "seg_id": out[-1][0]["seg_id"],
                })
            for r in seg:
                r["seg_id"] = out[-1][0]["seg_id"]
            out[-1].extend(seg)
        else:
            out.append(seg)
    return [r for seg in out for r in seg]


# ---------------------------------------------------------------- events --

def rim_pass_verdict(rows: list[dict],
                     rim_box: tuple | None) -> bool | None:
    """Trajectory-through-rim made/miss (2026-08-25:
    the spine knows the ball's path AT the rim — a make passes DOWN
    THROUGH the mouth, a miss enters over it and never exits below).

    rows: ts-ordered track rows (any source — interp rows carry the
    physics prediction through net occlusion) around the arrival.
    True  = seen above the mouth, then below the rim, both in-x.
    False = seen above the mouth in-x but never below (bounce/short).
    None  = no rim box, too few rows, or never over the mouth (side
            approach under the RIM_ARRIVE_SCALE slop) — can't judge."""
    if rim_box is None or len(rows) < RIM_PASS_MIN_ROWS:
        return None
    x1, y1, x2, y2 = rim_box
    cx, w = (x1 + x2) / 2, max(x2 - x1, 1e-6)
    above = False
    for r in rows:
        in_x = abs(r["x"] - cx) <= RIM_PASS_X_FRAC * w
        if not in_x:
            continue
        if r["y"] < y1:
            above = True
        elif above and r["y"] > y2:
            return True
    return False if above else None


RIM_LOCAL_FRAMES = 45  # rim reference = detections within ±this window
RIM_LOCAL_K = 15       # ...keeping only the K nearest-in-time boxes
RIM_SPLIT_PX = 120     # x gap separating two rims inside one window


def static_rims(rims_by_frame: dict) -> list[tuple]:
    """Median rim box per court END over every detection in the clip.

    SUPERSEDED in derive_events by local_rims (2026-08-26 ruling: the
    product works on ANY footage — a whole-clip median only exists on a
    fixed camera; clip 2's panning view put the median in no-man's-land
    and arrivals went 39 -> 9). Retained for v10-era diagnostics
    (spine_residuals_probe reproduces the v10 artifacts)."""
    all_boxes = [b for boxes in rims_by_frame.values() for b in boxes]
    if not all_boxes:
        return []
    mid = float(np.median([(b[0] + b[2]) / 2 for b in all_boxes]))
    out = []
    for side in (lambda c: c <= mid, lambda c: c > mid):
        group = [b for b in all_boxes if side((b[0] + b[2]) / 2)]
        if group:
            arr = np.array(group, dtype=np.float64)
            out.append(tuple(np.median(arr, axis=0)))
    return out


def local_rims(rim_frames: list[int], rim_boxes: list[tuple],
               frame_idx: int) -> list[tuple]:
    """Median rim box per x-cluster over detections within
    ±RIM_LOCAL_FRAMES of frame_idx — the footage-agnostic rim reference.

    Keeps v9's jitter kill (a single-frame conf-.3 box sat ~60px off the
    true mouth — truth 22200) without the fixed-camera assumption: on
    static footage the windowed median equals the whole-clip median; on
    a panning camera it follows the view. Only the RIM_LOCAL_K boxes
    nearest in TIME enter the median — a full ±window median lags a fast
    pan by half the sweep (measured ~160px at 8px/frame in the pinned
    test). Clusters split on x gaps > RIM_SPLIT_PX so a two-rim view
    keeps two references. rim_frames must be sorted ascending, parallel
    to rim_boxes."""
    from bisect import bisect_left, bisect_right

    lo = bisect_left(rim_frames, frame_idx - RIM_LOCAL_FRAMES)
    hi = bisect_right(rim_frames, frame_idx + RIM_LOCAL_FRAMES)
    pairs = sorted(zip(rim_frames[lo:hi], rim_boxes[lo:hi]),
                   key=lambda fb: abs(fb[0] - frame_idx))[:RIM_LOCAL_K]
    boxes = sorted((b for _f, b in pairs), key=lambda b: (b[0] + b[2]) / 2)
    if not boxes:
        return []
    clusters: list[list[tuple]] = [[boxes[0]]]
    for b in boxes[1:]:
        prev = clusters[-1][-1]
        if (b[0] + b[2]) / 2 - (prev[0] + prev[2]) / 2 > RIM_SPLIT_PX:
            clusters.append([b])
        else:
            clusters[-1].append(b)
    return [tuple(np.median(np.array(c, dtype=np.float64), axis=0))
            for c in clusters]


def _nearest_rim(static: list[tuple], x: float) -> tuple | None:
    best = None
    for b in static:
        d = abs((b[0] + b[2]) / 2 - x)
        if best is None or d < best[0]:
            best = (d, b)
    return best[1] if best else None


def derive_events(track: list[dict], bodies_by_frame: dict,
                  rims_by_frame: dict) -> list[dict]:
    """HOLD / RELEASE / RIM_ARRIVAL / SHOT from the fused track.

    Grammar v2 (2026-08-25, mechanism observed on the HS clip):
    * CONTACT RUNS are the atoms — every stretch of the ball on one body,
      ANY duration. A 134ms catch-and-shoot is a real contact even though
      it can never be a possession.
    * POSSESSIONS (hold/release events) are runs >= HOLD_MIN_FRAMES,
      merged across same-body dribble gaps (v1 emitted a shot + 2 fake
      putbacks from one player's three dribble micro-holds).
    * A SHOT's shooter is the LAST CONTACT before the flight — arrival-
      centric binding. v1 bound arrivals to the last qualifying
      possession, which named the PASSER on catch-and-shoot (truth #3's
      shot rolled back 3s to the dribbler's release).

    Spine v3 (2026-08-25, the ev-9 defender-binding class):
    * GRAZE GATE — a sub-possession contact that begins while the ball is
      already rising >= GRAZE_UP_VY px/frame is the launched ball passing
      a closeout defender's bbox, not a release. Grazes never take a shot
      binding (kind=contact_graze, kept as data).
    * CONTACT RUNS PERSIST as kind=contact rows so downstream consumers
      (shot_attribution's team gate) can re-point a wrong-team binding to
      the previous same-flight contact.

    bodies_by_frame: frame -> [(quark track_id, bbox)]
    rims_by_frame:   frame -> [bbox]
    """
    def holder_at(row) -> tuple[int, float] | None:
        best = None
        for tid, (x1, y1, x2, y2) in bodies_by_frame.get(row["frame_idx"], ()):
            dx = max(x1 - row["x"], 0.0, row["x"] - x2)
            dy = max(y1 - row["y"], 0.0, row["y"] - y2)
            d = float(np.hypot(dx, dy))
            if d <= HOLD_DIST_PX and (best is None or d < best[1]):
                best = (tid, d)
        return best

    # footage-agnostic rim reference (2026-08-26 ruling): time-local
    # medians, never a whole-clip box
    _rf: list[int] = []
    _rb: list[tuple] = []
    for _f in sorted(rims_by_frame):
        for _b in rims_by_frame[_f]:
            _rf.append(_f)
            _rb.append(_b)
    _rim_memo: dict[int, list[tuple]] = {}

    def rims_at(frame_idx: int) -> list[tuple]:
        got = _rim_memo.get(frame_idx)
        if got is None:
            got = local_rims(_rf, _rb, frame_idx)
            _rim_memo[frame_idx] = got
        return got

    def at_rim(row, scale: float = RIM_ARRIVE_SCALE) -> bool:
        for x1, y1, x2, y2 in rims_at(row["frame_idx"]):
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if (abs(row["x"] - cx) <= w * scale
                    and abs(row["y"] - cy) <= h * scale):
                return True
        return False

    # pass 1: contact runs (consecutive on-body frames, one body) + arrivals
    runs: list[dict] = []
    cur: dict | None = None
    arrivals: list[dict] = []
    prev_seg = None
    recent: list[tuple[int, float]] = []  # (frame_idx, y) confirmed rows

    def incoming_vy(before_frame: int) -> float | None:
        pts = [(f, y) for f, y in recent if f < before_frame]
        if len(pts) < 2:
            return None
        (f0, y0), (f1, y1) = pts[0], pts[-1]
        return (y1 - y0) / max(f1 - f0, 1)

    for row in track:
        if row["seg_id"] != prev_seg:
            cur = None
            prev_seg = row["seg_id"]
            recent = []
        h = holder_at(row) if row["source"] != "interp" else None
        if h is None or (cur is not None and h[0] != cur["track_id"]):
            if cur is not None and (h is not None
                                    or cur["gap"] >= HOLD_BREAK_FRAMES):
                runs.append(cur)
                cur = None
            elif cur is not None:
                cur["gap"] += 1
        if h is not None:
            if cur is None or cur["track_id"] != h[0]:
                if cur is not None:
                    runs.append(cur)
                vy = incoming_vy(row["frame_idx"])
                cur = {"track_id": h[0], "ts_ms": row["ts_ms"],
                       "ts_end_ms": row["ts_ms"], "n": 1, "gap": 0,
                       "x": row["x"], "y": row["y"],
                       "in_vy": vy if vy is not None else 0.0}
            else:
                cur["ts_end_ms"] = row["ts_ms"]
                cur["n"] += 1
                cur["gap"] = 0
                cur["x"], cur["y"] = row["x"], row["y"]
        if row["source"] != "interp":
            recent.append((row["frame_idx"], row["y"]))
            if len(recent) > GRAZE_VEL_FRAMES:
                recent.pop(0)
        if at_rim(row) and row["source"] != "interp" and cur is None:
            # FLIGHT gate (v9 finding): a ball inside an active contact
            # run can't "arrive" — static rims removed the per-frame
            # detection lookup's accidental temporal suppression and a
            # HELD ball under the basket started firing arrivals
            # (spine shots 25->40, precision .275). Arrivals come from
            # flight only.
            # Descent gate (truth-66200 class): a ball RISING past the
            # rim REGION (the 2x slop) is the shot going up, not an
            # arrival — the ascent-arrival stole the event and the real
            # drop got putback-merged away. A rising ball inside the
            # TIGHT rim box still arrives (dunk/layup class).
            vy_in = incoming_vy(row["frame_idx"] + 1)
            rising = (vy_in is not None and vy_in < -GRAZE_UP_VY
                      and not at_rim(row, scale=1.0))
            if not rising and (not arrivals
                               or row["ts_ms"] - arrivals[-1]["ts_ms"] > 1000):
                arrivals.append({"kind": "rim_arrival", "ts_ms": row["ts_ms"],
                                 "x": row["x"], "y": row["y"],
                                 "frame_idx": row["frame_idx"]})
    if cur is not None:
        runs.append(cur)

    # graze verdict: sub-possession contact entered by a rising ball
    for run in runs:
        run["graze"] = (run["n"] < HOLD_MIN_FRAMES
                        and run.get("in_vy", 0.0) <= -GRAZE_UP_VY)

    # possessions: qualifying runs merged across same-body dribble gaps
    events: list[dict] = []
    poss: list[dict] = []
    for run in runs:
        if run["n"] < HOLD_MIN_FRAMES:
            continue
        if (poss and poss[-1]["track_id"] == run["track_id"]
                and run["ts_ms"] - poss[-1]["ts_end_ms"] <= HOLD_MERGE_GAP_MS):
            poss[-1]["ts_end_ms"] = run["ts_end_ms"]
            poss[-1]["x"], poss[-1]["y"] = run["x"], run["y"]
        else:
            poss.append(dict(run))
    for p in poss:
        events.append({"kind": "hold", "ts_ms": p["ts_ms"],
                       "ts_end_ms": p["ts_end_ms"], "track_id": p["track_id"],
                       "x": p["x"], "y": p["y"]})
        events.append({"kind": "release", "ts_ms": p["ts_end_ms"],
                       "track_id": p["track_id"], "x": p["x"], "y": p["y"]})
    events.extend(arrivals)

    # contact runs persist (any duration; grazes labeled, data kept)
    for run in runs:
        events.append({"kind": "contact_graze" if run["graze"] else "contact",
                       "ts_ms": run["ts_ms"], "ts_end_ms": run["ts_end_ms"],
                       "track_id": run["track_id"],
                       "x": run["x"], "y": run["y"],
                       "n_track_frames": run["n"]})

    # SHOT = arrival bound to the LAST non-graze contact run (any
    # duration) that ended a plausible flight before it
    shots: list[dict] = []
    for arr in arrivals:
        best = None
        for run in runs:
            if run["graze"]:
                continue
            dt = arr["ts_ms"] - run["ts_end_ms"]
            if MIN_FLIGHT_MS <= dt <= RELEASE_RIM_MAX_MS and (
                    best is None or run["ts_end_ms"] > best["ts_end_ms"]):
                best = run
        if best is not None:
            shots.append({"kind": "shot", "ts_ms": arr["ts_ms"],
                          "ts_end_ms": None,
                          "track_id": best["track_id"],
                          "x": best["x"], "y": best["y"],
                          "release_ts_ms": best["ts_end_ms"],
                          "arr_frame": arr["frame_idx"]})

    # putback merge: taps in one rim scramble arrive on top of each other.
    # The FIRST is the attempt; the rest stay as kind=putback (data kept,
    # consumers read kind=shot only).
    shots.sort(key=lambda s: s["ts_ms"])
    last_kept = None
    for s in shots:
        if last_kept is not None and s["ts_ms"] - last_kept <= PUTBACK_MERGE_MS:
            s["kind"] = "putback"
        else:
            last_kept = s["ts_ms"]

    # geometric made/miss per shot: the track's own path through the rim
    by_ts = sorted(track, key=lambda r: r["ts_ms"])
    for s in shots:
        lo = s["ts_ms"] + RIM_PASS_WINDOW_MS[0]
        hi = s["ts_ms"] + RIM_PASS_WINDOW_MS[1]
        win = [r for r in by_ts if lo <= r["ts_ms"] <= hi]
        # rim judged where (and WHEN) the ball arrived — the local boxes
        # at the arrival frame, nearest to the shooter's side
        rim = (_nearest_rim(rims_at(s["arr_frame"]), s["x"])
               if s.get("x") is not None else None)
        s["made_geom"] = rim_pass_verdict(win, rim)
    events.extend(shots)
    return events


# ----------------------------------------------------------------- stage --

def run(video: Path, out_root: Path, job_id: str,
        wasb_root: Path | None = None,
        wasb_weights: Path | None = None,
        binding_stage: Path | None = None) -> dict:
    job_dir = out_root / job_id
    if stage_complete(job_dir / "ball_track") and stage_complete(
            job_dir / "ball_events"):
        return {"job_id": job_id, "skipped": True,
                "reason": "stage already complete"}
    started = time.monotonic()

    det_by_frame = _det_balls(job_dir)
    ts_of: dict[int, int] = {}
    rims_by_frame: dict[int, list] = defaultdict(list)
    t = read_stage(job_dir / "detections")
    for row in t.to_pylist():
        ts_of.setdefault(int(row["frame_idx"]), int(row["ts_ms"]))
        if row["cls"] == CLS_RIM and (row["conf"] or 0) >= 0.3:
            rims_by_frame[int(row["frame_idx"])].append(
                (row["x1"], row["y1"], row["x2"], row["y2"]))

    wasb_by_frame: dict[int, list] = {}
    if wasb_weights is not None:
        # k3 cache is a different artifact than the v2 single-peak file
        # (wasb_peaks.json) — that one stays untouched on disk
        cache_path = job_dir / "_vlm_cache" / "wasb_peaks_k3.json"
        if cache_path.exists():
            wasb_by_frame = {
                int(k): [tuple(p) for p in v]
                for k, v in json.loads(cache_path.read_text()).items()}
            print(f"BALL-TRACK wasb peaks from cache: {len(wasb_by_frame)}",
                  flush=True)
        else:
            wasb_by_frame = wasb_full_pass(
                video, wasb_root or Path("/work/wasb"), wasb_weights,
                log=lambda m: print(f"BALL-TRACK {m}", flush=True))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(
                {str(k): [list(p) for p in v]
                 for k, v in wasb_by_frame.items()}))

    track = fuse_track(det_by_frame, wasb_by_frame, ts_of)

    # tiled second pass (ev13 supply class): where the fused track still
    # has >= GAP_MIN_FRAMES holes, re-run WASB at 2x effective resolution
    # on exactly those frames and re-fuse. Cached like the k3 pass.
    n_tiled_frames = 0
    if wasb_weights is not None and track:
        spans = uncovered_spans(track, track[0]["frame_idx"],
                                track[-1]["frame_idx"])
        needed = {f for a, b in spans for f in range(a, b + 1)}
        if needed:
            tile_cache = job_dir / "_vlm_cache" / "wasb_peaks_tiled.json"
            if tile_cache.exists():
                tiled = {int(k): [tuple(p) for p in v]
                         for k, v in json.loads(
                             tile_cache.read_text()).items()}
            else:
                tiled = tiled_wasb_pass(
                    video, wasb_root or Path("/work/wasb"), wasb_weights,
                    needed,
                    log=lambda m: print(f"BALL-TRACK {m}", flush=True))
                tile_cache.parent.mkdir(parents=True, exist_ok=True)
                tile_cache.write_text(json.dumps(
                    {str(k): [list(p) for p in v]
                     for k, v in tiled.items()}))
            n_tiled_frames = len(tiled)
            if tiled:
                merged = dict(wasb_by_frame)
                for f, ps in tiled.items():
                    have = merged.get(f)
                    if have is None:
                        merged[f] = ps
                    else:
                        if isinstance(have, tuple):
                            have = [have]
                        merged[f] = list(have) + list(ps)
                track = fuse_track(det_by_frame, merged, ts_of)

    stage = binding_stage
    if stage is None:
        for name in ("quark_binding", "track_binding"):
            if stage_complete(job_dir / name):
                stage = job_dir / name
                break
    bodies_by_frame: dict[int, list] = defaultdict(list)
    if stage is not None:
        b = read_stage(stage)
        cols = {c: b.column(c).to_numpy(zero_copy_only=False)
                for c in ("frame_idx", "track_id", "x1", "y1", "x2", "y2")}
        keep = np.ones(len(cols["frame_idx"]), dtype=bool)
        if "coasted" in b.column_names:
            keep = ~b.column("coasted").to_numpy(zero_copy_only=False)
        for i in np.nonzero(keep)[0]:
            bodies_by_frame[int(cols["frame_idx"][i])].append(
                (int(cols["track_id"][i]),
                 (cols["x1"][i], cols["y1"][i],
                  cols["x2"][i], cols["y2"][i])))

    events = derive_events(track, bodies_by_frame, dict(rims_by_frame))

    writer = ArtifactWriter(job_dir / "ball_track", BALL_TRACK_SCHEMA)
    for r in track:
        writer.add({"job_id": job_id, **{k: r[k] for k in (
            "frame_idx", "ts_ms", "x", "y", "conf", "source", "seg_id")}})
    writer.close()
    ew = ArtifactWriter(job_dir / "ball_events", BALL_EVENTS_SCHEMA)
    for e in events:
        ew.add({"job_id": job_id,
                "kind": e["kind"], "ts_ms": e["ts_ms"],
                "ts_end_ms": e.get("ts_end_ms"),
                "track_id": e.get("track_id"),
                "x": e.get("x"), "y": e.get("y"),
                "release_ts_ms": e.get("release_ts_ms"),
                "n_track_frames": e.get("n_track_frames"),
                "made_geom": e.get("made_geom")})
    ew.close()

    n_conf = sum(1 for r in track if r["source"] != "interp")
    frames_spanned = (track[-1]["frame_idx"] - track[0]["frame_idx"] + 1
                      if track else 0)
    return {
        "job_id": job_id,
        "track_rows": len(track),
        "confirmed_rows": n_conf,
        "coverage_of_span": round(n_conf / max(frames_spanned, 1), 4),
        "segments": len({r["seg_id"] for r in track}),
        "det_frames": len(det_by_frame),
        "wasb_frames": len(wasb_by_frame),
        "tiled_frames": n_tiled_frames,
        "holds": sum(1 for e in events if e["kind"] == "hold"),
        "releases": sum(1 for e in events if e["kind"] == "release"),
        "rim_arrivals": sum(1 for e in events if e["kind"] == "rim_arrival"),
        "shots": sum(1 for e in events if e["kind"] == "shot"),
        "putbacks_merged": sum(1 for e in events if e["kind"] == "putback"),
        "contacts": sum(1 for e in events if e["kind"] == "contact"),
        "grazes_gated": sum(1 for e in events
                            if e["kind"] == "contact_graze"),
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def spine_shooters(job_dir: Path) -> dict[int, int]:
    """rim-arrival ts_ms -> shooter quark track_id, for consumers."""
    if not stage_complete(job_dir / "ball_events"):
        return {}
    out: dict[int, int] = {}
    for row in read_stage(job_dir / "ball_events").to_pylist():
        if row["kind"] == "shot" and row["track_id"] is not None:
            out[int(row["ts_ms"])] = int(row["track_id"])
    return out


def spine_contacts(job_dir: Path) -> list[dict]:
    """Non-graze contact runs, ts-ordered — the team gate's alternates."""
    if not stage_complete(job_dir / "ball_events"):
        return []
    rows = [r for r in read_stage(job_dir / "ball_events").to_pylist()
            if r["kind"] == "contact" and r["track_id"] is not None]
    rows.sort(key=lambda r: r["ts_end_ms"] or r["ts_ms"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--wasb-root", type=Path, default=Path("/work/wasb"))
    ap.add_argument("--wasb-weights", type=Path, default=None,
                    help="explicit WASB checkpoint; omit = detector-only "
                         "fusion (degraded: no flight coverage)")
    ap.add_argument("--binding-stage", type=Path, default=None)
    args = ap.parse_args()
    print(json.dumps(run(args.video, args.out, args.job_id,
                         wasb_root=args.wasb_root,
                         wasb_weights=args.wasb_weights,
                         binding_stage=args.binding_stage), indent=2))


if __name__ == "__main__":
    main()
