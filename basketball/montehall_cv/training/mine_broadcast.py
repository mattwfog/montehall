"""P1 broadcast label mining: PBP shot windows -> COCO detection shards.

Inside each aligned shooting-play window the current detector runs at low
confidence; labels are kept only with corroboration (plan P1):
- player: straight confidence bar (broadcast players are the detector's
  strong suit; the collapse was ball/rim)
- ball: latent-state chain filter (Ask 0b, 2026-07-16 — design principles,
  training plan §7 cell 5): candidates are linked into velocity-bounded
  chains; a chain is believed only if it MOVES (a real ball is dribbled,
  passed, or shot inside a shot window) — a smaller spread suffices when
  the chain is possession-anchored to a player box, a larger spread is
  required in open space (flight). Unbelieved candidates are dropped and
  windows with no believed chain emit NO ball labels (abstain > wrong).
  Replaces the pairwise temporal-neighbor test, which stationary false
  positives (crowd heads) trivially self-corroborated — the 2026-07-16
  audit's ~10-15% clearly-wrong band, clustered in ball-invisible windows.
- rim: spatial persistence across nearby frames (static per camera hold)
Negatives are mined from clock-absent spans (replays/graphics/crowd) as
zero-annotation images — the false-positive reservoir.

Output: one COCO source dir per game (train/valid + _annotations.coco.json,
categories 1=player 2=ball 3=rim), merge_coco-ready. Every image filename
carries (game, play-window, ts_ms) provenance so bad alignments are
excisable later.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from montehall_cv.store.artifacts import read_stage
from montehall_cv.store.records import DetClass

CATEGORIES = [
    {"id": 1, "name": "player"},
    {"id": 2, "name": "ball"},
    {"id": 3, "name": "rim"},
]
WINDOW_PRE_S = 6.0  # PBP clock lags the shot: window reaches back
WINDOW_POST_S = 2.0
FRAME_STRIDE = 6  # ~5 fps at 30 fps source
DETECT_FLOOR = 0.15
PLAYER_KEEP = 0.5
# Ball floors tightened after the game06 smoke check (2026-07-13): the loose
# rules kept ~5 "balls"/frame — spectator heads/caps — in a one-ball sport.
BALL_CORROB_FLOOR = 0.35
# Latent-state chain filter (Ask 0b). Pixel constants are at 640-wide
# (fmt-18 360p) base and scale linearly with frame width; confidence alone
# never keeps a ball label — belief in the chain does.
BALL_CHAIN_LINK_S = 0.65     # max time gap between chained candidates
BALL_CHAIN_LINK_PX = 90.0    # max travel across one link at 640w
BALL_CHAIN_MIN_LEN = 3       # observations before a chain is believable
BALL_SPREAD_ANCHORED_PX = 16.0  # dribble bounce scale at 640w
BALL_SPREAD_FLIGHT_PX = 45.0    # unanchored chains must clearly fly
BALL_PLAYER_ANCHOR_PX = 28.0    # ball-center-to-player-box margin at 640w
BALL_BASE_WIDTH = 640.0
MAX_BALLS_PER_FRAME = 2  # airborne + a hand-off ambiguity; never a crowd
MAX_RIMS_PER_FRAME = 2
RIM_KEEP_FLOOR = 0.3
RIM_NEIGHBOR_S = 1.5
RIM_NEIGHBOR_IOU = 0.3
NEGATIVE_GAP_S = 8.0  # no confident clock read for this long = non-live
NEGATIVE_FRACTION = 0.12
VAL_FRACTION = 0.15


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter))


def _center(box: np.ndarray) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2)


def shot_windows(align_dir: Path) -> list[tuple[float, float, str]]:
    plays = read_stage(align_dir / "pbp_alignment").to_pylist()
    windows = [
        (p["video_t"] - WINDOW_PRE_S, p["video_t"] + WINDOW_POST_S, p["play_id"])
        for p in plays
        if p["shooting_play"]
    ]
    return sorted(windows)


def nonlive_spans(align_dir: Path, duration_s: float) -> list[tuple[float, float]]:
    reads = read_stage(align_dir / "clock_reads").to_pylist()
    confident = sorted(
        r["video_t"] for r in reads if r["conf"] >= 0.85 and r["clock_s"] is not None
    )
    spans: list[tuple[float, float]] = []
    prev = 0.0
    for t in [*confident, duration_s]:
        if t - prev > NEGATIVE_GAP_S:
            spans.append((prev + 1.0, t - 1.0))
        prev = t
    return spans


def _ball_chains(balls: list[dict], scale: float) -> list[list[dict]]:
    """Greedy velocity-bounded linking of ball candidates into chains."""
    chains: list[list[dict]] = []
    for r in sorted(balls, key=lambda r: r["t"]):
        cx, cy = _center(r["box"])
        best = None
        for chain in chains:
            tail = chain[-1]
            dt = r["t"] - tail["t"]
            if dt <= 0 or dt > BALL_CHAIN_LINK_S:
                continue
            tx, ty = _center(tail["box"])
            if ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5 <= BALL_CHAIN_LINK_PX * scale:
                if best is None or tail["t"] > best[-1]["t"]:
                    best = chain
        if best is not None:
            best.append(r)
        else:
            chains.append([r])
    return chains


def _chain_spread(chain: list[dict]) -> float:
    """Max pairwise center distance — did the candidate actually move?"""
    centers = [_center(r["box"]) for r in chain]
    spread = 0.0
    for i, (x1, y1) in enumerate(centers):
        for x2, y2 in centers[i + 1:]:
            spread = max(spread, ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)
    return spread


def _ball_latent_filter(
    balls: list[dict], players: list[dict], scale: float
) -> list[dict]:
    """Keep only ball candidates that belong to a believed chain (Ask 0b).

    Belief = the chain moves: spread >= flight scale in open space, or the
    smaller dribble scale when possession-anchored (most observations within
    the anchor margin of a same-frame player box). Everything else — above
    all the stationary self-similar crowd blob — is dropped; a window with
    no believed chain emits nothing.
    """
    players_by_fname: dict[str, list[dict]] = defaultdict(list)
    for p in players:
        players_by_fname[p["fname"]].append(p)
    margin = BALL_PLAYER_ANCHOR_PX * scale

    def anchored(r: dict) -> bool:
        cx, cy = _center(r["box"])
        for p in players_by_fname.get(r["fname"], ()):
            b = p["box"]
            if (b[0] - margin <= cx <= b[2] + margin
                    and b[1] - margin <= cy <= b[3] + margin):
                return True
        return False

    kept: list[dict] = []
    candidates = [r for r in balls if r["conf"] >= BALL_CORROB_FLOOR]
    for chain in _ball_chains(candidates, scale):
        if len(chain) < BALL_CHAIN_MIN_LEN:
            continue
        spread = _chain_spread(chain)
        if spread >= BALL_SPREAD_FLIGHT_PX * scale:
            kept.extend(chain)
            continue
        anchored_n = sum(1 for r in chain if anchored(r))
        if (spread >= BALL_SPREAD_ANCHORED_PX * scale
                and anchored_n * 2 >= len(chain)):
            kept.extend(chain)
    return kept


def _corroborate(records: list[dict], frame_w: float = BALL_BASE_WIDTH) -> list[dict]:
    """Apply per-class keep rules over one game's window detections."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_class[r["cls"]].append(r)
    kept: list[dict] = []
    players = [r for r in by_class[int(DetClass.PERSON)] if r["conf"] >= PLAYER_KEEP]
    kept.extend(players)

    scale = frame_w / BALL_BASE_WIDTH
    kept.extend(_ball_latent_filter(by_class[int(DetClass.BALL)], players, scale))

    rims = sorted(by_class[int(DetClass.RIM)], key=lambda r: r["t"])
    for i, r in enumerate(rims):
        if r["conf"] < RIM_KEEP_FLOOR:
            continue
        for j in range(max(0, i - 12), min(len(rims), i + 13)):
            if j == i:
                continue
            n = rims[j]
            if abs(n["t"] - r["t"]) > RIM_NEIGHBOR_S:
                continue
            if _iou(r["box"], n["box"]) >= RIM_NEIGHBOR_IOU:
                kept.append(r)
                break
    return _cap_per_frame(kept)


def _cap_per_frame(kept: list[dict]) -> list[dict]:
    """At most N ball/rim labels per frame, highest confidence first."""
    caps = {int(DetClass.BALL): MAX_BALLS_PER_FRAME, int(DetClass.RIM): MAX_RIMS_PER_FRAME}
    seen: dict[tuple[str, int], int] = {}
    out: list[dict] = []
    for r in sorted(kept, key=lambda r: -r["conf"]):
        cap = caps.get(r["cls"])
        if cap is not None:
            key = (r["fname"], r["cls"])
            if seen.get(key, 0) >= cap:
                continue
            seen[key] = seen.get(key, 0) + 1
        out.append(r)
    return out


def mine(
    video: Path,
    align_dir: Path,
    out_dir: Path,
    weights: Path,
    game_tag: str,
    max_frames: int = 2500,
) -> dict:
    import cv2

    from montehall_cv.pipeline.detect import RFDetrDetector
    from montehall_cv.pipeline.video import probe

    started = time.monotonic()
    done_marker = out_dir / "_SUCCESS"
    if done_marker.exists():
        return json.loads((out_dir / "mine_report.json").read_text())

    info = probe(video)
    duration_s = (info.duration_ms or 0) / 1000.0
    windows = shot_windows(align_dir)
    negatives = nonlive_spans(align_dir, duration_s)
    detector = RFDetrDetector(threshold=DETECT_FLOOR, weights=weights)

    frames_tmp = out_dir / "frames_tmp"
    frames_tmp.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    frame_files: dict[str, tuple[int, int]] = {}
    neg_files: list[str] = []
    batch_imgs: list[np.ndarray] = []
    batch_meta: list[tuple[str, float]] = []

    def flush_batch() -> None:
        if not batch_imgs:
            return
        for (fname, t), det in zip(batch_meta, detector.detect(batch_imgs)):
            for box, conf, cls in zip(det.xyxy, det.conf, det.cls):
                records.append(
                    {"fname": fname, "t": t, "box": box, "conf": float(conf), "cls": int(cls)}
                )
        batch_imgs.clear()
        batch_meta.clear()

    # Seek per window instead of full-game decode (~10x less decode work).
    # PyAV, not cv2.VideoCapture: several replays are AV1 (opencv's bundled
    # ffmpeg has no software AV1 path and silently read-fails — the A10G
    # empty-shard incident, 2026-07-13); PyAV bundles dav1d and the
    # av.time_base seek pattern is proven on these exact files (clock_probe).
    import av

    container = av.open(str(video))
    stream = container.streams.video[0]
    stride_s = FRAME_STRIDE / float(stream.average_rate or 30.0)

    def grab_span(start_s: float, end_s: float):
        container.seek(int(max(0.0, start_s) * av.time_base))
        last = -1e9
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < start_s - 0.25 or t - last < stride_s:
                continue
            if t > end_s:
                return
            last = t
            yield frame.to_ndarray(format="bgr24"), t

    seen_ts: set[int] = set()
    for start_s, end_s, play_id in windows:
        for bgr, t in grab_span(start_s, end_s):
            ts_ms = int(t * 1000)
            if ts_ms in seen_ts:  # overlapping windows share frames
                continue
            seen_ts.add(ts_ms)
            fname = f"{game_tag}_p{play_id}_t{ts_ms}.jpg"
            cv2.imwrite(str(frames_tmp / fname), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            frame_files[fname] = (bgr.shape[1], bgr.shape[0])
            batch_imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            batch_meta.append((fname, t))
            if len(batch_imgs) >= 16:
                flush_batch()
    flush_batch()

    neg_budget = int(max_frames * NEGATIVE_FRACTION)
    for span_start, span_end in negatives:
        if len(neg_files) >= neg_budget:
            break
        for bgr, t in grab_span(span_start, min(span_end, span_start + 2.0)):
            ts_ms = int(t * 1000)
            fname = f"{game_tag}_neg_t{ts_ms}.jpg"
            cv2.imwrite(str(frames_tmp / fname), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            frame_files[fname] = (bgr.shape[1], bgr.shape[0])
            neg_files.append(fname)
            break  # one frame per non-live span, spread across the game
    container.close()

    if windows and not records:
        # Windows existed but the decoder produced nothing detectable —
        # broken decode, not an empty game. Never stamp success on this.
        raise RuntimeError(
            f"{game_tag}: {len(windows)} windows but 0 raw detections — "
            "decode or detector failure; refusing to write an empty shard"
        )

    frame_w = float(next(iter(frame_files.values()))[0]) if frame_files else BALL_BASE_WIDTH
    kept = _corroborate(records, frame_w=frame_w)
    ann_by_frame: dict[str, list[dict]] = defaultdict(list)
    for r in kept:
        ann_by_frame[r["fname"]].append(r)

    positive_frames = sorted(ann_by_frame)
    if len(positive_frames) > max_frames - len(neg_files):
        picks = np.linspace(0, len(positive_frames) - 1,
                            max_frames - len(neg_files)).astype(int)
        positive_frames = [positive_frames[i] for i in sorted(set(picks.tolist()))]
    all_frames = positive_frames + neg_files
    split_at = int(len(positive_frames) * (1 - VAL_FRACTION))
    split_of = {f: ("train" if i < split_at else "valid")
                for i, f in enumerate(positive_frames)}
    split_of.update({f: "train" for f in neg_files})

    coco: dict[str, dict] = {
        s: {"images": [], "annotations": [], "categories": CATEGORIES}
        for s in ("train", "valid")
    }
    img_id = ann_id = 0
    for fname in all_frames:
        split = split_of[fname]
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        shutil.move(str(frames_tmp / fname), str(out_dir / split / fname))
        w, h = frame_files[fname]
        img_id += 1
        coco[split]["images"].append(
            {"id": img_id, "file_name": fname, "width": w, "height": h}
        )
        for r in ann_by_frame.get(fname, []):
            x1, y1, x2, y2 = (float(v) for v in r["box"])
            ann_id += 1
            coco[split]["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": {int(DetClass.PERSON): 1, int(DetClass.BALL): 2,
                                    int(DetClass.RIM): 3}[r["cls"]],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
    for split in ("train", "valid"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        (out_dir / split / "_annotations.coco.json").write_text(json.dumps(coco[split]))
    shutil.rmtree(frames_tmp, ignore_errors=True)

    n_by_class = defaultdict(int)
    for r in kept:
        n_by_class[r["cls"]] += 1
    report = {
        "game": game_tag,
        "windows": len(windows),
        "frames_kept": len(all_frames),
        "negatives": len(neg_files),
        "ann_player": n_by_class[int(DetClass.PERSON)],
        "ann_ball": n_by_class[int(DetClass.BALL)],
        "ann_rim": n_by_class[int(DetClass.RIM)],
        "raw_detections": len(records),
        "ball_candidates": sum(
            1 for r in records
            if r["cls"] == int(DetClass.BALL) and r["conf"] >= BALL_CORROB_FLOOR
        ),
        "frame_w": frame_w,
        "wall_min": round((time.monotonic() - started) / 60, 1),
    }
    (out_dir / "mine_report.json").write_text(json.dumps(report, indent=2))
    done_marker.touch()
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--align-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--game-tag", required=True)
    ap.add_argument("--max-frames", type=int, default=2500)
    args = ap.parse_args()
    report = mine(args.video, args.align_dir, args.out, args.weights,
                  args.game_tag, args.max_frames)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
