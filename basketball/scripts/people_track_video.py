"""People-track video: every person boxed with a persistent own-ID,
optionally with the ball overlay in the same render.

The people counterpart of ball_track_video (2026-08-03): boxes
follow each person continuously; each carries OUR id. With a v3+
relink map the ids are time-ranged lock/person ids (P-labels, R for
the refs/other class, jersey number when known); unmapped tracks are
gray "?" residue. With --ball-weights the SAME clip also carries the
yellow ball ring (2026-08-04: ball and people tracking in one render, so
both can be reviewed in motion) — identical WASB pass, motion-
consistency and scene gates as ball_track_video v3.

Usage (inside cvbench; GPU when --ball-weights):
    python scripts/people_track_video.py --video "<file>" \
        --game-dir /work/out-harvest/cal_fsu_v3real \
        --relink-map /work/models/ball-v1/relink_smoke_v4.json \
        --ball-weights /work/weights/wasb/wasb_basketball_best.pth.tar \
        --out /work/models/ball-v1/people_ball_smoke.mp4 \
        --start-s 600 --duration-s 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from bisect import bisect_left
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from montehall_cv.store.artifacts import read_stage
from ball_track_video import load_scene_ok

BOX_W = 3
MATCH_MS = 17  # binding rows are at 60fps; sampled frames align within one


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--game-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=0.0)
    ap.add_argument("--relink-map", type=Path, default=None,
                    help="people_relink json: track_id -> person_id; "
                         "one color per PERSON instead of per track")
    ap.add_argument("--ball-weights", type=Path, default=None,
                    help="WASB checkpoint; draws the yellow ball ring "
                         "in the same render")
    ap.add_argument("--wasb-root", type=Path, default=Path("/work/wasb"))
    ap.add_argument("--pose", action="store_true",
                    help="stick-figure skeletons (KeypointRCNN) inside "
                         "locked boxes, in the lock's color")
    ap.add_argument("--binding-stage", type=Path, default=None,
                    help="alternate binding-shaped stage dir (e.g. a "
                         "quark_probe quark_binding); default "
                         "<game-dir>/track_binding")
    ap.add_argument("--hide-coasted", action="store_true",
                    help="drop rows the binding marked coasted (seeker "
                         "guessing through an occlusion, no detection "
                         "under the box). The renderer otherwise draws "
                         "every row, so a coast-heavy binding paints "
                         "boxes over empty floor. No-op on bindings "
                         "without a `coasted` column (e.g. ByteTrack "
                         "track_binding).")
    args = ap.parse_args()

    import av
    from PIL import Image, ImageDraw, ImageFont

    started = time.monotonic()
    binding = read_stage(args.binding_stage
                         or args.game_dir / "track_binding")
    bts = binding.column("ts_ms").to_numpy()
    btrack = binding.column("track_id").to_numpy()
    bx1 = binding.column("x1").to_numpy()
    by1 = binding.column("y1").to_numpy()
    bx2 = binding.column("x2").to_numpy()
    by2 = binding.column("y2").to_numpy()
    n_coast_hidden = 0
    if args.hide_coasted and "coasted" in binding.column_names:
        keep = ~binding.column("coasted").to_numpy(zero_copy_only=False)
        n_coast_hidden = int((~keep).sum())
        bts, btrack, bx1, by1, bx2, by2 = (
            a[keep] for a in (bts, btrack, bx1, by1, bx2, by2))
    order = np.argsort(bts, kind="stable")
    bts, btrack, bx1, by1, bx2, by2 = (
        a[order] for a in (bts, btrack, bx1, by1, bx2, by2))

    # ⚠ NO entities/team join: the replayed binding is NOT faithful to
    # the original run's track ids (VERIFY_TRACK_BINDING 08-03 — 6,910
    # span mismatches), so the stored entities table keys don't apply.
    # v1 ids are the replayed ByteTrack ids, compact-renumbered. Team
    # color + cross-cut persistence belong to the attribute re-link
    # layer, measured against this baseline.
    scene_ok = None  # bound after the first frame gives frame height

    person_of: dict[int, int] = {}
    person_meta: dict[int, dict] = {}
    seg_of_track: dict[int, list[tuple]] = {}  # tid -> [(t0,t1,person)]
    coasts: list[tuple] = []  # (person, ta, box_a, tb, box_b)
    locks_only = False
    if args.relink_map:
        relink = json.loads(args.relink_map.read_text())
        person_meta = {int(p): m
                       for p, m in relink.get("person_meta", {}).items()}
        if "segments" in relink:  # v3+: time-ranged (track can swap teams)
            for tid, t0, t1, p in relink["segments"]:
                seg_of_track.setdefault(int(tid), []).append(
                    (int(t0), int(t1), int(p)))
            for segs in seg_of_track.values():
                segs.sort()
        else:
            person_of = {int(t): p for t, p in relink["person_of"].items()}
        # v5 (lock engine w/ Kalman coasts): draw ONLY locked persons —
        # no gray residue boxes on coaches/bench/crowd (08-04) —
        # and bridge lock gaps with lerped coast boxes, the people
        # counterpart of the ball ring's dim interp state
        if "coasts" in relink:
            locks_only = True
            coasts = [(int(p), float(ta), box_a, float(tb), box_b)
                      for p, ta, box_a, tb, box_b in relink["coasts"]]

    display_id: dict = {}  # (kind, id) -> compact int by first appearance
    RESIDUE = (150, 150, 150)  # sub-quorum tracks: no identity claimed
    palette = [(255, 214, 0), (59, 160, 255), (255, 138, 59),
               (120, 220, 120), (230, 120, 230), (120, 220, 220),
               (255, 90, 90), (240, 240, 240), (170, 120, 255),
               (255, 180, 120), (90, 200, 90), (255, 120, 180),
               (120, 160, 255), (220, 220, 120), (120, 255, 200),
               (200, 140, 90), (140, 140, 255), (255, 230, 140),
               (100, 230, 230), (230, 100, 100)]

    def person_at(track_id: int, ts_ms: float) -> int | None:
        if track_id in person_of:
            return person_of[track_id]
        for t0, t1, p in seg_of_track.get(track_id, ()):
            if t0 <= ts_ms <= t1:
                return p
        return None

    def person_label(p: int) -> tuple[str, tuple]:
        key = ("p", p)
        if key not in display_id:
            display_id[key] = len(display_id) + 1
        n = display_id[key]
        meta = person_meta.get(p) or {}
        prefix = "R" if meta.get("team") == 2 else "P"
        number = meta.get("number")
        text = f"{prefix}{n} #{number}" if number else f"{prefix}{n}"
        return text, palette[n % len(palette)]

    def label_of(track_id: int, ts_ms: float) -> tuple[str, tuple] | None:
        # person id when the re-link mapped this track at this instant
        # (v3+ maps are time-ranged: one track can span a team swap).
        # Refs (team 2) label as R. Unmapped under a v5 locks-only map
        # = draw NOTHING (coaches/bench/crowd never render); under a
        # v2-v4 map = gray "?" residue; v1 keeps per-track colors.
        p = person_at(track_id, ts_ms)
        if p is not None:
            return person_label(p)
        if locks_only:
            return None
        if person_meta:
            return "?", RESIDUE
        key = ("t", track_id)
        if key not in display_id:
            display_id[key] = len(display_id) + 1
        n = display_id[key]
        return f"T{n}", palette[n % len(palette)]

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    def frames():
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

    # ---- optional ball pass: WASB peaks over the same sampled frames
    # (ball_track_video v3 gates: motion consistency + scene signature)
    ball_track: list = []
    if args.ball_weights is not None:
        import cv2
        import torch

        from ball_track_video import (
            BATCH, SCORE_MIN, WASB_W, build_track, consistent_peaks)
        from wasb_shot_window_probe import (
            IMAGENET_MEAN, IMAGENET_STD, heatmap_of, load_wasb)

        model, dev, model_cfg = load_wasb(args.wasb_root, args.ball_weights)
        wh = (model_cfg["inp_width"], model_cfg["inp_height"])
        peaks: list[tuple] = []
        buf: list[tuple] = []
        pend: list[tuple] = []
        native_scale = None
        native_h = None
        out_times: list[float] = []

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
                    peaks.append((pend[k][0],
                                  float(yx[1]) * native_scale,
                                  float(yx[0]) * native_scale, score))
            pend.clear()

        n_sampled = 0
        for ts, frame in frames():
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
                print(f"PEOPLEBALL-A: {n_sampled} frames, "
                      f"{len(peaks)} peaks", flush=True)
        flush()
        peaks = consistent_peaks(peaks)
        ball_scene_ok = load_scene_ok(args.game_dir, native_h) \
            if native_h else None
        ball_track = build_track(peaks, out_times, ball_scene_ok)
        print(f"PEOPLEBALL-A-DONE peaks={len(peaks)} "
              f"cover={sum(p is not None for p in ball_track)}"
              f"/{len(ball_track)}", flush=True)

    RING_R, YELLOW = 16, (255, 214, 0)

    # ---- optional pose pass state (stick figures, 08-04) -------
    pose_model = pose_dev = None
    if args.pose:
        import torch
        from torchvision.models.detection import (
            KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn)

        pose_dev = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        pose_model = keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
        ).to(pose_dev).eval()
    # COCO 17-keypoint skeleton edges
    SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
                (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
                (14, 16), (0, 5), (0, 6)]
    POSE_SCORE_MIN = 0.7
    POSE_KP_MIN = 0.5
    POSE_IOU_MIN = 0.4

    def frame_poses(rgb: np.ndarray):
        """[(box, keypoints(17,3))] for confident persons in the frame."""
        import torch
        with torch.no_grad():
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
            pred = pose_model([t.to(pose_dev)])[0]
        out = []
        keep = pred["scores"] >= POSE_SCORE_MIN
        for box, kps in zip(pred["boxes"][keep].cpu().numpy(),
                            pred["keypoints"][keep].cpu().numpy()):
            out.append((box, kps))
        return out

    def draw_skeleton(draw, kps, color) -> None:
        for a, b in SKELETON:
            if kps[a][2] >= POSE_KP_MIN and kps[b][2] >= POSE_KP_MIN:
                draw.line([(float(kps[a][0]), float(kps[a][1])),
                           (float(kps[b][0]), float(kps[b][1]))],
                          fill=color, width=3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".mp4.tmp")
    n_out = n_boxes = 0
    t0 = None
    with av.open(str(tmp), "w", format="mp4") as out_container:
        ostream = out_container.add_stream(
            "libx264", options={"crf": "23", "preset": "veryfast"})
        ostream.pix_fmt = "yuv420p"
        ostream.codec_context.time_base = Fraction(1, 1000)
        n_ring = n_coast = n_skel = 0
        for i, (ts, frame) in enumerate(frames()):
            if ostream.width == 0:
                ostream.width, ostream.height = frame.width, frame.height
            if t0 is None:
                t0 = ts
                scene_ok = load_scene_ok(args.game_dir, frame.height)
            rgb = frame.to_ndarray(format="rgb24")
            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)
            if scene_ok is not None and scene_ok(ts):
                poses = frame_poses(rgb) if pose_model is not None else []
                lo = bisect_left(bts, ts * 1000 - MATCH_MS)
                hi = bisect_left(bts, ts * 1000 + MATCH_MS)
                for m in range(lo, hi):
                    labeled = label_of(int(btrack[m]), float(bts[m]))
                    if labeled is None:
                        continue  # locks-only: coaches/crowd never drawn
                    text, color = labeled
                    box = [float(bx1[m]), float(by1[m]),
                           float(bx2[m]), float(by2[m])]
                    draw.rectangle(box, outline=color, width=BOX_W)
                    draw.text((box[0], max(0.0, box[1] - 26)), text,
                              fill=color, font=font)
                    n_boxes += 1
                    if poses:
                        best, best_iou = None, POSE_IOU_MIN
                        for pbox, kps in poses:
                            ix = min(box[2], pbox[2]) - max(box[0], pbox[0])
                            iy = min(box[3], pbox[3]) - max(box[1], pbox[1])
                            if ix <= 0 or iy <= 0:
                                continue
                            inter = ix * iy
                            union = ((box[2] - box[0]) * (box[3] - box[1])
                                     + (pbox[2] - pbox[0])
                                     * (pbox[3] - pbox[1]) - inter)
                            if inter / union > best_iou:
                                best, best_iou = kps, inter / union
                        if best is not None:
                            draw_skeleton(draw, best, color)
                            n_skel += 1
                # coast boxes: the lock's predicted/bridged position
                # while its target was out of sight — thin outline,
                # the people analog of the ball's dim interp ring
                for p, ta, box_a, tb, box_b in coasts:
                    if not (ta < ts * 1000 < tb):
                        continue
                    f = (ts * 1000 - ta) / max(tb - ta, 1e-6)
                    cb = [a + (b - a) * f for a, b in zip(box_a, box_b)]
                    text, color = person_label(p)
                    draw.rectangle(cb, outline=color, width=1)
                    draw.text((cb[0], max(0.0, cb[1] - 26)), text,
                              fill=color, font=font)
                    n_coast += 1
            pos = ball_track[i] if i < len(ball_track) else None
            if pos is not None:
                x, y, kind = pos
                draw.ellipse([x - RING_R, y - RING_R,
                              x + RING_R, y + RING_R], outline=YELLOW,
                             width=5 if kind == "det" else 2)
                n_ring += 1
            vf = av.VideoFrame.from_ndarray(np.asarray(img), format="rgb24")
            vf.pts = int((ts - t0) * 1000)
            out_container.mux(ostream.encode(vf))
            n_out += 1
            if n_out % 2000 == 0:
                print(f"PEOPLETRACK: {n_out} frames", flush=True)
        out_container.mux(ostream.encode())
    tmp.rename(args.out)
    print("PEOPLETRACK_DONE " + json.dumps({
        "out": str(args.out), "frames": n_out, "boxes": n_boxes,
        "coast_rows_hidden": n_coast_hidden,
        "ball_ring_frames": n_ring, "coast_boxes": n_coast,
        "skeletons": n_skel,
        "distinct_display_ids": len(display_id),
        "bytes": args.out.stat().st_size,
        "wall_s": round(time.monotonic() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
