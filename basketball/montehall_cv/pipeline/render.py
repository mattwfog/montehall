"""Annotated video renderer — CPU-only overlays on the source footage.

Draws the pipeline's understanding onto every frame: team-colored player
boxes with jersey/entity labels, ball marker + short trail, rim boxes, a
possession/score banner, and a court minimap fed by the geometry-
translated positions. Decode and encode ride PyAV (VFR-safe pts
passthrough); drawing is PIL. No GPU, no models — safe to run on spark
while a training job owns the CUDA allocator.

Output: <out>/<job_id>/annotated/annotated.mp4 (+ _SUCCESS marker).
Silent (no audio track) at v0.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from montehall_cv.court import NCAA_LENGTH_FT, NCAA_WIDTH_FT
from montehall_cv.store.artifacts import read_stage, stage_complete
from montehall_cv.store.records import DetClass, possession_offense_cluster

TEAM_COLORS = {0: (59, 160, 255), 1: (255, 138, 59)}  # blue / orange
REF_COLOR = (150, 150, 150)
UNGROUPED_COLOR = (220, 220, 220)
BALL_COLOR = (255, 214, 0)
RIM_COLOR = (0, 200, 120)
TRAIL_MS = 900
MIN_JERSEY_PROB = 0.6
MINIMAP_PX_PER_FT = 2.0
MINIMAP_MARGIN_PX = 12


def run(
    video: Path, out_root: Path, job_id: str,
    scale: float = 1.0, max_frames: int = 0,
) -> dict:
    from montehall_cv.pipeline.video import decode_frames, probe

    job_dir = out_root / job_id
    stage_dir = job_dir / "annotated"
    if stage_complete(stage_dir):
        return {"job_id": job_id, "skipped": True, "reason": "stage already complete"}
    started = time.monotonic()
    stage_dir.mkdir(parents=True, exist_ok=True)

    ctx = _load_context(job_dir)
    info = probe(video)
    out_w = _even(int(info.width * scale))
    out_h = _even(int(info.height * scale))

    out_path = stage_dir / "annotated.mp4"
    tmp_path = stage_dir / "annotated.mp4.tmp"
    font = _font(max(12, int(16 * scale)))
    n_frames = 0
    with av.open(str(tmp_path), "w", format="mp4") as container:
        stream = container.add_stream("libx264", options={"crf": "23", "preset": "veryfast"})
        stream.width = out_w
        stream.height = out_h
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 1000)

        for frame in decode_frames(video, max_frames=max_frames):
            annotated = _draw_frame(frame.image, frame.frame_idx, frame.ts_ms, ctx, font)
            if scale != 1.0:
                annotated = annotated.resize((out_w, out_h), Image.Resampling.BILINEAR)
            video_frame = av.VideoFrame.from_ndarray(np.asarray(annotated), format="rgb24")
            video_frame.pts = frame.ts_ms
            container.mux(stream.encode(video_frame))
            n_frames += 1
        container.mux(stream.encode())  # flush
    tmp_path.rename(out_path)
    (stage_dir / "_SUCCESS").touch()

    elapsed = time.monotonic() - started
    return {
        "job_id": job_id,
        "frames": n_frames,
        "output": str(out_path),
        "bytes": out_path.stat().st_size,
        "wall_seconds": round(elapsed, 1),
    }


def _load_context(job_dir: Path) -> dict:
    """Everything drawing needs, grouped for per-frame lookup."""
    entities = {
        r["track_id"]: r for r in read_stage(job_dir / "entities").to_pylist()
    }
    jerseys: dict[int, tuple[str, float]] = {}
    for row in read_stage(job_dir / "identity").to_pylist():
        entity = entities.get(row["track_id"])
        if entity is None or row["prob"] < MIN_JERSEY_PROB:
            continue
        eid = entity["entity_id"]
        if eid not in jerseys or row["prob"] > jerseys[eid][1]:
            jerseys[eid] = (row["candidate"], row["prob"])

    by_frame: dict[int, list[dict]] = defaultdict(list)
    ball_track: list[tuple[int, float, float]] = []  # (ts_ms, cx_px, cy_px)
    for row in read_stage(job_dir / "localized").to_pylist():
        by_frame[row["frame_idx"]].append(row)
        if row["cls"] == int(DetClass.BALL):
            ball_track.append(
                (row["ts_ms"], (row["x1"] + row["x2"]) / 2, (row["y1"] + row["y2"]) / 2)
            )
    ball_track.sort(key=lambda t: t[0])

    possessions = (
        read_stage(job_dir / "possessions").to_pylist()
        if stage_complete(job_dir / "possessions")
        else []
    )
    fgm_events = sorted(
        (
            e for e in read_stage(job_dir / "box_events").to_pylist()
            if e["event_type"] == "FGM" and e["team_cluster"] in (0, 1)
        ),
        key=lambda e: e["ts_ms"],
    ) if stage_complete(job_dir / "box_events") else []

    return {
        "entities": entities,
        "jerseys": {eid: j for eid, (j, _) in jerseys.items()},
        "by_frame": by_frame,
        "ball_track": ball_track,
        "possessions": possessions,
        "fgm_events": fgm_events,
    }


def _draw_frame(
    image: np.ndarray, frame_idx: int, ts_ms: int, ctx: dict, font: ImageFont.ImageFont | ImageFont.FreeTypeFont
) -> Image.Image:
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)

    for row in ctx["by_frame"].get(frame_idx, ()):
        cls = row["cls"]
        box = (row["x1"], row["y1"], row["x2"], row["y2"])
        if cls == int(DetClass.BALL):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            r = max(6.0, (box[2] - box[0]) / 2)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=BALL_COLOR, width=3)
        elif cls == int(DetClass.RIM):
            draw.rectangle(box, outline=RIM_COLOR, width=3)
        elif cls == int(DetClass.REF):
            draw.rectangle(box, outline=REF_COLOR, width=2)
        else:
            entity = ctx["entities"].get(row["track_id"])
            if entity is None:
                draw.rectangle(box, outline=UNGROUPED_COLOR, width=1)
                continue
            color = TEAM_COLORS.get(entity["team_cluster"], UNGROUPED_COLOR)
            draw.rectangle(box, outline=color, width=3)
            jersey = ctx["jerseys"].get(entity["entity_id"])
            label = f"#{jersey}" if jersey else f"e{entity['entity_id']}"
            _label(draw, box[0], box[1], label, color, font)

    _ball_trail(draw, ctx["ball_track"], ts_ms)
    _banner(draw, ctx, ts_ms, font)
    _minimap(img, ctx["by_frame"].get(frame_idx, ()), ctx["entities"])
    return img


def _label(draw: ImageDraw.ImageDraw, x: float, y: float, text: str,
           color: tuple, font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> None:
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=color
    )
    draw.text((x, y), text, fill=(0, 0, 0), font=font)


def _ball_trail(draw: ImageDraw.ImageDraw, track: list, ts_ms: int) -> None:
    points = [(x, y) for t, x, y in track if ts_ms - TRAIL_MS <= t <= ts_ms]
    if len(points) >= 2:
        draw.line(points, fill=BALL_COLOR, width=2)


def _banner(draw: ImageDraw.ImageDraw, ctx: dict, ts_ms: int,
            font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> None:
    made = {0: 0, 1: 0}
    for e in ctx["fgm_events"]:
        if e["ts_ms"] > ts_ms:
            break
        made[e["team_cluster"]] += 1
    parts = [f"FGM  A {made[0]} - {made[1]} B"]
    for p in ctx["possessions"]:
        if p["ts_start_ms"] <= ts_ms <= p["ts_end_ms"]:
            cluster = possession_offense_cluster(p)
            side = "?" if cluster is None else {0: "A", 1: "B"}.get(cluster, "?")
            parts.append(f"possession {p['possession_id']} (team {side})")
            break
    _label(draw, 14, 12, "   ".join(parts), (20, 20, 20), font)


def _minimap(img: Image.Image, rows, entities: dict) -> None:
    map_w = int(NCAA_LENGTH_FT * MINIMAP_PX_PER_FT)
    map_h = int(NCAA_WIDTH_FT * MINIMAP_PX_PER_FT)
    x0 = MINIMAP_MARGIN_PX
    y0 = img.height - map_h - MINIMAP_MARGIN_PX

    overlay = Image.new("RGBA", (map_w, map_h), (15, 15, 15, 170))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle((0, 0, map_w - 1, map_h - 1), outline=(255, 255, 255, 200))
    odraw.line(
        (map_w / 2, 0, map_w / 2, map_h), fill=(255, 255, 255, 140), width=1
    )
    for row in rows:
        if row["court_x"] is None:
            continue
        px = row["court_x"] * MINIMAP_PX_PER_FT
        # court y grows toward the far sideline; image y grows down
        py = map_h - row["court_y"] * MINIMAP_PX_PER_FT
        if not (0 <= px < map_w and 0 <= py < map_h):
            continue
        if row["cls"] == int(DetClass.BALL):
            color, r = (*BALL_COLOR, 255), 3
        else:
            entity = entities.get(row["track_id"])
            if entity is None:
                continue
            color, r = (*TEAM_COLORS.get(entity["team_cluster"], UNGROUPED_COLOR), 255), 4
        odraw.ellipse((px - r, py - r, px + r, py + r), fill=color)
    img.paste(overlay, (x0, y0), overlay)


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old PIL fallback
        return ImageFont.load_default()


def _even(n: int) -> int:
    return n if n % 2 == 0 else n - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="smoke runs: stop after N frames")
    args = parser.parse_args()
    print(json.dumps(run(args.video, args.out, args.job_id,
                         scale=args.scale, max_frames=args.max_frames), indent=2))


if __name__ == "__main__":
    main()
