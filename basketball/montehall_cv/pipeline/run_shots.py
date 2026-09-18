"""Shot-event stage: ball-near-rim windows -> VLM made/miss adjudication.

Perception proposes (ball detections entering an expanded rim region,
grouped into candidate windows); the adjudication tier grades each window
from a short frame sequence cropped around the rim. Verdicts land as event
rows with rationale — the audit trail for confidence-gated stats.

Output: <out>/<job_id>/shot_events/
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa

from montehall_cv.pipeline.video import decode_frames
from montehall_cv.store.artifacts import ArtifactWriter, read_stage, stage_complete
from montehall_cv.store.records import DetClass
from montehall_cv.store.vlm_cache import VlmCache, content_key

SHOT_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("event_id", pa.int32()),
        pa.field("frame_start", pa.int32()),
        pa.field("frame_end", pa.int32()),
        pa.field("ts_start_ms", pa.int64()),
        pa.field("ts_end_ms", pa.int64()),
        pa.field("court_end", pa.string(), nullable=True),
        pa.field("n_ball_obs", pa.int32()),
        pa.field("verdict_attempt", pa.bool_(), nullable=True),
        pa.field("verdict_made", pa.bool_(), nullable=True),
        pa.field("confidence", pa.float32(), nullable=True),
        pa.field("rationale", pa.string(), nullable=True),
    ]
)

REGION_SCALE_X = 2.5
REGION_SCALE_UP = 2.0
REGION_SCALE_DOWN = 4.0
MERGE_GAP_MS = 2500  # time-based: frame-gap merging halves at 60fps (AV1
# broadcasts) and court_end flapping on a moving camera splits one rim event
# into stutter windows — Clemson-Duke 2026-07-14: 823 windows / 348 real
# bursts / 150 true shots. Real consecutive-shot pairs 2-3s apart are 1-3
# per game (putbacks); same-clock FT pairs share one rim sequence anyway.
MAX_WINDOW_MS = 6000  # chain cap: a rebound scramble must not snowball
FRAMES_PER_WINDOW = 5  # floor; long merged windows sample denser (below)
FRAMES_PER_WINDOW_MAX = 12
FRAME_SAMPLE_EVERY_MS = 600  # ~one sampled frame per 0.6s of window span
WINDOW_PAD_FRAMES = 8
RIM_MATCH_FRAMES = 5  # detector rim boxes matched to a ball obs within +-N frames
END_INHERIT_FRAMES = 300  # projection-less rim inherits its end from a projected rim within +-N frames
END_INHERIT_DIST_SCALE = 2.0  # ... whose centre lies within this many rim-widths in pixel space
COURT_MID_X_FT = 47.0
VLM_MODEL = "claude-sonnet-5"

PROMPT = (
    "These frames are a time-ordered sequence cropped around a basketball rim "
    "during game film. Decide: (1) does a genuine field-goal attempt occur "
    "(ball shot toward this basket), and (2) if so, is it MADE (ball passes "
    "down through the rim/net)? Tips, putbacks and blocked shots count as "
    "attempts. Passes, dribbles or the ball merely passing near the rim do "
    "not. Respond ONLY with JSON: "
    '{"attempt": true|false, "made": true|false|null, '
    '"confidence": 0.0-1.0, "rationale": "<one sentence>"}'
)


def run(video: Path, out_root: Path, job_id: str) -> dict:
    job_dir = out_root / job_id
    if stage_complete(job_dir / "shot_events"):
        return {"job_id": job_id, "skipped": True, "reason": "stage already complete"}
    started = time.monotonic()

    windows, window_source = _candidate_windows(job_dir)
    frames_needed = _frame_plan(windows)
    frames = _collect(video, frames_needed)

    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)

    cache = VlmCache(job_dir / "_vlm_cache" / "shot_events.jsonl")
    writer = ArtifactWriter(job_dir / "shot_events", SHOT_EVENTS_SCHEMA)
    n_attempts = 0
    n_made = 0
    for event_id, window in enumerate(windows):
        verdict = _adjudicate(client, frames, window, cache=cache)
        if verdict.get("attempt"):
            n_attempts += 1
            if verdict.get("made"):
                n_made += 1
        writer.add(
            {
                "job_id": job_id,
                "event_id": event_id,
                "frame_start": window["frame_start"],
                "frame_end": window["frame_end"],
                "ts_start_ms": window["ts_start_ms"],
                "ts_end_ms": window["ts_end_ms"],
                "court_end": window["court_end"],
                "n_ball_obs": window["n_ball_obs"],
                "verdict_attempt": verdict.get("attempt"),
                "verdict_made": verdict.get("made"),
                "confidence": verdict.get("confidence"),
                "rationale": verdict.get("rationale"),
            }
        )
    writer.close()

    return {
        "job_id": job_id,
        "candidate_windows": len(windows),
        "window_source": window_source,
        "attempts": n_attempts,
        "made": n_made,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def _rims_by_segment(job_dir: Path) -> list[dict]:
    return [r for r in read_stage(job_dir / "rims").to_pylist() if r["validated"]]


def _candidate_windows(job_dir: Path) -> tuple[list[dict], str]:
    """(candidate windows, source). Detector rim boxes win when present.

    The fine-tuned detector sees the rim in ~90% of frames across every
    camera segment; the VLM rim-locate validates far fewer (Clemson
    2026-07-09: 1 rim over 4 segments -> 2 windows on a clip the legacy
    pipeline's output claims 16 FGA for). Stock-detector runs (no RIM class) fall
    back to the VLM rims stage."""
    balls, rims_by_frame = _localized_split(job_dir)
    if rims_by_frame:
        return _merge_hits(_detector_hits(balls, rims_by_frame)), "detector_rims"
    return _merge_hits(_vlm_hits(balls, _rims_by_segment(job_dir))), "vlm_rims"


def _localized_split(job_dir: Path) -> tuple[list[dict], dict[int, list[dict]]]:
    balls: list[dict] = []
    rims_by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in read_stage(job_dir / "localized").to_pylist():
        if row["cls"] == int(DetClass.BALL):
            balls.append(row)
        elif row["cls"] == int(DetClass.RIM):
            rims_by_frame[row["frame_idx"]].append(row)
    return balls, dict(rims_by_frame)


def _region_around(rim_box: dict, court_end: str | None) -> dict:
    w = rim_box["x2"] - rim_box["x1"]
    h = rim_box["y2"] - rim_box["y1"]
    cx = (rim_box["x1"] + rim_box["x2"]) / 2
    return {
        "x1": cx - w * REGION_SCALE_X / 2,
        "x2": cx + w * REGION_SCALE_X / 2,
        "y1": rim_box["y1"] - h * REGION_SCALE_UP,
        "y2": rim_box["y2"] + h * REGION_SCALE_DOWN,
        "court_end": court_end,
    }


def _rim_ends(rims_by_frame: dict[int, list[dict]]) -> dict[tuple[int, int], str | None]:
    """(frame_idx, idx-in-frame) -> court end for every rim detection.

    A rim with a court projection gets its own end. One without inherits
    from the nearest-in-time projected rim whose pixel centre sits within
    END_INHERIT_DIST_SCALE rim-widths — the same physical hoop seen in a
    frame where the homography held. No neighbour -> None: the window is
    still adjudicated; only end-dependent attribution degrades. (Requiring
    a projection to *trigger* windows silently dropped 5 of 8 real attempts
    on the 07df788c benchmark, 2026-07-12 — 79% of rims in one 30s stretch
    had no projection.)"""
    projected: list[tuple[int, float, float, str]] = []
    for frame_idx, rims in rims_by_frame.items():
        for rim in rims:
            if rim["court_x"] is not None:
                end = "left" if rim["court_x"] < COURT_MID_X_FT else "right"
                cx = (rim["x1"] + rim["x2"]) / 2
                cy = (rim["y1"] + rim["y2"]) / 2
                projected.append((frame_idx, cx, cy, end))
    projected.sort(key=lambda p: p[0])
    frames = [p[0] for p in projected]

    ends: dict[tuple[int, int], str | None] = {}
    for frame_idx, rims in rims_by_frame.items():
        for i, rim in enumerate(rims):
            if rim["court_x"] is not None:
                ends[(frame_idx, i)] = "left" if rim["court_x"] < COURT_MID_X_FT else "right"
                continue
            cx = (rim["x1"] + rim["x2"]) / 2
            cy = (rim["y1"] + rim["y2"]) / 2
            max_d = END_INHERIT_DIST_SCALE * max(rim["x2"] - rim["x1"], 1.0)
            best: tuple[int, str] | None = None
            lo = bisect_left(frames, frame_idx - END_INHERIT_FRAMES)
            hi = bisect_right(frames, frame_idx + END_INHERIT_FRAMES)
            for pf, px, py, pend in projected[lo:hi]:
                if math.hypot(cx - px, cy - py) > max_d:
                    continue
                gap = abs(pf - frame_idx)
                if best is None or gap < best[0]:
                    best = (gap, pend)
            ends[(frame_idx, i)] = best[1] if best else None
    return ends


def _detector_hits(
    balls: list[dict], rims_by_frame: dict[int, list[dict]]
) -> list[tuple[int, int, str | None, dict]]:
    """Ball obs inside a scaled detector rim box within +-RIM_MATCH_FRAMES.

    court_end from the rim's court projection (a ground-plane homography
    distorts an elevated rim's depth but not which half it lands in),
    inherited or None when the projection is missing — never a reason to
    drop the window."""
    ends = _rim_ends(rims_by_frame)
    hits = []
    for ball in balls:
        bx = (ball["x1"] + ball["x2"]) / 2
        by = (ball["y1"] + ball["y2"]) / 2
        for offset in sorted(range(-RIM_MATCH_FRAMES, RIM_MATCH_FRAMES + 1), key=abs):
            matched = False
            rim_frame = ball["frame_idx"] + offset
            for i, rim in enumerate(rims_by_frame.get(rim_frame, ())):
                end = ends[(rim_frame, i)]
                region = _region_around(rim, end)
                if region["x1"] <= bx <= region["x2"] and region["y1"] <= by <= region["y2"]:
                    hits.append((ball["frame_idx"], ball["ts_ms"], end, region))
                    matched = True
                    break
            if matched:
                break
    return hits


def _vlm_hits(
    balls: list[dict], rims: list[dict]
) -> list[tuple[int, int, str, dict]]:
    """Legacy path: ball obs inside a VLM-validated rim segment region."""
    regions = []
    for rim in rims:
        region = _region_around(rim, rim["court_end"])
        region["frame_start"] = rim["frame_start"]
        region["frame_end"] = rim["frame_end"]
        regions.append(region)

    hits = []
    for row in balls:
        bx = (row["x1"] + row["x2"]) / 2
        by = (row["y1"] + row["y2"]) / 2
        for region in regions:
            if (
                region["frame_start"] <= row["frame_idx"] <= region["frame_end"]
                and region["x1"] <= bx <= region["x2"]
                and region["y1"] <= by <= region["y2"]
            ):
                hits.append((row["frame_idx"], row["ts_ms"], region["court_end"], region))
                break
    return hits


def _ends_compatible(a: str | None, b: str | None) -> bool:
    """One rim event, one window: a None projection (homography dropout on a
    moving camera) must not split the burst it sits inside."""
    return a is None or b is None or a == b


def _union_region(a: dict, b: dict) -> dict:
    return {
        "x1": min(a["x1"], b["x1"]),
        "y1": min(a["y1"], b["y1"]),
        "x2": max(a["x2"], b["x2"]),
        "y2": max(a["y2"], b["y2"]),
    }


def _merge_hits(hits: Sequence[tuple[int, int, str | None, dict]]) -> list[dict]:
    windows: list[dict] = []
    for frame_idx, ts_ms, court_end, region in sorted(hits, key=lambda t: t[0]):
        last = windows[-1] if windows else None
        if (
            last is not None
            and _ends_compatible(last["court_end"], court_end)
            and ts_ms - last["ts_end_ms"] <= MERGE_GAP_MS
            and ts_ms - last["ts_start_ms"] <= MAX_WINDOW_MS
        ):
            last["frame_end"] = frame_idx
            last["ts_end_ms"] = ts_ms
            last["n_ball_obs"] += 1
            last["region"] = _union_region(last["region"], region)
            if last["court_end"] is None:
                last["court_end"] = court_end
        else:
            windows.append(
                {
                    "frame_start": frame_idx,
                    "frame_end": frame_idx,
                    "ts_start_ms": ts_ms,
                    "ts_end_ms": ts_ms,
                    "court_end": court_end,
                    "n_ball_obs": 1,
                    "region": dict(region),
                }
            )
    return windows


def _frames_for_span(span_ms: int) -> int:
    """Sample density follows window duration: 5 frames over a 0.2s stutter
    window is dense; 5 over a 6s merged burst misses the release/rim moment
    (the 2026-07-14 attempt-recall drop). One frame per ~0.6s, capped."""
    by_span = span_ms // FRAME_SAMPLE_EVERY_MS + 1
    return int(min(max(FRAMES_PER_WINDOW, by_span), FRAMES_PER_WINDOW_MAX))


def _frame_plan(windows: list[dict]) -> dict[int, list[int]]:
    plan: dict[int, list[int]] = defaultdict(list)
    for i, window in enumerate(windows):
        start = max(window["frame_start"] - WINDOW_PAD_FRAMES, 0)
        end = window["frame_end"] + WINDOW_PAD_FRAMES
        n = _frames_for_span(window["ts_end_ms"] - window["ts_start_ms"])
        picks = np.linspace(start, end, n).astype(int)
        window["sample_frames"] = sorted(set(picks.tolist()))
        for f in window["sample_frames"]:
            plan[f].append(i)
    return plan


def _collect(video: Path, plan: dict[int, list[int]]) -> dict[int, np.ndarray]:
    frames: dict[int, np.ndarray] = {}
    remaining = set(plan)
    for frame in decode_frames(video):
        if frame.frame_idx in remaining:
            frames[frame.frame_idx] = frame.image
            remaining.discard(frame.frame_idx)
            if not remaining:
                break
    return frames


def _adjudicate(client, frames: dict[int, np.ndarray], window: dict, cache: VlmCache | None = None) -> dict:
    from PIL import Image

    region = window["region"]
    content = []
    for frame_idx in window["sample_frames"]:
        image = frames.get(frame_idx)
        if image is None:
            continue
        h, w = image.shape[:2]
        x1, y1 = max(int(region["x1"]), 0), max(int(region["y1"]), 0)
        x2, y2 = min(int(region["x2"]), w), min(int(region["y2"]), h)
        if x2 - x1 < 16 or y2 - y1 < 16:
            continue
        buf = io.BytesIO()
        Image.fromarray(image[y1:y2, x1:x2]).save(buf, format="JPEG", quality=88)
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(buf.getvalue()).decode(),
                },
            }
        )
    if not content:
        return {}

    key = content_key(VLM_MODEL, PROMPT, *(c["source"]["data"] for c in content))
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit

    content.append({"type": "text", "text": PROMPT})
    response = client.messages.create(
        model=VLM_MODEL, max_tokens=300, messages=[{"role": "user", "content": content}]
    )
    text = next((b.text for b in response.content if hasattr(b, "text")), "")
    verdict: dict = {}
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            verdict = parsed
    if cache is not None:
        cache.put(key, verdict)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.video, args.out, args.job_id), indent=2))


if __name__ == "__main__":
    main()
