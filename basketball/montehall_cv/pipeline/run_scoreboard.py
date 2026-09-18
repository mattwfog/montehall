"""Scoreboard oracle: the in-frame score digits are the make/miss authority.

Same 3+2+1 shape as the rim stage: a VLM locates the board once per video,
cheap local perception reads it everywhere (lit-LED mask flags candidate
score changes, PARSeq reads the digits around each), and hard rules
validate (scores only step by +1/+2/+3, never decrease, chain-consistent).
Validated score changes land in <out>/<job_id>/score_events/ and
run_boxscore reconciles shot windows against them — a board-confirmed
delta overrides the frame-judged made/missed verdict, and the delta value
classifies FT (+1) vs two (+2) vs three (+3).

Fail-open: no visible board (or no API key for the locate) writes an
empty complete stage and downstream keeps the VLM verdicts.

Output: <out>/<job_id>/score_events/ + score_events/board.json (meta)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from montehall_cv.pipeline.video import decode_frames, probe, sample_frames_at
from montehall_cv.store.artifacts import ArtifactWriter, stage_complete
from montehall_cv.store.vlm_cache import VlmCache, content_key

SCORE_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("event_id", pa.int32()),
        pa.field("side", pa.string()),  # home | guest
        pa.field("ts_ms", pa.int64()),  # change detected on the board
        pa.field("before_val", pa.int16()),
        pa.field("after_val", pa.int16()),
        pa.field("delta", pa.int16()),
        pa.field("read_conf", pa.float32()),
    ]
)

VLM_MODEL = "claude-sonnet-5"
BOARD_PROMPT = (
    "This is a frame of basketball game film. Find the scoreboard showing "
    "the two team scores (an LED wall board or an overlay graphic). Respond "
    "ONLY with JSON, coordinates normalized 0-1 relative to this image: "
    '{"found": true|false, "board_box": [x1,y1,x2,y2]}. The box should '
    "contain the whole scoreboard."
)
DIGITS_PROMPT = (
    "This is a cropped basketball scoreboard. Locate the two team SCORE "
    "digit groups — the running point totals, NOT the clock, period, fouls, "
    "timeouts or player-foul numbers. Respond ONLY with JSON, coordinates "
    "normalized 0-1 relative to THIS cropped image: "
    '{"home_score_box": [x1,y1,x2,y2], "guest_score_box": [x1,y1,x2,y2]}. '
    "Each box must tightly contain just that side's score digits."
)

DETECT_STEP = 10        # sample every Nth frame for change detection
LIT_R = 150             # lit-LED mask: R floor
LIT_R_MINUS_B = 60      # lit-LED mask: R-B floor (LED digits are red/amber)
CHANGE_FRAC = 0.10      # xor-changed lit pixels / previous lit pixels
CHANGE_MIN_PX = 30
MERGE_WINDOW_S = 2.5    # candidates on one side within this merge
MAX_CANDIDATES = 250    # noise guard per video
READ_OFFSETS_S = (-2.0, 2.0)
BOX_EXPAND = 0.5        # refine search region around the VLM's guess
STABLE_LIT_FRAC = 0.3   # pixel is board structure if lit in >30% of samples
VALID_DELTAS = (1, 2, 3)


def run(video: Path, out_root: Path, job_id: str) -> dict:
    job_dir = out_root / job_id
    if stage_complete(job_dir / "score_events"):
        return {"job_id": job_id, "skipped": True, "reason": "stage already complete"}
    started = time.monotonic()

    boxes, first_ts, last_ts = _locate(video, job_dir)
    if boxes is None:
        _write(job_dir, job_id, [], {"found": False, "reason": "no board located"})
        return {"job_id": job_id, "board_found": False,
                "wall_seconds": round(time.monotonic() - started, 1)}

    reader = _BoardReader(video, boxes)
    sane, reason = _sanity(reader, first_ts, last_ts)
    if not sane:
        reader.close()
        _write(job_dir, job_id, [], {"found": False, "reason": f"sanity: {reason}"})
        return {"job_id": job_id, "board_found": False, "sanity": reason,
                "wall_seconds": round(time.monotonic() - started, 1)}

    candidates = _detect_changes(video, boxes)
    events = _read_and_validate(reader, candidates)
    reader.close()
    meta = {
        "found": True,
        "boxes": {k: [int(v) for v in b] for k, b in boxes.items()},
        "coverage_start_ms": int(first_ts * 1000),
        "coverage_end_ms": int(last_ts * 1000),
        "candidates": len(candidates),
    }
    _write(job_dir, job_id, events, meta)
    return {
        "job_id": job_id,
        "board_found": True,
        "candidates": len(candidates),
        "score_events": len(events),
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def _ask(client, cache: VlmCache, image, prompt: str) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    data = base64.standard_b64encode(buf.getvalue()).decode()
    key = content_key(VLM_MODEL, prompt, data)
    verdict = cache.get(key)
    if verdict is not None:
        return verdict
    response = client.messages.create(
        model=VLM_MODEL, max_tokens=300,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
            {"type": "text", "text": prompt},
        ]}],
    )
    text = next((b.text for b in response.content if getattr(b, "text", None)), "")
    try:
        verdict = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        verdict = {"found": False}
    cache.put(key, verdict)
    return verdict


def _locate(video: Path, job_dir: Path):
    """Two-round VLM locate: board box on the full frame, then score-digit
    boxes on the zoomed board crop (the VLM's box precision on a full 1080p
    frame is not digit-tight — 2026-07-12: the one-shot variant latched the
    fouls row and a baseline stripe). Boxes are refined against the
    temporally-stable lit-LED footprint; run() sanity-gates the result."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None, 0.0, 0.0
    import anthropic
    from PIL import Image

    client = anthropic.Anthropic(api_key=api_key)
    cache = VlmCache(job_dir / "_vlm_cache" / "score_events.jsonl")

    # Never buffer a decode pass here: a full broadcast at every_n=10 is
    # tens of GB of RGB (the 72-79GB OOM, 2026-07-13). Span comes from
    # container metadata; every frame this stage looks at is seek-fetched.
    first_ts, last_ts = _span(video)
    if last_ts <= first_ts:
        return None, 0.0, 0.0
    span = last_ts - first_ts

    probe_frames = list(sample_frames_at(
        video, [first_ts + span * f for f in (0.25, 0.5, 0.75)]
    ))
    if not probe_frames:
        return None, first_ts, last_ts
    h, w = probe_frames[0].image.shape[:2]

    votes: dict[str, list[list[float]]] = {"home": [], "guest": []}
    for probe_frame in probe_frames:
        frame_img = probe_frame.image
        board = _ask(client, cache, Image.fromarray(frame_img), BOARD_PROMPT)
        box = board.get("board_box")
        if not board.get("found") or not (isinstance(box, list) and len(box) == 4):
            continue
        bx1, by1 = max(int(box[0] * w), 0), max(int(box[1] * h), 0)
        bx2, by2 = min(int(box[2] * w), w), min(int(box[3] * h), h)
        if bx2 - bx1 < 40 or by2 - by1 < 30:
            continue
        crop = Image.fromarray(frame_img[by1:by2, bx1:bx2])
        crop = crop.resize((crop.width * 2, crop.height * 2))
        digits = _ask(client, cache, crop, DIGITS_PROMPT)
        for side, field in (("home", "home_score_box"), ("guest", "guest_score_box")):
            db = digits.get(field)
            if isinstance(db, list) and len(db) == 4:
                votes[side].append([
                    bx1 + float(db[0]) * (bx2 - bx1), by1 + float(db[1]) * (by2 - by1),
                    bx1 + float(db[2]) * (bx2 - bx1), by1 + float(db[3]) * (by2 - by1),
                ])

    if not votes["home"] or not votes["guest"]:
        return None, first_ts, last_ts

    boxes = {}
    for side in ("home", "guest"):
        px = np.median(np.array(votes[side]), axis=0).tolist()
        boxes[side] = _refine(video, first_ts, last_ts, px, w, h)
    return boxes, first_ts, last_ts


def _span(video: Path) -> tuple[float, float]:
    """Coverage span in seconds: metadata duration, ts-only scan fallback.

    The fallback decodes but never retains frames — memory-flat on any
    length of video."""
    info = probe(video)
    if info.duration_ms:
        return 0.0, info.duration_ms / 1000.0
    first_ts = last_ts = 0.0
    seen = False
    for frame in decode_frames(video, every_n=DETECT_STEP):
        ts = frame.ts_ms / 1000.0
        if not seen:
            first_ts, seen = ts, True
        last_ts = ts
    return first_ts, last_ts


REFINE_SAMPLES = 40


def _refine(video, first_ts, last_ts, box, w, h):
    """Snap a coarse digit box to the temporally-stable lit footprint.

    Seek-fetches REFINE_SAMPLES frames spread over the span and crops each
    immediately — peak memory is one frame plus the crop accumulator."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(int(x1 - bw * BOX_EXPAND), 0)
    x2 = min(int(x2 + bw * BOX_EXPAND), w)
    y1 = max(int(y1 - bh * BOX_EXPAND), 0)
    y2 = min(int(y2 + bh * BOX_EXPAND), h)
    span = last_ts - first_ts
    times = [
        first_ts + span * (i + 0.5) / REFINE_SAMPLES for i in range(REFINE_SAMPLES)
    ]
    acc = np.zeros((y2 - y1, x2 - x1), dtype=np.int32)
    n = 0
    for frame in sample_frames_at(video, times):
        img = frame.image
        if img.shape[0] < y2 or img.shape[1] < x2:
            continue
        acc += _lit(img[y1:y2, x1:x2])
        n += 1
    if n == 0:
        return [x1, y1, x2, y2]
    stable = acc >= max(int(n * STABLE_LIT_FRAC), 1)
    ys, xs = np.nonzero(stable)
    if len(xs) < 20:
        return [x1, y1, x2, y2]
    pad = 4
    return [max(x1 + int(xs.min()) - pad, 0), max(y1 + int(ys.min()) - pad, 0),
            min(x1 + int(xs.max()) + pad, w), min(y1 + int(ys.max()) + pad, h)]


def _lit(box: np.ndarray) -> np.ndarray:
    r = box[:, :, 0].astype(np.int16)
    b = box[:, :, 2].astype(np.int16)
    return ((r > LIT_R) & (r - b > LIT_R_MINUS_B)).astype(np.uint8)


def _detect_changes(video: Path, boxes) -> list[tuple[float, str]]:
    prev: dict[str, np.ndarray] = {}
    raw: list[tuple[float, str]] = []
    for frame in decode_frames(video, every_n=DETECT_STEP):
        for side, (x1, y1, x2, y2) in boxes.items():
            m = _lit(frame.image[y1:y2, x1:x2])
            pm = prev.get(side)
            if pm is not None:
                changed = int(np.logical_xor(m, pm).sum())
                if changed > CHANGE_MIN_PX and changed / max(int(pm.sum()), 40) > CHANGE_FRAC:
                    raw.append((frame.ts_ms / 1000.0, side))
            prev[side] = m
    merged: list[tuple[float, str]] = []
    for t, side in raw:
        if any(s == side and t - mt < MERGE_WINDOW_S for mt, s in merged[-4:]):
            continue
        merged.append((t, side))
    return merged[:MAX_CANDIDATES]


# LED seven-segment confusions PARSeq makes on board digits (8<->B is the
# classic); scores are digits by construction, so map before rejecting.
_LED_CONFUSIONS = str.maketrans("BODIlSZGbgqAT", "8001152669947")
SCORE_MIN_CHAR_CONF = 0.35  # chain rules + persistence voting carry safety,
# not per-char confidence — the jersey _accept bars reject clean board reads


class _BoardReader:
    """Seek-and-OCR the score digit boxes at arbitrary times.

    A read is a majority vote over crops; two-batch reads (persist=True)
    sample again ~2s later and require agreement across batches — a
    translucent board with players bleeding through misreads single
    frames, not two time-separated batches the same way."""

    def __init__(self, video: Path, boxes: dict) -> None:
        import av

        from montehall_cv.pipeline.jersey import JerseyOcr

        self._boxes = boxes
        self._ocr = JerseyOcr(device="cpu", batch_size=16)
        self._container = av.open(str(video))
        self._stream = self._container.streams.video[0]

    def _crops(self, side: str, t: float, n: int = 3) -> list[np.ndarray]:
        x1, y1, x2, y2 = self._boxes[side]
        self._container.seek(int(max(t, 0.0) / self._stream.time_base), stream=self._stream)
        crops = []
        for frame in self._container.decode(self._stream):
            if frame.time is not None and frame.time >= t - 0.02:
                img = frame.to_ndarray(format="rgb24")
                crops.append(np.ascontiguousarray(img[y1:y2, x1:x2]))
                if len(crops) == n:
                    break
        return crops

    def _decode(self, crops: list[np.ndarray]) -> list[tuple[int, float]]:
        if not crops:
            return []
        torch = self._ocr._torch
        batch = torch.stack([self._ocr._preprocess(c) for c in crops])
        with torch.no_grad():
            probs = self._ocr._model(batch).softmax(-1)
        labels, confs = self._ocr._model.tokenizer.decode(probs)
        out = []
        for label, char_confs in zip(labels, confs):
            text = label.strip().translate(_LED_CONFUSIONS)
            if not text.isdigit() or not 0 <= int(text) <= 200:
                continue
            conf = min((float(x) for x in char_confs), default=0.0)
            if conf < SCORE_MIN_CHAR_CONF:
                continue
            out.append((int(text), conf))
        return out

    def read_at(self, side: str, t: float, persist: bool = False) -> tuple[int | None, float]:
        reads = self._decode(self._crops(side, t))
        if persist:
            reads += self._decode(self._crops(side, t + 1.8))
        need = 4 if persist else 2
        by_val: dict[int, list[float]] = {}
        for val, conf in reads:
            by_val.setdefault(val, []).append(conf)
        best = max(by_val.items(), key=lambda kv: len(kv[1]), default=None)
        if best is None or len(best[1]) < need:
            return None, 0.0
        return best[0], max(best[1])

    def close(self) -> None:
        self._container.close()


def _sanity(reader: _BoardReader, first_ts: float, last_ts: float) -> tuple[bool, str]:
    """Located boxes must read as plausible scores: parseable at well-
    separated times and non-decreasing per side. A wrong box (fouls row,
    clock, a red stripe) fails here and the stage fails open."""
    span = last_ts - first_ts
    for side in ("home", "guest"):
        vals = []
        for frac in (0.2, 0.5, 0.8):
            v, _ = reader.read_at(side, first_ts + span * frac)
            if v is not None:
                vals.append(v)
        if len(vals) < 2:
            return False, f"{side} digits unreadable"
        if any(b < a for a, b in zip(vals, vals[1:])):
            return False, f"{side} reads decrease over time: {vals}"
    return True, ""


def _read_and_validate(reader: _BoardReader, candidates) -> list[dict]:
    """PARSeq reads around each candidate; chain rules keep real steps."""
    events: list[dict] = []
    running: dict[str, int | None] = {"home": None, "guest": None}
    for t, side in candidates:
        before, conf_b = reader.read_at(side, t + READ_OFFSETS_S[0])
        after, conf_a = reader.read_at(side, t + READ_OFFSETS_S[1], persist=True)
        if before is None or after is None:
            continue
        delta = after - before
        if delta not in VALID_DELTAS:
            continue
        last = running[side]
        if last is not None and before < last:
            continue  # OCR noise contradicting the running score
        running[side] = after
        events.append({
            "side": side,
            "ts_ms": int(t * 1000),
            "before_val": before,
            "after_val": after,
            "delta": delta,
            "read_conf": min(conf_b, conf_a),
        })
    return events


def _write(job_dir: Path, job_id: str, events: list[dict], meta: dict) -> None:
    writer = ArtifactWriter(job_dir / "score_events", SCORE_EVENTS_SCHEMA)
    for i, ev in enumerate(sorted(events, key=lambda e: e["ts_ms"])):
        writer.add({"job_id": job_id, "event_id": i, **ev})
    writer.close()
    (job_dir / "score_events" / "board.json").write_text(json.dumps(meta))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.video, args.out, args.job_id), indent=2))


if __name__ == "__main__":
    main()
