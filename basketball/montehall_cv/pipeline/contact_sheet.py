"""Contact-sheet jersey reads: aggregated-view VLM OCR (2026-07-10;
identity design §2 "input fusion", minus the
splat).

Per OCR-eligible tracklet, tile the best (legibility-gated when weights
given) torso crops into ONE grid image and ask the VLM for the jersey
number with the roster as vocabulary. One call sees every angle at once —
robustness per-frame OCR can't have. Verdicts are cached (VlmCache) so a
re-run never re-buys calls. Output artifact: contact_reads/ rows
(track_id, number|null, confidence, n_crops) + a summary comparing against
the per-frame identity posteriors already on disk.

Usage (spark cvbench):
  PYTHONPATH=/work/montehall-cv python -m montehall_cv.pipeline.contact_sheet \
    --video /work/testdata/07df788c/video.mp4 --out /work/out-v2 \
    --job-id 07df788c --roster-map /work/roster_map.json \
    --roster-job-id <full-id> [--legibility-weights ...]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image

from montehall_cv.pipeline.run_jersey import (
    _collect_crops,
    _crop_plan,
    _gate_legible,
)
from montehall_cv.roster import normalize_number, roster_entry, roster_number_set
from montehall_cv.store.artifacts import ArtifactWriter, read_stage, stage_complete
from montehall_cv.store.vlm_cache import VlmCache, content_key

MODEL = os.environ.get("MONTEHALL_CONTACT_SHEET_MODEL", "claude-sonnet-5")
CROPS_PER_SHEET = 12
CELL_PX = 128
MIN_ACCEPT_CONF = 0.6

CONTACT_READS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("track_id", pa.int32()),
        pa.field("number", pa.string(), nullable=True),
        pa.field("confidence", pa.float32()),
        pa.field("n_crops", pa.int16()),
    ]
)

PROMPT = (
    "This is a contact sheet: multiple crops of the SAME basketball player "
    "from different moments of one game. Read the jersey number on the "
    "player's shirt.{roster_line} Answer ONLY JSON: "
    '{{"number": "<digits or null>", "confidence": <0..1>}}. '
    "Use null when no number is clearly readable — never guess."
)
# ⚠ A stricter forced-abstention variant (visible-digit count first) was
# MEASURED WORSE on the 2026-08-24 HS-clip A/B: recall 1.0→.625,
# top-1 .25→.125, and the dominant '23' reads persisted (they are mostly
# real sightings — fragment duplication of two #23 players — not
# hallucination). Do not re-tighten without re-measuring; artifact:
# /work/models/quark-v1/shotsheet_hsclip_strict.json (spark).


def build_sheet(crops: list[np.ndarray]) -> bytes:
    """Letterboxed grid JPEG of up to CROPS_PER_SHEET crops."""
    crops = crops[:CROPS_PER_SHEET]
    cols = math.ceil(math.sqrt(len(crops)))
    rows = math.ceil(len(crops) / cols)
    sheet = Image.new("RGB", (cols * CELL_PX, rows * CELL_PX), (16, 16, 16))
    for i, crop in enumerate(crops):
        img = Image.fromarray(crop)
        img.thumbnail((CELL_PX, CELL_PX))
        x = (i % cols) * CELL_PX + (CELL_PX - img.width) // 2
        y = (i // cols) * CELL_PX + (CELL_PX - img.height) // 2
        sheet.paste(img, (x, y))
    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _read_sheet(client, sheet_jpeg: bytes, roster: set[str],
                cache: VlmCache) -> dict:
    roster_line = (
        f" The team's roster numbers are: {sorted(roster, key=int)}."
        if roster else ""
    )
    prompt = PROMPT.format(roster_line=roster_line)
    key = content_key(MODEL, prompt, sheet_jpeg)
    hit = cache.get(key)
    if hit is not None:
        return hit
    import base64

    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": base64.b64encode(sheet_jpeg).decode()}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = response.content[0].text
    try:
        parsed = json.loads(text[text.index("{"): text.rindex("}") + 1])
        number = parsed.get("number")
        number = normalize_number(str(number)) if number not in (None, "null") else None
        verdict = {"number": number, "confidence": float(parsed.get("confidence") or 0.0)}
    except (ValueError, KeyError, TypeError):
        verdict = {"number": None, "confidence": 0.0}
    cache.put(key, verdict)
    return verdict


def run(video: Path, out_root: Path, job_id: str,
        roster_map: Path | None = None, roster_job_id: str | None = None,
        legibility_weights: Path | None = None) -> dict:
    job_dir = out_root / job_id
    started = time.monotonic()

    from montehall_cv.pipeline import run_team_assoc as rta

    roster: set[str] = set()
    if roster_map is not None:
        entry = roster_entry(json.loads(roster_map.read_text()),
                             roster_job_id or job_id)
        if entry:
            roster = roster_number_set(entry["players"])

    eligible = set(rta._eligible_tracklets(job_dir))
    plan = _crop_plan(job_dir, eligible, CROPS_PER_SHEET * 2)
    crops, meta = _collect_crops(video, plan)
    if legibility_weights is not None and crops:
        crops, meta = _gate_legible(crops, meta, legibility_weights)

    by_track: dict[int, list[np.ndarray]] = {}
    for crop, (tid, _) in zip(crops, meta, strict=True):
        by_track.setdefault(tid, []).append(crop)

    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY") or Path(
        "/root/.anthropic_key").read_text().strip()
    client = anthropic.Anthropic(api_key=api_key)
    cache = VlmCache(job_dir / "vlm_cache.jsonl")

    writer = ArtifactWriter(job_dir / "contact_reads", CONTACT_READS_SCHEMA)
    accepted = 0
    roster_legal = 0
    for tid, tid_crops in sorted(by_track.items()):
        verdict = _read_sheet(client, build_sheet(tid_crops), roster, cache)
        ok = verdict["number"] is not None and verdict["confidence"] >= MIN_ACCEPT_CONF
        accepted += int(ok)
        roster_legal += int(bool(ok and roster and verdict["number"] in roster))
        writer.add({
            "job_id": job_id, "track_id": tid,
            "number": verdict["number"] if ok else None,
            "confidence": verdict["confidence"], "n_crops": len(tid_crops),
        })
    writer.close()

    # zero-shot per-frame baseline from the identity artifact on disk
    baseline_tracks = 0
    if stage_complete(job_dir / "identity"):
        best: dict[int, float] = {}
        for row in read_stage(job_dir / "identity").to_pylist():
            best[row["track_id"]] = max(best.get(row["track_id"], 0.0), row["prob"])
        baseline_tracks = sum(1 for p in best.values() if p >= MIN_ACCEPT_CONF)

    return {
        "job_id": job_id,
        "tracklets_sheeted": len(by_track),
        "contact_accepted": accepted,
        "contact_roster_legal": roster_legal,
        "perframe_baseline_tracks": baseline_tracks,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--roster-map", type=Path, default=None)
    parser.add_argument("--roster-job-id", default=None)
    parser.add_argument("--legibility-weights", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.video, args.out, args.job_id,
                         roster_map=args.roster_map,
                         roster_job_id=args.roster_job_id,
                         legibility_weights=args.legibility_weights), indent=2))


if __name__ == "__main__":
    main()
