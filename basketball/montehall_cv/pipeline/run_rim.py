"""Rim stage: camera segments -> VLM rim location -> geometric validation.

Output: <out>/<job_id>/rims/ — one row per (segment, rim candidate), with
court_end + validated set where the court homography corroborates the VLM.
Validated rows are future detector training labels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from montehall_cv.pipeline.rim import (
    MIN_SEGMENT_FRAMES,
    RimLocator,
    detect_segments,
    validate_rim,
)
from montehall_cv.pipeline.video import decode_frames, probe
from montehall_cv.store.artifacts import ArtifactWriter, read_stage, stage_complete
from montehall_cv.store.vlm_cache import VlmCache

RIMS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("segment_id", pa.int32()),
        pa.field("frame_start", pa.int32()),
        pa.field("frame_end", pa.int32()),
        pa.field("rep_frame_idx", pa.int32()),
        pa.field("x1", pa.float32()),
        pa.field("y1", pa.float32()),
        pa.field("x2", pa.float32()),
        pa.field("y2", pa.float32()),
        pa.field("vlm_conf", pa.float32()),
        pa.field("court_end", pa.string(), nullable=True),
        pa.field("validated", pa.bool_()),
    ]
)

SAMPLE_EVERY = 5


def run(video: Path, out_root: Path, job_id: str) -> dict:
    job_dir = out_root / job_id
    if stage_complete(job_dir / "rims"):
        return {"job_id": job_id, "skipped": True, "reason": "stage already complete"}
    started = time.monotonic()
    info = probe(video)

    # Pass 1: stream the sampled frames to camera segments. Only thumbnails are
    # compared (see detect_segments), so nothing full-resolution is retained —
    # a 2h game is ~43k sampled 1080p frames (~270GB) if materialized here.
    sampled_indices: list[int] = []

    def _stream_samples():
        for f in decode_frames(video, every_n=SAMPLE_EVERY):
            sampled_indices.append(f.frame_idx)
            yield f.frame_idx, f.image

    segments = [
        (s, e) for s, e in detect_segments(_stream_samples()) if e - s >= MIN_SEGMENT_FRAMES
    ]
    homographies = _valid_homographies(job_dir)

    # Only one representative frame per segment reaches the VLM. Pass 2: decode
    # again, keeping just those few frames instead of the entire sampled set.
    rep_indices = {
        idx
        for start, end in segments
        if (idx := _nearest_index(sampled_indices, (start + end) // 2)) is not None
    }
    by_idx = {
        f.frame_idx: f.image
        for f in decode_frames(video, every_n=SAMPLE_EVERY)
        if f.frame_idx in rep_indices
    }

    locator = RimLocator()
    cache = VlmCache(job_dir / "_vlm_cache" / "rims.jsonl")
    writer = ArtifactWriter(job_dir / "rims", RIMS_SCHEMA)
    n_candidates = 0
    n_validated = 0
    for segment_id, (start, end) in enumerate(segments):
        rep_idx = _nearest_index(sampled_indices, (start + end) // 2)
        if rep_idx is None or rep_idx not in by_idx:
            continue
        candidates = locator.locate(by_idx[rep_idx], cache=cache)
        h = _nearest_homography(homographies, rep_idx)
        for cand in candidates:
            court_end, validated = (None, False)
            if h is not None:
                court_end, validated = validate_rim(cand, h, info.width, info.height)
            writer.add(
                {
                    "job_id": job_id,
                    "segment_id": segment_id,
                    "frame_start": start,
                    "frame_end": end,
                    "rep_frame_idx": rep_idx,
                    "x1": cand.x1 * info.width,
                    "y1": cand.y1 * info.height,
                    "x2": cand.x2 * info.width,
                    "y2": cand.y2 * info.height,
                    "vlm_conf": cand.confidence,
                    "court_end": court_end,
                    "validated": validated,
                }
            )
            n_candidates += 1
            n_validated += int(validated)
    writer.close()

    return {
        "job_id": job_id,
        "camera_segments": len(segments),
        "rim_candidates": n_candidates,
        "validated": n_validated,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def _valid_homographies(job_dir: Path) -> list[tuple[int, np.ndarray]]:
    rows = read_stage(job_dir / "court_frames").to_pylist()
    return [
        (r["frame_idx"], np.array(r["h"], dtype=np.float64).reshape(3, 3))
        for r in rows
        if r["h"] is not None
    ]


def _nearest_homography(
    homographies: list[tuple[int, np.ndarray]], frame_idx: int, max_gap: int = 45
) -> np.ndarray | None:
    if not homographies:
        return None
    idx, h = min(homographies, key=lambda t: abs(t[0] - frame_idx))
    return h if abs(idx - frame_idx) <= max_gap else None


def _nearest_index(indices: list[int], target: int) -> int | None:
    if not indices:
        return None
    return min(indices, key=lambda i: abs(i - target))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.video, args.out, args.job_id), indent=2))


if __name__ == "__main__":
    main()
