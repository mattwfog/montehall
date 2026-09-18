"""Court stage runner: detections + video -> localized detections + on-court evidence.

Samples frames every K, segments court landmarks, solves image->court
homographies, then projects every detection's ground point through the
nearest valid solve (within K frames). Output:

    <out>/<job_id>/court_frames/   one row per sampled frame (H + quality)
    <out>/<job_id>/localized/      detections rows with court_x/y/conf filled
    <out>/<job_id>/evidence/       per-tracklet spatial on-court evidence
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from montehall_cv.court import CourtSpec, bbox_ground_point, project_points
from montehall_cv.pipeline.court_seg import CourtSegmenter, CourtSolve, solve_frame
from montehall_cv.pipeline.video import decode_frames, probe
from montehall_cv.store.artifacts import ArtifactWriter, read_stage, stage_complete
from montehall_cv.store.schemas import (
    COURT_FRAMES_SCHEMA,
    DETECTIONS_SCHEMA,
    EVIDENCE_SCHEMA,
)

ON_COURT_MARGIN_FT = 3.0


def run(
    video: Path,
    out_root: Path,
    job_id: str,
    weights: Path,
    sample_every: int = 15,
    seg_batch: int = 8,
) -> dict:
    job_dir = out_root / job_id
    if not stage_complete(job_dir / "detections"):
        raise FileNotFoundError(f"detections stage incomplete for {job_id}")
    if stage_complete(job_dir / "localized") and stage_complete(job_dir / "evidence"):
        return {"job_id": job_id, "skipped": True, "reason": "stages already complete"}

    started = time.monotonic()
    info = probe(video)
    segmenter = CourtSegmenter(weights)

    solves = _solve_samples(
        video, job_dir, job_id, segmenter, sample_every, seg_batch, info.width, info.height
    )
    valid = [s for s in solves if s.h is not None]

    n_localized, n_on_court, track_stats = _localize(job_dir, solves, sample_every)
    _write_evidence(job_dir, job_id, track_stats)

    elapsed = time.monotonic() - started
    residuals = [s.residual_ft for s in valid]
    return {
        "job_id": job_id,
        "samples": len(solves),
        "valid_homographies": len(valid),
        "median_residual_ft": round(float(np.median(residuals)), 2) if residuals else None,
        "detections_localized": n_localized,
        "detections_on_court": n_on_court,
        "tracklets_with_evidence": len(track_stats),
        "wall_seconds": round(elapsed, 1),
    }


def _solve_samples(
    video: Path,
    job_dir: Path,
    job_id: str,
    segmenter: CourtSegmenter,
    sample_every: int,
    seg_batch: int,
    frame_w: int,
    frame_h: int,
) -> list[CourtSolve]:
    writer = ArtifactWriter(job_dir / "court_frames", COURT_FRAMES_SCHEMA)
    solves: list[CourtSolve] = []
    batch: list = []

    def flush() -> None:
        if not batch:
            return
        masks = segmenter.masks([f.image for f in batch])
        for frame, mask in zip(batch, masks, strict=True):
            solve = solve_frame(mask, frame_w, frame_h, frame.frame_idx, frame.ts_ms)
            solves.append(solve)
            writer.add(
                {
                    "job_id": job_id,
                    "frame_idx": solve.frame_idx,
                    "ts_ms": solve.ts_ms,
                    "h": solve.h.flatten().tolist() if solve.h is not None else None,
                    "residual_ft": solve.residual_ft,
                    "n_points": solve.n_points,
                }
            )
        batch.clear()

    for frame in decode_frames(video, every_n=sample_every):
        batch.append(frame)
        if len(batch) >= seg_batch:
            flush()
    flush()
    writer.close()
    return solves


def _localize(
    job_dir: Path, solves: list[CourtSolve], sample_every: int
) -> tuple[int, int, dict]:
    court = CourtSpec()
    valid = [(s.frame_idx, s.h, s.residual_ft) for s in solves if s.h is not None]
    valid_idx = np.array([v[0] for v in valid]) if valid else np.empty(0)

    detections = read_stage(job_dir / "detections")
    writer = ArtifactWriter(job_dir / "localized", DETECTIONS_SCHEMA)
    track_stats: dict[int, dict] = defaultdict(lambda: {"n": 0, "n_localized": 0, "on": 0})

    n_localized = 0
    n_on_court = 0
    for row in detections.to_pylist():
        track_id = row["track_id"]
        if track_id is not None:
            track_stats[track_id]["n"] += 1
        solve = _nearest_solve(valid, valid_idx, row["frame_idx"], sample_every)
        if solve is not None:
            h, residual = solve
            ground = bbox_ground_point(row["x1"], row["y1"], row["x2"], row["y2"])
            cx, cy = project_points(h, np.array([ground]))[0]
            on_court = court.contains(float(cx), float(cy), margin_ft=ON_COURT_MARGIN_FT)
            row = {
                **row,
                "court_x": float(cx),
                "court_y": float(cy),
                "court_conf": 1.0 / (1.0 + (residual or 0.0)),
            }
            n_localized += 1
            n_on_court += int(on_court)
            if track_id is not None:
                track_stats[track_id]["n_localized"] += 1
                track_stats[track_id]["on"] += int(on_court)
        writer.add(row)
    writer.close()
    return n_localized, n_on_court, dict(track_stats)


def _nearest_solve(valid, valid_idx: np.ndarray, frame_idx: int, sample_every: int):
    if len(valid) == 0:
        return None
    pos = int(np.argmin(np.abs(valid_idx - frame_idx)))
    if abs(int(valid_idx[pos]) - frame_idx) > sample_every:
        return None
    return valid[pos][1], valid[pos][2]


def _write_evidence(job_dir: Path, job_id: str, track_stats: dict) -> None:
    writer = ArtifactWriter(job_dir / "evidence", EVIDENCE_SCHEMA)
    for track_id, stats in sorted(track_stats.items()):
        if stats["n_localized"] == 0:
            continue
        ratio = stats["on"] / stats["n_localized"]
        writer.add(
            {
                "job_id": job_id,
                "track_id": track_id,
                "frame_idx": None,
                "source_kind": "spatial",
                "value_json": json.dumps(
                    {
                        "on_court_ratio": round(ratio, 3),
                        "n_localized": stats["n_localized"],
                        "n_detections": stats["n"],
                    }
                ),
                "weight": stats["n_localized"] / stats["n"],
            }
        )
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sample-every", type=int, default=15)
    args = parser.parse_args()
    summary = run(args.video, args.out, args.job_id, args.weights, args.sample_every)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
