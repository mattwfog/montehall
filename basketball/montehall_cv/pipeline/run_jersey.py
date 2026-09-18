"""Jersey OCR stage: torso crops -> per-tracklet jersey posteriors.

For each on-court tracklet (spatial-evidence eligibility, the same set
run_team_assoc considers), samples its largest-bbox observations (a
geometric stand-in for the legibility gate), OCRs the torso crops, and
votes confidence-weighted into a jersey posterior. Top candidates land as
IdentityHypothesis rows (bound_at_stage=ocr_vote); every accepted read is
an Evidence row with its frame.

Runs BEFORE run_team_assoc in the chain so trusted reads can veto
association merges on fresh runs (identical-uniform lookalikes are the
failure the ReID gate can't see); depends only on run_court's outputs.

Outputs: <out>/<job_id>/identity/ + <out>/<job_id>/ocr_evidence/
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from montehall_cv.pipeline.jersey import (
    MIN_BBOX_HEIGHT_PX,
    JerseyLegibility,
    JerseyOcr,
    torso_crop,
)
from montehall_cv.pipeline.video import decode_frames
from montehall_cv.store.artifacts import ArtifactWriter, read_stage, stage_complete
from montehall_cv.store.schemas import EVIDENCE_SCHEMA, IDENTITY_SCHEMA

CROPS_PER_TRACKLET = 12
GATE_CANDIDATES_PER_TRACKLET = 36  # pool the gate selects from
MIN_LEGIBILITY = 0.5
TOP_CANDIDATES = 3


def run(video: Path, out_root: Path, job_id: str,
        legibility_weights: Path | None = None,
        ocr_weights: Path | None = None) -> dict:
    job_dir = out_root / job_id
    if stage_complete(job_dir / "identity") and stage_complete(job_dir / "ocr_evidence"):
        return {"job_id": job_id, "skipped": True, "reason": "stages already complete"}
    started = time.monotonic()

    from montehall_cv.pipeline import run_team_assoc as rta

    eligible = set(rta._eligible_tracklets(job_dir))
    per_tracklet = (
        GATE_CANDIDATES_PER_TRACKLET if legibility_weights is not None
        else CROPS_PER_TRACKLET
    )
    plan = _crop_plan(job_dir, eligible, per_tracklet)
    crops, meta = _collect_crops(video, plan)
    n_candidates = len(crops)
    if legibility_weights is not None and crops:
        crops, meta = _gate_legible(crops, meta, legibility_weights)
    reads = _ocr(crops, meta, ocr_weights)

    posteriors, n_reads = _vote(reads)
    _write(job_dir, job_id, posteriors, reads)

    tracked_with_jersey = sum(1 for p in posteriors.values() if p)
    return {
        "job_id": job_id,
        "eligible_tracklets": len(eligible),
        "tracklets_attempted": len(plan_tracklets(plan)),
        "gate": "trained" if legibility_weights is not None else "geometric",
        "ocr": "fine-tuned" if ocr_weights is not None else "zero-shot",
        "candidate_crops": n_candidates,
        "ocr_crops": len(crops),
        "accepted_reads": n_reads,
        "tracklets_with_jersey_posterior": tracked_with_jersey,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def plan_tracklets(plan: dict[int, list[tuple[int, tuple]]]) -> set[int]:
    return {tid for entries in plan.values() for tid, _ in entries}


def _crop_plan(
    job_dir: Path, eligible: set[int], per_tracklet: int
) -> dict[int, list[tuple[int, tuple]]]:
    """Pick the largest-height observations per tracklet (readability proxy)."""
    by_track: dict[int, list[dict]] = defaultdict(list)
    for row in read_stage(job_dir / "localized").to_pylist():
        tid = row["track_id"]
        if tid in eligible and (row["y2"] - row["y1"]) >= MIN_BBOX_HEIGHT_PX:
            by_track[tid].append(row)
    plan: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
    for tid, rows in by_track.items():
        rows.sort(key=lambda r: r["y1"] - r["y2"])  # tallest first
        for row in rows[:per_tracklet]:
            plan[row["frame_idx"]].append(
                (tid, (row["x1"], row["y1"], row["x2"], row["y2"]))
            )
    return dict(plan)


def _collect_crops(
    video: Path, plan: dict[int, list[tuple[int, tuple]]]
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    crops: list[np.ndarray] = []
    meta: list[tuple[int, int]] = []  # (track_id, frame_idx)
    remaining = set(plan)
    for frame in decode_frames(video):
        if frame.frame_idx in plan:
            for tid, bbox in plan[frame.frame_idx]:
                crop = torso_crop(frame.image, *bbox)
                if crop is not None:
                    crops.append(crop)
                    meta.append((tid, frame.frame_idx))
            remaining.discard(frame.frame_idx)
            if not remaining:
                break
    return crops, meta


def _gate_legible(
    crops: list[np.ndarray],
    meta: list[tuple[int, int]],
    weights: Path,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Keep each tracklet's most-legible crops (trained gate, Koshkina recipe)."""
    scores = JerseyLegibility(weights).scores(crops)
    by_track: dict[int, list[int]] = defaultdict(list)
    for i, (tid, _) in enumerate(meta):
        by_track[tid].append(i)
    keep: list[int] = []
    for indices in by_track.values():
        ranked = sorted(indices, key=lambda i: -scores[i])
        keep.extend(
            i for i in ranked[:CROPS_PER_TRACKLET] if scores[i] >= MIN_LEGIBILITY
        )
    keep.sort()
    return [crops[i] for i in keep], [meta[i] for i in keep]


def _ocr(crops: list[np.ndarray], meta: list[tuple[int, int]],
         ocr_weights: Path | None = None) -> list[dict]:
    results = JerseyOcr(weights=ocr_weights).read(crops) if crops else []
    return [
        {"track_id": tid, "frame_idx": fidx, "text": r.text, "confidence": r.confidence}
        for (tid, fidx), r in zip(meta, results, strict=True)
        if r is not None
    ]


MIN_AGREEING_READS = 2


def _vote(reads: list[dict]) -> tuple[dict[int, dict[str, float]], int]:
    """Confidence-weighted vote; a candidate needs >=2 agreeing reads to count."""
    votes: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for read in reads:
        votes[read["track_id"]][read["text"]] += read["confidence"]
        counts[read["track_id"]][read["text"]] += 1
    posteriors: dict[int, dict[str, float]] = {}
    for tid, candidate_weights in votes.items():
        supported = {
            c: w
            for c, w in candidate_weights.items()
            if counts[tid][c] >= MIN_AGREEING_READS
        }
        total = sum(supported.values())
        posteriors[tid] = (
            {c: w / total for c, w in supported.items()} if total > 0 else {}
        )
    return posteriors, len(reads)


def _write(job_dir: Path, job_id: str, posteriors: dict, reads: list[dict]) -> None:
    identity = ArtifactWriter(job_dir / "identity", IDENTITY_SCHEMA)
    for tid, posterior in sorted(posteriors.items()):
        ranked = sorted(posterior.items(), key=lambda kv: -kv[1])[:TOP_CANDIDATES]
        for candidate, prob in ranked:
            identity.add(
                {
                    "job_id": job_id,
                    "track_id": tid,
                    "candidate": candidate,
                    "prob": round(prob, 4),
                    "bound_at_stage": "ocr_vote",
                    "bound_at_frame": None,
                }
            )
    identity.close()

    evidence = ArtifactWriter(job_dir / "ocr_evidence", EVIDENCE_SCHEMA)
    for read in reads:
        evidence.add(
            {
                "job_id": job_id,
                "track_id": read["track_id"],
                "frame_idx": read["frame_idx"],
                "source_kind": "ocr_read",
                "value_json": json.dumps(
                    {"text": read["text"], "confidence": round(read["confidence"], 4)}
                ),
                "weight": read["confidence"],
            }
        )
    evidence.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--legibility-weights", type=Path, default=None,
                        help="trained legibility gate (legibility_resnet34.pt); "
                             "absent -> geometric tallest-crop proxy")
    parser.add_argument("--ocr-weights", type=Path, default=None,
                        help="fine-tuned PARSeq state_dict (train_parseq); "
                             "absent -> zero-shot hub checkpoint")
    args = parser.parse_args()
    print(json.dumps(run(args.video, args.out, args.job_id,
                         legibility_weights=args.legibility_weights,
                         ocr_weights=args.ocr_weights), indent=2))


if __name__ == "__main__":
    main()
