"""Roster-verified real jersey crops from a finished job (D9 real-data lever).

The PARSeq fine-tune is synthetic-only; this converts a job's own footage
into supervised OCR rows by cross-checking trusted jersey posteriors
against the uploader team's roster (montehall `players` table, exported as
a job_id-keyed JSON map). A posterior counts as verified only when:

1. its tracklet sits in the roster-bound team cluster (the k-means cluster
   whose trusted reads match the roster best, by margin — rosters cover the
   uploader's team only, so the other cluster is never harvested);
2. the top candidate is in the roster's number set at >= min posterior mass;
3. the read is truncation-unambiguous: a single-digit candidate is rejected
   whenever any two-digit roster number contains that digit, because
   OCR truncation ("52" read as "5") produces exactly those collisions and
   a roster match would then verify a wrong label.

Crops from verified tracklets (legibility-gated when weights are given)
land as a train_parseq dataset dir (jpgs + labels.jsonl, legible=True) to
mix with the synth sets. Verification failing binds nothing and harvests
nothing — footage that isn't the roster's team yields an empty dataset,
not a poisoned one.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

MIN_POSTERIOR = 0.6

# verification primitives live in montehall_cv.roster (shared with box-score
# attribution); re-exported here for existing callers and tests
from montehall_cv.roster import (  # noqa: E402,F401
    MIN_BIND_MATCHES,
    TEAM_CLUSTERS,
    bind_roster_cluster,
    is_truncation_ambiguous,
    normalize_number,
    roster_entry,
    roster_number_set,
    top_posteriors,
    verified_tracks,
)


def _gate_legible(crops, meta, gate):
    """run_jersey's per-tracklet top-K legibility gate, with a caller-owned
    gate instance (device choice: the harvest runs CPU-side while the GPU
    trains — nothing may touch CUDA during a train run)."""
    from montehall_cv.pipeline import run_jersey as rj

    scores = gate.scores(crops)
    by_track: dict[int, list[int]] = defaultdict(list)
    for i, (tid, _) in enumerate(meta):
        by_track[tid].append(i)
    keep: list[int] = []
    for indices in by_track.values():
        ranked = sorted(indices, key=lambda i: -scores[i])
        keep.extend(
            i for i in ranked[:rj.CROPS_PER_TRACKLET]
            if scores[i] >= rj.MIN_LEGIBILITY
        )
    keep.sort()
    return [crops[i] for i in keep], [meta[i] for i in keep]


def build(video: Path, job_dir: Path, out_dir: Path, job_id: str,
          roster_map: Path, legibility_weights: Path | None = None,
          min_posterior: float = MIN_POSTERIOR, device: str = "cuda") -> dict:
    from PIL import Image

    from montehall_cv.pipeline import run_jersey as rj
    from montehall_cv.pipeline.jersey import JerseyLegibility
    from montehall_cv.store.artifacts import read_stage

    entry = roster_entry(json.loads(roster_map.read_text()), job_id)
    if entry is None:
        stats = {"job_id": job_id, "harvested": 0, "reason": "no roster for job"}
        print(json.dumps(stats), flush=True)
        return stats
    roster = roster_number_set(entry["players"])

    entity_rows = read_stage(job_dir / "entities").to_pylist()
    cluster_of = {r["track_id"]: r["team_cluster"] for r in entity_rows}
    identity_rows = read_stage(job_dir / "identity").to_pylist()
    reads = top_posteriors(identity_rows, min_posterior)

    bound_cluster, match_counts = bind_roster_cluster(reads, cluster_of, roster)
    stats = {
        "job_id": job_id,
        "team_name": entry["team_name"],
        "roster_numbers": sorted(roster),
        "trusted_reads": len(reads),
        "cluster_matches": match_counts,
        "bound_cluster": bound_cluster,
    }
    if bound_cluster is None:
        stats.update({"harvested": 0, "reason": "no cluster bound (fail-safe)"})
        print(json.dumps(stats), flush=True)
        return stats

    labels = verified_tracks(reads, cluster_of, roster, bound_cluster)
    per_tracklet = (
        rj.GATE_CANDIDATES_PER_TRACKLET if legibility_weights is not None
        else rj.CROPS_PER_TRACKLET
    )
    plan = rj._crop_plan(job_dir, set(labels), per_tracklet)
    crops, meta = rj._collect_crops(video, plan)
    if legibility_weights is not None and crops:
        crops, meta = _gate_legible(crops, meta,
                                    JerseyLegibility(legibility_weights, device=device))

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (crop, (tid, frame_idx)) in enumerate(zip(crops, meta, strict=True)):
        name = f"{job_id[:8]}_{tid}_{i:04d}.jpg"
        Image.fromarray(crop).save(out_dir / name, quality=92)
        rows.append({"file": name, "label": labels[tid], "legible": True,
                     "job_id": job_id, "track_id": tid, "frame_idx": frame_idx})
    with open(out_dir / "labels.jsonl", "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    stats.update({
        "verified_tracklets": len(labels),
        "verified_labels": sorted(set(labels.values())),
        "harvested": len(rows),
        "gate": "trained" if legibility_weights is not None else "geometric",
    })
    print(json.dumps(stats), flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True,
                        help="finished pipeline job dir (entities/ + identity/ + localized/)")
    parser.add_argument("--out", type=Path, required=True,
                        help="train_parseq dataset dir (jpgs + labels.jsonl, appended)")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--roster-map", type=Path, required=True,
                        help="job_id-keyed roster JSON (prod players export) "
                             "or a bare per-job roster.json (app-written "
                             "input/<job_id>/roster.json)")
    parser.add_argument("--legibility-weights", type=Path, default=None)
    parser.add_argument("--min-posterior", type=float, default=MIN_POSTERIOR)
    parser.add_argument("--device", default="cuda",
                        help="legibility gate device; use cpu while the GPU trains")
    args = parser.parse_args()
    build(args.video, args.job_dir, args.out, args.job_id, args.roster_map,
          args.legibility_weights, args.min_posterior, args.device)


if __name__ == "__main__":
    main()
