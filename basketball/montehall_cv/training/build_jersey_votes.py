"""Coach-vote-verified jersey crops (identity design C2 — the flywheel leg).

Every confirmed identity vote is a human-verified entity→number label —
stronger than roster binding, the scarcest supervision in the stack. This
harvests those entities' crops (legibility-gated when weights are given)
into the same train_parseq dataset shape as build_jersey_real
(jpgs + labels.jsonl), so OCR retrains ride the coach's answers.

Input: a votes export JSON, {job_id: {"<entity_id>": "number"}} — the
latest confirm per (video, entity), superseding denies excluded. Export
from the montehall DB (read-only, in an api pod):

    SELECT g.job_id, v.entity_id, v.player_number
    FROM cv_identity_votes v
    JOIN game_video_analyses g ON g.id = v.video_id
    WHERE v.verdict = 'confirm'
      AND v.id IN (SELECT max(id) FROM cv_identity_votes
                   GROUP BY video_id, entity_id);

Bad-merge caveat: a confirm labels the ENTITY; if the entity fused two
players, some crops carry the wrong number. Denies exist to catch fused
entities before this harvest — still, keep vote harvests in their own
dataset dir so a generation can be dropped wholesale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(video: Path, job_dir: Path, out_dir: Path, job_id: str,
          votes_map: Path, legibility_weights: Path | None = None,
          device: str = "cuda") -> dict:
    from PIL import Image

    from montehall_cv.pipeline import run_jersey as rj
    from montehall_cv.pipeline.jersey import JerseyLegibility
    from montehall_cv.roster import normalize_number
    from montehall_cv.store.artifacts import read_stage
    from montehall_cv.training.build_jersey_real import _gate_legible

    votes_all = json.loads(votes_map.read_text())
    votes: dict[int, str] = {
        int(entity_id): number
        for entity_id, raw in (votes_all.get(job_id) or {}).items()
        if (number := normalize_number(str(raw))) is not None
    }
    if not votes:
        stats = {"job_id": job_id, "harvested": 0, "reason": "no confirmed votes"}
        print(json.dumps(stats), flush=True)
        return stats

    label_of_track = {
        row["track_id"]: votes[row["entity_id"]]
        for row in read_stage(job_dir / "entities").to_pylist()
        if row["entity_id"] in votes
    }
    per_tracklet = (
        rj.GATE_CANDIDATES_PER_TRACKLET if legibility_weights is not None
        else rj.CROPS_PER_TRACKLET
    )
    plan = rj._crop_plan(job_dir, set(label_of_track), per_tracklet)
    crops, meta = rj._collect_crops(video, plan)
    if legibility_weights is not None and crops:
        crops, meta = _gate_legible(
            crops, meta, JerseyLegibility(legibility_weights, device=device)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (crop, (tid, frame_idx)) in enumerate(zip(crops, meta, strict=True)):
        name = f"vote_{job_id[:8]}_{tid}_{i:04d}.jpg"
        Image.fromarray(crop).save(out_dir / name, quality=92)
        rows.append({
            "file": name, "label": label_of_track[tid], "legible": True,
            "job_id": job_id, "track_id": tid, "frame_idx": frame_idx,
            "source": "coach_vote",
        })
    with open(out_dir / "labels.jsonl", "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    stats = {
        "job_id": job_id,
        "voted_entities": len(votes),
        "labeled_tracklets": len(label_of_track),
        "harvested": len(rows),
        "labels": sorted(set(label_of_track.values())),
        "gate": "trained" if legibility_weights is not None else "geometric",
    }
    print(json.dumps(stats), flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="train_parseq dataset dir (jpgs + labels.jsonl, appended)")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--votes-map", type=Path, required=True,
                        help="votes export JSON (see module docstring)")
    parser.add_argument("--legibility-weights", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    build(args.video, args.job_dir, args.out, args.job_id, args.votes_map,
          legibility_weights=args.legibility_weights, device=args.device)


if __name__ == "__main__":
    main()
