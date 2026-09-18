"""Entity evidence sheets: per-entity crop grids for the coach naming flow
(identity design A6, 2026-07-10).

For every entity holding a box-score stat line, tile its best torso crops
(legibility-gated when weights are given) into one JPEG —
entity_evidence/e<entity_id>.jpg. The runner ships these with outputs and
the app shows them in the "Who is this?" modal, so the coach answers from
the same aggregated view the VLM reads (contact_sheet.build_sheet).

CPU-only (decode + PIL); safe alongside GPU work.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from montehall_cv.pipeline.contact_sheet import CROPS_PER_SHEET, build_sheet
from montehall_cv.pipeline.run_jersey import _collect_crops, _crop_plan, _gate_legible
from montehall_cv.store.artifacts import read_stage, stage_complete

STAGE = "entity_evidence"


def run(video: Path, out_root: Path, job_id: str,
        legibility_weights: Path | None = None) -> dict:
    job_dir = out_root / job_id
    stage_dir = job_dir / STAGE
    if stage_complete(stage_dir):
        return {"job_id": job_id, "skipped": True, "reason": "stage already complete"}
    if not stage_complete(job_dir / "box_score"):
        return {"job_id": job_id, "skipped": True, "reason": "no box_score stage"}
    started = time.monotonic()
    stage_dir.mkdir(parents=True, exist_ok=True)

    line_entities = {
        row["entity_id"]
        for row in read_stage(job_dir / "box_score").to_pylist()
        if row["entity_id"] is not None
    }
    entity_of_track: dict[int, int] = {
        row["track_id"]: row["entity_id"]
        for row in read_stage(job_dir / "entities").to_pylist()
        if row["entity_id"] in line_entities
    }
    plan = _crop_plan(job_dir, set(entity_of_track), CROPS_PER_SHEET)
    crops, meta = _collect_crops(video, plan)
    if legibility_weights is not None and crops:
        crops, meta = _gate_legible(crops, meta, legibility_weights)

    by_entity: dict[int, list] = defaultdict(list)
    for crop, (tid, _) in zip(crops, meta, strict=True):
        by_entity[entity_of_track[tid]].append(crop)

    written = 0
    for entity_id, entity_crops in sorted(by_entity.items()):
        (stage_dir / f"e{entity_id}.jpg").write_bytes(build_sheet(entity_crops))
        written += 1
    (stage_dir / "_SUCCESS").touch()

    return {
        "job_id": job_id,
        "line_entities": len(line_entities),
        "sheets_written": written,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--legibility-weights", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.video, args.out, args.job_id,
                         legibility_weights=args.legibility_weights), indent=2))


if __name__ == "__main__":
    main()
