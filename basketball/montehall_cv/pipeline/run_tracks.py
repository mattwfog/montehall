"""Entity overlay tracks (identity design B1, 2026-07-10).

Compact per-entity pixel-box tracks for the in-video "Who is this?" overlay:
for every entity holding a box-score stat line, downsample its localized
detections to ~4Hz keyframes — the SPA linearly interpolates between them
while the raw video plays (Waze-style prompts need a box on the player,
not 30fps precision). Output: entity_tracks/tracks.json, a few KB per
entity vs the full detections parquet.

Shape: {"video": {"width": W, "height": H},
        "entities": {"<entity_id>": [[ts_ms, x1, y1, x2, y2], ...]}}
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from montehall_cv.store.artifacts import read_stage, stage_complete

STAGE = "entity_tracks"
SAMPLE_MS = 250  # ~4Hz keyframes


def run(video: Path, out_root: Path, job_id: str) -> dict:
    from montehall_cv.pipeline.video import probe

    job_dir = out_root / job_id
    stage_dir = job_dir / STAGE
    if stage_complete(stage_dir):
        return {"job_id": job_id, "skipped": True, "reason": "stage already complete"}
    for needed in ("box_score", "entities", "localized"):
        if not stage_complete(job_dir / needed):
            return {"job_id": job_id, "skipped": True, "reason": f"no {needed} stage"}
    started = time.monotonic()
    stage_dir.mkdir(parents=True, exist_ok=True)

    line_entities = {
        row["entity_id"]
        for row in read_stage(job_dir / "box_score").to_pylist()
        if row["entity_id"] is not None
    }
    entity_of_track = {
        row["track_id"]: row["entity_id"]
        for row in read_stage(job_dir / "entities").to_pylist()
        if row["entity_id"] in line_entities
    }

    last_kept: dict[int, int] = {}
    samples: dict[int, list[list[float]]] = defaultdict(list)
    rows = sorted(
        (r for r in read_stage(job_dir / "localized").to_pylist()
         if r["track_id"] in entity_of_track),
        key=lambda r: r["ts_ms"],
    )
    for row in rows:
        entity_id = entity_of_track[row["track_id"]]
        if row["ts_ms"] - last_kept.get(entity_id, -SAMPLE_MS) < SAMPLE_MS:
            continue
        last_kept[entity_id] = row["ts_ms"]
        samples[entity_id].append([
            int(row["ts_ms"]),
            round(float(row["x1"]), 1), round(float(row["y1"]), 1),
            round(float(row["x2"]), 1), round(float(row["y2"]), 1),
        ])

    info = probe(video)
    payload = {
        "video": {"width": info.width, "height": info.height},
        "entities": {str(eid): pts for eid, pts in sorted(samples.items())},
    }
    (stage_dir / "tracks.json").write_text(json.dumps(payload))
    (stage_dir / "_SUCCESS").touch()
    return {
        "job_id": job_id,
        "entities_tracked": len(samples),
        "keyframes": sum(len(v) for v in samples.values()),
        "bytes": (stage_dir / "tracks.json").stat().st_size,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.video, args.out, args.job_id), indent=2))


if __name__ == "__main__":
    main()
