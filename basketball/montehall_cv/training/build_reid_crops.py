"""Build ReID training crops from a job's own on-court tracklets (D9 domain data).

TrackID3x3 is 3v3 footage; deployment footage is different. This harvests
<identity>/<crop>.jpg folders (train_reid_v2.py's data contract) from a
finished pipeline job so ReID can train on the actual camera domain.

Identity = track_id, deliberately NOT entity_id: entity merges were made by
the previous ReID model, so training on them would recycle its mistakes.
A player split across tracklets becomes multiple pseudo-identities — benign
CE dilution, never a wrong "same" label.

Reuses run_team_assoc's eligibility/observation/decode path (single CPU
decode pass, on-court evidence filter) with a training-sized per-track
sample cap instead of the embedding-sized one.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build(video: Path, job_dir: Path, out_dir: Path, prefix: str,
          crops_per_track: int = 24, min_crops: int = 2) -> dict:
    import numpy as np
    from PIL import Image

    from montehall_cv.pipeline import run_team_assoc as rta

    tracklets = rta._eligible_tracklets(job_dir)
    obs = rta._tracklet_observations(job_dir, set(tracklets))

    plan: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
    for tid, rows in obs.items():
        picks = np.linspace(0, len(rows) - 1, min(crops_per_track, len(rows))).astype(int)
        for i in sorted(set(picks.tolist())):
            row = rows[i]
            plan[row["frame_idx"]].append(
                (tid, (row["x1"], row["y1"], row["x2"], row["y2"]))
            )
    crops, owners = rta._collect_crops(video, dict(plan))

    by_track: dict[int, list] = defaultdict(list)
    for crop, tid in zip(crops, owners, strict=True):
        by_track[tid].append(crop)

    n_ids, n_crops = 0, 0
    for tid, tcrops in sorted(by_track.items()):
        if len(tcrops) < min_crops:
            continue
        ident_dir = out_dir / f"{prefix}_{tid}"
        ident_dir.mkdir(parents=True, exist_ok=True)
        for i, crop in enumerate(tcrops):
            Image.fromarray(crop).save(ident_dir / f"{i:03d}.jpg", quality=92)
        n_ids += 1
        n_crops += len(tcrops)

    stats = {"prefix": prefix, "eligible_tracklets": len(tracklets),
             "identities": n_ids, "crops": n_crops}
    print(json.dumps(stats), flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--job-dir", type=Path, required=True,
                        help="finished pipeline job dir (evidence/ + localized/)")
    parser.add_argument("--out", type=Path, required=True,
                        help="crop root; identities land as <prefix>_<track_id>/")
    parser.add_argument("--prefix", required=True,
                        help="identity namespace, e.g. the clip's short job id")
    parser.add_argument("--crops-per-track", type=int, default=24)
    parser.add_argument("--min-crops", type=int, default=2)
    args = parser.parse_args()
    build(args.video, args.job_dir, args.out, args.prefix,
          args.crops_per_track, args.min_crops)


if __name__ == "__main__":
    main()
