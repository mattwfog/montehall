"""Regenerate per-detection track binding by replaying the tracker.

The extract stages persist detections (per-det, no track ids) and
tracklets (span summaries only) — the per-frame det->track binding was
never written. PersonTracker is supervision ByteTrack fed sequentially:
deterministic over identical input. Replaying the stored detections in
frame order reproduces the ORIGINAL track ids, and the proof is exact:
the replayed tracklet_rows must equal the stored tracklets stage.

Output stage: <game_dir>/track_binding (one row per tracked person box
per frame). VERIFY_* lines report the span comparison — treat any
mismatch as "binding NOT faithful, do not join entities/evidence."

Usage (inside cvbench, CPU):
    python scripts/regen_track_binding.py /work/out-harvest/<game_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from montehall_cv.pipeline.detect import FrameDetections
from montehall_cv.pipeline.track import PersonTracker
from montehall_cv.store.artifacts import ArtifactWriter, read_stage, stage_complete

TRACK_BINDING_SCHEMA = pa.schema([
    pa.field("job_id", pa.string()),
    pa.field("frame_idx", pa.int32()),
    pa.field("ts_ms", pa.int64()),
    pa.field("track_id", pa.int32()),
    pa.field("x1", pa.float32()),
    pa.field("y1", pa.float32()),
    pa.field("x2", pa.float32()),
    pa.field("y2", pa.float32()),
    pa.field("conf", pa.float32()),
])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game_dir", type=Path)
    ap.add_argument("--frame-rate", type=float, default=None,
                    help="tracker frame rate; default = infer from ts")
    args = ap.parse_args()
    game_dir = args.game_dir
    stage_dir = game_dir / "track_binding"
    if stage_complete(stage_dir):
        print("TRACK_BINDING skipped (stage complete)")
        return

    det = read_stage(game_dir / "detections")
    job_id = det.column("job_id")[0].as_py()
    frame_idx = det.column("frame_idx").to_numpy()
    ts = det.column("ts_ms").to_numpy()
    cls = det.column("cls").to_numpy()
    xyxy = np.stack([det.column(c).to_numpy()
                     for c in ("x1", "y1", "x2", "y2")], axis=1)
    conf = det.column("conf").to_numpy()
    order = np.lexsort((np.arange(len(frame_idx)), frame_idx))
    frame_idx, ts, cls, conf = (a[order] for a in (frame_idx, ts, cls, conf))
    xyxy = xyxy[order]

    # feed EVERY frame in range, including detection-empty ones — ByteTrack
    # ages tracks per update() call; skipping empties changes lifecycles
    uniq_frames, starts = np.unique(frame_idx, return_index=True)
    bounds = list(starts) + [len(frame_idx)]
    have = {int(f): k for k, f in enumerate(uniq_frames)}
    all_frames = np.arange(int(frame_idx.min()), int(frame_idx.max()) + 1)
    if args.frame_rate is None:
        span_s = (ts.max() - ts.min()) / 1000.0
        rate = (uniq_frames.size - 1) / span_s if span_s > 0 else 30.0
    else:
        rate = args.frame_rate
    print(f"TRACK_BINDING frames={uniq_frames.size} rate={rate:.2f}",
          flush=True)

    tracker = PersonTracker(frame_rate=rate)
    writer = ArtifactWriter(stage_dir, TRACK_BINDING_SCHEMA)
    n_rows = 0
    empty = FrameDetections(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        conf=np.zeros(0, dtype=np.float32),
        cls=np.zeros(0, dtype=np.int8))
    last_ts = int(ts[0])
    for f in all_frames:
        k = have.get(int(f))
        if k is None:
            tracker.update(int(f), last_ts, empty)
            continue
        i, j = bounds[k], bounds[k + 1]
        last_ts = int(ts[i])
        dets = FrameDetections(
            xyxy=xyxy[i:j], conf=conf[i:j], cls=cls[i:j])
        tracked = tracker.update(int(f), int(ts[i]), dets)
        for m in range(tracked.track_ids.size):
            writer.add({
                "job_id": job_id, "frame_idx": int(f),
                "ts_ms": int(ts[i]),
                "track_id": int(tracked.track_ids[m]),
                "x1": float(tracked.xyxy[m, 0]),
                "y1": float(tracked.xyxy[m, 1]),
                "x2": float(tracked.xyxy[m, 2]),
                "y2": float(tracked.xyxy[m, 3]),
                "conf": float(tracked.conf[m]),
            })
            n_rows += 1
        if int(f) % 20000 == 0:
            print(f"TRACK_BINDING: {int(f)}/{all_frames[-1]} frames",
                  flush=True)
    writer.close()

    # exact verification against the stored spans
    from montehall_cv.store.records import DetClass

    stored = {r["track_id"]: r for r in
              read_stage(game_dir / "tracklets").to_pylist()
              if r["cls"] == int(DetClass.PERSON)}
    replayed = {r["track_id"]: r for r in tracker.tracklet_rows(job_id)}
    same_ids = set(stored) == set(replayed)
    span_mismatches = sum(
        1 for tid in (set(stored) & set(replayed))
        if (stored[tid]["frame_start"], stored[tid]["frame_end"],
            stored[tid]["n_detections"])
        != (replayed[tid]["frame_start"], replayed[tid]["frame_end"],
            replayed[tid]["n_detections"]))
    print("VERIFY_TRACK_BINDING " + json.dumps({
        "rows": n_rows,
        "stored_tracklets": len(stored),
        "replayed_tracklets": len(replayed),
        "same_id_set": same_ids,
        "span_mismatches": span_mismatches,
        "faithful": bool(same_ids and span_mismatches == 0),
    }), flush=True)


if __name__ == "__main__":
    main()
