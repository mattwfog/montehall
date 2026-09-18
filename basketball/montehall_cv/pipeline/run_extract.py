"""Extract stage runner: video -> detections + tracklets parquet artifacts.

Usage (inside the GPU container on spark):
    python -m montehall_cv.pipeline.run_extract \
        --video /data/video.mp4 --out /data/out --job-id <uuid>

Output layout (all incremental + resumable, see store.artifacts):
    <out>/<job_id>/detections/part-*.parquet + _SUCCESS
    <out>/<job_id>/tracklets/part-*.parquet + _SUCCESS
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from montehall_cv.pipeline.detect import RFDetrDetector
from montehall_cv.pipeline.track import PersonTracker
from montehall_cv.pipeline.video import decode_frames, probe
from montehall_cv.store.artifacts import ArtifactWriter, stage_complete
from montehall_cv.store.records import DetClass
from montehall_cv.store.schemas import DETECTIONS_SCHEMA, TRACKLETS_SCHEMA

PROGRESS_EVERY_FRAMES = 300


def run(
    video: Path,
    out_root: Path,
    job_id: str,
    every_n: int = 1,
    batch_size: int = 16,
    conf: float = 0.4,
    max_frames: int = 0,
    resolution: int | None = None,
) -> dict:
    job_dir = out_root / job_id
    det_dir = job_dir / "detections"
    trk_dir = job_dir / "tracklets"
    if stage_complete(det_dir) and stage_complete(trk_dir):
        return {"job_id": job_id, "skipped": True, "reason": "stages already complete"}

    info = probe(video)
    detector = RFDetrDetector(threshold=conf, resolution=resolution, batch_size=batch_size)
    tracker = PersonTracker(frame_rate=(info.average_fps or 30.0) / every_n)
    det_writer = ArtifactWriter(det_dir, DETECTIONS_SCHEMA)
    trk_writer = ArtifactWriter(trk_dir, TRACKLETS_SCHEMA)

    started = time.monotonic()
    n_frames = 0
    last_ts_ms = 0
    batch: list = []

    def process_batch() -> None:
        nonlocal n_frames, last_ts_ms
        if not batch:
            return
        results = detector.detect([f.image for f in batch])
        for frame, dets in zip(batch, results, strict=True):
            tracked = tracker.update(frame.frame_idx, frame.ts_ms, dets)
            det_idx = 0
            for i in range(len(tracked.track_ids)):
                x1, y1, x2, y2 = tracked.xyxy[i].tolist()
                det_writer.add(
                    _row(job_id, frame, det_idx, DetClass.PERSON, x1, y1, x2, y2,
                         float(tracked.conf[i]), int(tracked.track_ids[i]))
                )
                det_idx += 1
            # untracked classes ride along raw: ball feeds controls, rim
            # feeds run_shots' detector-rim windows (fine-tuned models only)
            for det_cls in (DetClass.BALL, DetClass.RIM):
                mask = dets.cls == int(det_cls)
                for i in mask.nonzero()[0].tolist():
                    x1, y1, x2, y2 = dets.xyxy[i].tolist()
                    det_writer.add(
                        _row(job_id, frame, det_idx, det_cls, x1, y1, x2, y2,
                             float(dets.conf[i]), None)
                    )
                    det_idx += 1
            n_frames += 1
            last_ts_ms = frame.ts_ms
        batch.clear()
        if n_frames % PROGRESS_EVERY_FRAMES < batch_size:
            elapsed = time.monotonic() - started
            print(
                f"progress frames={n_frames} video_ts={last_ts_ms / 1000:.1f}s "
                f"dets={det_writer.rows_written} fps={n_frames / elapsed:.1f}",
                flush=True,
            )

    for frame in decode_frames(video, every_n=every_n, max_frames=max_frames):
        batch.append(frame)
        if len(batch) >= batch_size:
            process_batch()
    process_batch()

    if n_frames > 0 and det_writer.rows_written == 0:
        # Real game footage always yields detections; a zero-detection pass
        # means the detector rig is broken (e.g. silent CPU fallback after
        # the container lost the GPU) — refuse to stamp success.
        raise RuntimeError(
            f"extract produced 0 detections over {n_frames} frames — "
            "detector rig broken, not marking stage complete"
        )
    det_writer.close()
    trk_writer.add_many(tracker.tracklet_rows(job_id))
    trk_writer.close()

    elapsed = time.monotonic() - started
    summary = {
        "job_id": job_id,
        "video": str(video),
        "codec": info.codec,
        "resolution": f"{info.width}x{info.height}",
        "frames_processed": n_frames,
        "video_seconds_processed": round(last_ts_ms / 1000, 1),
        "detections": det_writer.rows_written,
        "tracklets": len(tracker.tracklet_rows(job_id)),
        "wall_seconds": round(elapsed, 1),
        "pipeline_fps": round(n_frames / elapsed, 1) if elapsed > 0 else None,
        "speed_vs_realtime": round((last_ts_ms / 1000) / elapsed, 2) if elapsed > 0 else None,
    }
    return summary


def _row(job_id, frame, det_idx, cls, x1, y1, x2, y2, conf, track_id) -> dict:
    return {
        "job_id": job_id,
        "frame_idx": frame.frame_idx,
        "ts_ms": frame.ts_ms,
        "det_idx": det_idx,
        "cls": int(cls),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "conf": min(max(conf, 0.0), 1.0),
        "track_id": track_id,
        "is_detected": True,
        "court_x": None,
        "court_y": None,
        "court_conf": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=None)
    args = parser.parse_args()
    if not args.video.exists():
        print(f"video not found: {args.video}", file=sys.stderr)
        raise SystemExit(2)
    summary = run(
        args.video, args.out, args.job_id,
        every_n=args.every_n, batch_size=args.batch, conf=args.conf,
        max_frames=args.max_frames, resolution=args.resolution,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
