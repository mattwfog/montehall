"""Tracking stage: ByteTrack over person detections -> track ids + tracklet summaries.

supervision's ByteTrack (MIT) carries the skeleton; the ratified upgrade
path is roboflow/trackers BoT-SORT + own-trained OSNet ReID (design D7).
The ball is NOT tracked here — ball association is trajectory-based and
comes with its own stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from montehall_cv.pipeline.detect import FrameDetections
from montehall_cv.store.records import DetClass


@dataclass
class _TrackAccumulator:
    frame_start: int
    frame_end: int
    ts_start_ms: int
    ts_end_ms: int
    n: int = 0
    conf_sum: float = 0.0

    def update(self, frame_idx: int, ts_ms: int, conf: float) -> None:
        self.frame_end = frame_idx
        self.ts_end_ms = ts_ms
        self.n += 1
        self.conf_sum += conf


@dataclass(frozen=True)
class TrackedFrame:
    xyxy: np.ndarray
    conf: np.ndarray
    track_ids: np.ndarray  # (n,) int32


@dataclass
class PersonTracker:
    """Sequential per-frame tracker; feed frames in decode order."""

    frame_rate: float = 30.0
    _tracks: dict[int, _TrackAccumulator] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import supervision as sv

        self._tracker = sv.ByteTrack(frame_rate=int(round(self.frame_rate)))

    def update(self, frame_idx: int, ts_ms: int, dets: FrameDetections) -> TrackedFrame:
        import supervision as sv

        person_mask = dets.cls == int(DetClass.PERSON)
        sv_dets = sv.Detections(
            xyxy=dets.xyxy[person_mask].reshape(-1, 4),
            confidence=dets.conf[person_mask],
            class_id=np.zeros(int(person_mask.sum()), dtype=int),
        )
        tracked = self._tracker.update_with_detections(sv_dets)
        track_ids = (
            tracked.tracker_id.astype(np.int32)
            if tracked.tracker_id is not None
            else np.empty(0, dtype=np.int32)
        )
        confs = (
            tracked.confidence.astype(np.float32)
            if tracked.confidence is not None
            else np.zeros(len(tracked), dtype=np.float32)
        )
        for tid, conf in zip(track_ids.tolist(), confs.tolist(), strict=True):
            acc = self._tracks.get(tid)
            if acc is None:
                self._tracks[tid] = _TrackAccumulator(
                    frame_start=frame_idx,
                    frame_end=frame_idx,
                    ts_start_ms=ts_ms,
                    ts_end_ms=ts_ms,
                    n=1,
                    conf_sum=conf,
                )
            else:
                acc.update(frame_idx, ts_ms, conf)
        return TrackedFrame(
            xyxy=np.asarray(tracked.xyxy, dtype=np.float32).reshape(-1, 4),
            conf=confs,
            track_ids=track_ids,
        )

    def tracklet_rows(self, job_id: str) -> list[dict]:
        return [
            {
                "job_id": job_id,
                "track_id": tid,
                "cls": int(DetClass.PERSON),
                "frame_start": acc.frame_start,
                "frame_end": acc.frame_end,
                "ts_start_ms": acc.ts_start_ms,
                "ts_end_ms": acc.ts_end_ms,
                "n_detections": acc.n,
                "mean_conf": acc.conf_sum / acc.n if acc.n else 0.0,
            }
            for tid, acc in sorted(self._tracks.items())
        ]
