"""Event-time shooter reads: name the stat row from jerseys read AT the event.

Item-3 ground truth (2026-07-12) showed cluster-level naming smears: one
trusted read names a long-gap-merged entity, and every play the merged
blob made inherits the name (#30's and #0's plays credited to #1). The
market-leader pattern is local: read the shooter's jersey around the
moment of the attempt and let THAT evidence name THAT stat row. Cluster
identity stays as the prior when the event offers no read; a conflict
between the two demotes the row to an anonymous entity row (never a
wrong name — 0-fabrication).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from montehall_cv.store.artifacts import read_stage, stage_complete

# Around the release: the window start is ball-near-rim, so the shooter
# was squared up shortly before it.
EVENT_WINDOW_BEFORE_MS = 2500
EVENT_WINDOW_AFTER_MS = 1200
CROPS_PER_EVENT = 10
EVENT_MIN_AGREE = 2  # trusted = this many accepted reads agreeing
ACTIVE_TRACK_SLACK_MS = 1500  # shooter's tracklet must overlap the event this closely
CONCURRENCY_SLACK_MS = 1000  # a number seen on another live track this close vetoes the prior
INTRA_ENTITY_OVERLAP_MS = 500  # own-track overlap beyond this = two bodies in one entity


class TrackIndex:
    """Tracklet spans + trusted jersey posteriors, indexed for naming.

    Ground truth 2026-07-12: entity naming smears on bad merges (Clemson:
    #30's and #0's plays credited to #1) yet correct fragments must SHARE
    a name (UConn: white #24 split across 3 entities, all reading 24).
    The discriminator is concurrency — one number is one body, so a
    number read on a track that is on court simultaneously with the
    shooter's track can never name the shooter."""

    def __init__(self, job_dir: Path, min_prob: float) -> None:
        self.ready = stage_complete(job_dir / "localized") and stage_complete(
            job_dir / "identity"
        )
        self.span: dict[int, list[int]] = defaultdict(lambda: [1 << 62, -(1 << 62)])
        self.posterior: dict[int, tuple[str, float]] = {}
        self.tracks_of_number: dict[str, list[int]] = defaultdict(list)
        if not self.ready:
            return
        for row in read_stage(job_dir / "localized").to_pylist():
            if row["cls"] == 0:
                span = self.span[row["track_id"]]
                span[0] = min(span[0], row["ts_ms"])
                span[1] = max(span[1], row["ts_ms"])
        for row in read_stage(job_dir / "identity").to_pylist():
            tid = row["track_id"]
            if row["prob"] >= min_prob and (
                tid not in self.posterior or row["prob"] > self.posterior[tid][1]
            ):
                self.posterior[tid] = (row["candidate"], row["prob"])
        for tid, (num, _) in self.posterior.items():
            self.tracks_of_number[num].append(tid)

    def _covers(self, tid: int, ts_ms: int, slack_ms: int) -> bool:
        span = self.span.get(tid)
        return span is not None and (
            span[0] - slack_ms <= ts_ms <= span[1] + slack_ms
        )

    def active_name(self, track_ids, ts_ms: int) -> str | None:
        """The active tracklet's own number; conflicting reads abstain."""
        named = {
            self.posterior[tid][0]
            for tid in track_ids
            if tid in self.posterior and self._covers(tid, ts_ms, ACTIVE_TRACK_SLACK_MS)
        }
        return next(iter(named)) if len(named) == 1 else None

    def plausibly_one_body(self, track_ids) -> bool:
        """One body is never in two tracklets at once: an entity whose own
        tracks overlap in time is a proven bad merge (the Clemson '1' blob
        carried the real #1's perimeter track alongside the shooter's) —
        its entity-level name must not project onto event rows."""
        spans = sorted(
            (self.span[tid] for tid in track_ids if tid in self.span),
            key=lambda s: s[0],
        )
        return all(
            later[0] >= earlier[1] - INTRA_ENTITY_OVERLAP_MS
            for earlier, later in zip(spans, spans[1:])
        )

    def concurrent_reader_elsewhere(self, number: str, own_tracks, ts_ms: int) -> bool:
        """True when a track READING this number is on court at ts outside
        the shooter's own tracks — the number provably belongs to another
        body right now, so it must not name this row."""
        own = set(own_tracks)
        return any(
            tid not in own and self._covers(tid, ts_ms, CONCURRENCY_SLACK_MS)
            for tid in self.tracks_of_number.get(number, ())
        )


def active_track_names(
    job_dir: Path,
    wants: list[tuple[int, int, int]],
    entity_tracks: dict[int, list[int]],
    min_prob: float,
) -> dict[int, str]:
    """(event_id, shooter entity_id, ts_start_ms) -> {event_id: number}."""
    if not wants:
        return {}
    idx = TrackIndex(job_dir, min_prob)
    if not idx.ready:
        return {}
    out: dict[int, str] = {}
    for event_id, entity_id, ts_start_ms in wants:
        name = idx.active_name(entity_tracks.get(entity_id, ()), ts_start_ms)
        if name is not None:
            out[event_id] = name
    return out


def event_shooter_reads(
    video: Path,
    job_dir: Path,
    wants: list[tuple[int, int, int]],
    entity_tracks: dict[int, list[int]],
    ocr_weights: Path | None = None,
) -> dict[int, str]:
    """(event_id, shooter entity_id, ts_start_ms) -> {event_id: number}.

    Only trusted reads are returned; an event with no legible agreement
    is simply absent (the caller falls back to the cluster prior)."""
    if not wants:
        return {}

    rows_by_track: dict[int, list[dict]] = defaultdict(list)
    for row in read_stage(job_dir / "localized").to_pylist():
        if row["cls"] == 0:
            rows_by_track[row["track_id"]].append(row)

    plan: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
    for event_id, entity_id, ts_start_ms in wants:
        lo = ts_start_ms - EVENT_WINDOW_BEFORE_MS
        hi = ts_start_ms + EVENT_WINDOW_AFTER_MS
        candidates = [
            row
            for tid in entity_tracks.get(entity_id, ())
            for row in rows_by_track.get(tid, ())
            if lo <= row["ts_ms"] <= hi
        ]
        candidates.sort(key=lambda r: r["y1"] - r["y2"])  # tallest first
        for row in candidates[:CROPS_PER_EVENT]:
            plan[row["frame_idx"]].append(
                (event_id, (row["x1"], row["y1"], row["x2"], row["y2"]))
            )
    if not plan:
        return {}

    crops, meta = _collect(video, plan)
    if not crops:
        return {}

    from montehall_cv.pipeline.jersey import JerseyOcr

    ocr = JerseyOcr(device="cpu", batch_size=16, weights=ocr_weights)
    votes: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for read, (event_id, _) in zip(ocr.read(crops), meta):
        if read is not None:
            votes[event_id][read.text] += 1

    out: dict[int, str] = {}
    for event_id, counts in votes.items():
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        if ranked[0][1] >= EVENT_MIN_AGREE and (
            len(ranked) == 1 or ranked[0][1] > ranked[1][1]
        ):
            out[event_id] = ranked[0][0]
    return out


def _collect(video: Path, plan: dict[int, list[tuple[int, tuple]]]):
    """Seek-decode only the planned frames (a handful per event)."""
    import av

    from montehall_cv.pipeline.jersey import torso_crop

    crops: list[np.ndarray] = []
    meta: list[tuple[int, int]] = []
    container = av.open(str(video))
    stream = container.streams.video[0]
    fps = float(stream.average_rate or 30.0)
    last_decoded = -(10 ** 9)
    for frame_idx in sorted(plan):
        if frame_idx <= last_decoded:
            continue
        if frame_idx - last_decoded > 90:
            container.seek(int(frame_idx / fps / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            idx = round(frame.time * fps)
            if idx in plan and idx > last_decoded:
                img = frame.to_ndarray(format="rgb24")
                for event_id, bbox in plan[idx]:
                    crop = torso_crop(img, *bbox)
                    if crop is not None:
                        crops.append(crop)
                        meta.append((event_id, idx))
            if idx >= frame_idx:
                last_decoded = max(last_decoded, idx)
                break
    container.close()
    return crops, meta
