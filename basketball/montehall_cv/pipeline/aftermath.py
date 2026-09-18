"""Post-shot collective-movement features.

After a made basket the defense inbounds from behind the baseline and all
ten players flow toward the other end; after a miss, bodies converge on
the rim for the rebound. Probed on the two ground-truth-labeled benchmark
clips (2026-07-12): centroid away-flow separates makes (19-42 ft) from
misses (3-15 ft) with fast-break-rebound outliers — strong enough to veto
a frame-judged "made" verdict when no scoreboard is readable, not strong
enough to be the primary oracle (that's run_scoreboard).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from montehall_cv.store.artifacts import read_stage, stage_complete

_BUCKET_MS = 250
# centroid sampled this long after the window, and again at the horizon:
_FLOW_FROM_S = 1.5
_FLOW_TO_S = 6.0


class Aftermath:
    """Bucketed court-frame positions -> per-window movement features."""

    def __init__(self, job_dir: Path) -> None:
        self._by_bucket: dict[int, list[float]] = defaultdict(list)
        if not stage_complete(job_dir / "positions"):
            return
        for row in read_stage(job_dir / "positions").to_pylist():
            if row["cls"] == 0 and row["court_x"] is not None:
                self._by_bucket[row["ts_ms"] // _BUCKET_MS].append(row["court_x"])

    def _centroid_x(self, ts_s: float) -> float | None:
        bucket = int(ts_s * 1000) // _BUCKET_MS
        pts: list[float] = []
        for b in (bucket - 1, bucket, bucket + 1):
            pts.extend(self._by_bucket.get(b, ()))
        return sum(pts) / len(pts) if pts else None

    def away_flow_ft(self, ts_end_ms: int, court_end: str | None) -> float | None:
        """Centroid x displacement AWAY from the attacked rim after the
        window; positive = transition the other way (make-like), small or
        negative = rebound scramble (miss-like). None when the end is
        unknown or tracking has no court-frame coverage there."""
        if court_end not in ("left", "right"):
            return None
        te = ts_end_ms / 1000.0
        c1 = self._centroid_x(te + _FLOW_FROM_S)
        c2 = self._centroid_x(te + _FLOW_TO_S)
        if c1 is None or c2 is None:
            return None
        return (c2 - c1) * (-1.0 if court_end == "right" else 1.0)
