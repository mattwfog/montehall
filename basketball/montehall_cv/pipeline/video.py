"""VFR-safe video decode: timestamps from container pts, never frame counts."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np


@dataclass(frozen=True)
class DecodedFrame:
    frame_idx: int
    ts_ms: int
    image: np.ndarray  # HxWx3 RGB uint8


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration_ms: int | None
    codec: str
    average_fps: float | None


def probe(path: Path) -> VideoInfo:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration_ms = (
            int(container.duration * 1000 / av.time_base) if container.duration else None
        )
        fps = float(stream.average_rate) if stream.average_rate else None
        return VideoInfo(
            width=stream.width,
            height=stream.height,
            duration_ms=duration_ms,
            codec=stream.codec_context.name,
            average_fps=fps,
        )


def sample_frames_at(path: Path, times_s: "list[float]") -> Iterator[DecodedFrame]:
    """Seek-decode the first frame at/after each requested time (seconds).

    One container, ascending seeks: O(len(times_s)) decoded frames total.
    The broadcast-scale replacement for buffering a decode_frames() pass —
    a full game at every_n=10 is tens of GB of decoded RGB, which is the
    run_scoreboard OOM class. frame_idx is -1 (seek loses the ordinal).
    Times past end-of-stream yield nothing.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        for t in sorted(float(t) for t in times_s):
            container.seek(int(max(t, 0.0) / time_base), stream=stream)
            for frame in container.decode(stream):
                stamp = frame.pts if frame.pts is not None else frame.dts
                if stamp is None:
                    continue
                ts_s = float(stamp * time_base)
                if ts_s >= t - 0.02:
                    yield DecodedFrame(
                        frame_idx=-1,
                        ts_ms=int(ts_s * 1000),
                        image=frame.to_ndarray(format="rgb24"),
                    )
                    break


def decode_frames(path: Path, every_n: int = 1, max_frames: int = 0) -> Iterator[DecodedFrame]:
    """Yield RGB frames with pts-derived timestamps.

    every_n subsamples by decode ordinal; max_frames=0 means no limit.
    Frames without pts (broken muxing) fall back to dts; frames with
    neither are dropped rather than given a fabricated time.
    """
    if every_n < 1:
        raise ValueError("every_n must be >= 1")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        emitted = 0
        for decode_idx, frame in enumerate(container.decode(stream)):
            if decode_idx % every_n != 0:
                continue
            stamp = frame.pts if frame.pts is not None else frame.dts
            if stamp is None:
                continue
            ts_ms = int(stamp * time_base * 1000)
            yield DecodedFrame(
                frame_idx=decode_idx,
                ts_ms=ts_ms,
                image=frame.to_ndarray(format="rgb24"),
            )
            emitted += 1
            if max_frames and emitted >= max_frames:
                return
