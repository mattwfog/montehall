"""sample_frames_at correctness: seek lands at/after the requested time.

Gates the run_scoreboard streaming rework (2026-07-14): the board locate
and refine now seek-fetch a handful of frames instead of buffering a
decode pass. A wrong seek silently fails the stage open — the board oracle
disappears from run_boxscore without any error — so the fetch contract is
worth its own test.
"""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from montehall_cv.pipeline.video import sample_frames_at

W, H, N_FRAMES, FRAME_MS = 64, 48, 30, 100


def _write_video(path: Path) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 1000)
        for i in range(N_FRAMES):
            img = np.full((H, W, 3), min(30 + 7 * i, 255), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i * FRAME_MS
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def test_samples_land_at_or_after_requested_times(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_video(video)
    times = [0.0, 0.95, 2.5]
    frames = list(sample_frames_at(video, times))
    assert len(frames) == len(times)
    for t, frame in zip(sorted(times), frames):
        assert frame.ts_ms / 1000.0 >= t - 0.02
        # within one frame interval of the request — seek, not scan-from-0
        assert frame.ts_ms / 1000.0 <= t + 3 * FRAME_MS / 1000.0
        assert frame.image.shape == (H, W, 3)


def test_times_past_end_yield_nothing(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_video(video)
    frames = list(sample_frames_at(video, [1.0, 60.0]))
    assert len(frames) == 1  # the in-range request only


def test_unsorted_input_returns_ascending(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _write_video(video)
    frames = list(sample_frames_at(video, [2.0, 0.5]))
    assert [f.ts_ms for f in frames] == sorted(f.ts_ms for f in frames)
    assert len(frames) == 2
