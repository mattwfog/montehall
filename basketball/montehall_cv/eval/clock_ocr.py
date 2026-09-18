"""Score-bug game-clock OCR via zero-shot PARSeq.

Proven 20/20 at conf .90-1.00 on the ESPN 720p score bug (2026-07-10 probe,
CPU). CLOCK_BOX is the empirical MM:SS crop at 1280x720; other frame sizes
scale it proportionally — the ESPN bug is layout-anchored, not pixel-anchored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# x1, y1, x2, y2 at 1280x720 (frame_12m probe, 2026-07-10)
CLOCK_BOX_720P = (596, 592, 680, 634)
REFERENCE_SIZE = (1280, 720)


def scaled_clock_box(width: int, height: int) -> tuple[int, int, int, int]:
    sx = width / REFERENCE_SIZE[0]
    sy = height / REFERENCE_SIZE[1]
    x1, y1, x2, y2 = CLOCK_BOX_720P
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))


def parse_clock(text: str) -> float | None:
    """MM:SS -> seconds; under-1:00 the bug shows SS.T tenths."""
    if ":" in text:
        m, s = text.split(":", 1)
        if m.isdigit() and s.isdigit() and int(s) < 60:
            return float(int(m) * 60 + int(s))
        return None
    if "." in text and text.replace(".", "").isdigit():
        try:
            value = float(text)
        except ValueError:
            return None
        return value if value < 60 else None
    return None


@dataclass(frozen=True)
class ClockRead:
    video_t: float
    text: str
    conf: float
    clock_s: float | None


class ClockReader:
    """Batched PARSeq reads over clock crops. CPU-safe (GPU optional)."""

    def __init__(self, device: str = "cpu", batch_size: int = 64) -> None:
        import torch

        self._torch = torch
        self._model = torch.hub.load("baudm/parseq", "parseq", pretrained=True).eval()
        self._model = self._model.to(device)
        self._device = device
        self._batch = batch_size
        self._h, self._w = self._model.hparams.img_size

    def read(self, crops: list[np.ndarray], times: list[float]) -> list[ClockRead]:
        if len(crops) != len(times):
            raise ValueError(f"{len(crops)} crops vs {len(times)} times")
        import torchvision.transforms.functional as tf
        from PIL import Image

        out: list[ClockRead] = []
        for i in range(0, len(crops), self._batch):
            chunk = crops[i : i + self._batch]
            tensors = [
                tf.normalize(
                    tf.to_tensor(tf.resize(Image.fromarray(c), [self._h, self._w])),
                    [0.5] * 3,
                    [0.5] * 3,
                )
                for c in chunk
            ]
            batch = self._torch.stack(tensors).to(self._device)
            with self._torch.no_grad():
                logits = self._model(batch)
            labels, confs = self._model.tokenizer.decode(logits.softmax(-1))
            for t, label, conf in zip(times[i : i + self._batch], labels, confs):
                c = float(conf.min())
                out.append(ClockRead(video_t=t, text=label, conf=c, clock_s=parse_clock(label)))
        return out


def iter_clock_crops(video_path: str, step_s: float = 2.0):
    """Yield (crop_rgb, video_t) at step_s stride; box scaled to frame size."""
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    box: tuple[int, int, int, int] | None = None
    last = -1e9
    for frame in container.decode(stream):
        t = float(frame.pts * stream.time_base)
        if t - last < step_s:
            continue
        last = t
        img = frame.to_ndarray(format="rgb24")
        if box is None:
            box = scaled_clock_box(img.shape[1], img.shape[0])
        x1, y1, x2, y2 = box
        yield img[y1:y2, x1:x2], t
    container.close()
