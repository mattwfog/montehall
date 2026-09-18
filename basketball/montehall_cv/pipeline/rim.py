"""Rim location: VLM proposes, court geometry validates.

The rim is static within a camera segment, so cost scales with cuts, not
frames. Every geometrically-validated answer doubles as a training label
for the eventual detector distillation (bootstrap -> validate -> distill).
"""

from __future__ import annotations

import base64
import io
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from montehall_cv.court import NCAA_WIDTH_FT, project_points
from montehall_cv.store.vlm_cache import VlmCache, content_key

RIM_GROUND_FT = {"left": (5.25, NCAA_WIDTH_FT / 2), "right": (88.75, NCAA_WIDTH_FT / 2)}
MAX_DX_FRAC = 0.12
MIN_ABOVE_FRAC = 0.01
MAX_ABOVE_FRAC = 0.65

CUT_DIFF_THRESHOLD = 25.0
MIN_SEGMENT_FRAMES = 30

VLM_MODEL = "claude-sonnet-5"
PROMPT = (
    "Locate every basketball rim/hoop visible in this game-film frame. "
    "Respond with ONLY a JSON array (no prose). Each element: "
    '{"x1": float, "y1": float, "x2": float, "y2": float, "confidence": float} '
    "with coordinates normalized to 0-1 (x1,y1 = top-left of a tight box around "
    "the rim ring + net, x2,y2 = bottom-right). Empty array if no rim is visible."
)


@dataclass(frozen=True)
class RimCandidate:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


def detect_segments(
    frames: Iterable[tuple[int, np.ndarray]], threshold: float = CUT_DIFF_THRESHOLD
) -> list[tuple[int, int]]:
    """Camera segments from subsampled frames: (frame_idx, image) -> [(start, end)].

    Streams the input — only the previous thumbnail is retained, never the frames
    themselves — so a multi-hour game costs O(1) memory here instead of buffering
    tens of thousands of full-resolution frames.
    """
    cuts: list[int] = []
    prev: np.ndarray | None = None
    last_idx: int | None = None
    for frame_idx, image in frames:
        cur = _thumb(image)
        if prev is None or float(np.abs(cur - prev).mean()) > threshold:
            cuts.append(frame_idx)
        prev = cur
        last_idx = frame_idx
    if last_idx is None:
        return []
    ends = cuts[1:] + [last_idx + 1]
    return [(start, end - 1) for start, end in zip(cuts, ends, strict=True) if end - start > 1]


def _thumb(image: np.ndarray) -> np.ndarray:
    return image[::16, ::16].mean(axis=2).astype(np.float32)


class RimLocator:
    def __init__(self, model: str = VLM_MODEL) -> None:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def locate(self, image: np.ndarray, cache: VlmCache | None = None) -> list[RimCandidate]:
        from PIL import Image

        key = content_key(self._model, PROMPT, str(image.shape), image.tobytes())
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                return [RimCandidate(**c) for c in hit["candidates"]]

        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="JPEG", quality=85)
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
        )
        text = next(
            (block.text for block in response.content if hasattr(block, "text")), ""
        )
        result = _parse(text)
        if cache is not None:
            cache.put(key, {"candidates": [asdict(c) for c in result]})
        return result


def _parse(text: str) -> list[RimCandidate]:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for item in raw:
        try:
            candidate = RimCandidate(
                x1=float(item["x1"]),
                y1=float(item["y1"]),
                x2=float(item["x2"]),
                y2=float(item["y2"]),
                confidence=float(item.get("confidence", 0.5)),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= candidate.x1 < candidate.x2 <= 1 and 0 <= candidate.y1 < candidate.y2 <= 1:
            out.append(candidate)
    return out


def validate_rim(
    candidate: RimCandidate, h_image_to_court: np.ndarray, frame_w: int, frame_h: int
) -> tuple[str | None, bool]:
    """Check a rim box sits plausibly above a known rim ground position.

    Returns (court_end, validated). The homography maps the ground plane
    only, so the elevated rim must be ABOVE its ground point in image space
    and horizontally near it.
    """
    h_court_to_image = np.linalg.inv(h_image_to_court)
    cx = (candidate.x1 + candidate.x2) / 2 * frame_w
    cy = (candidate.y1 + candidate.y2) / 2 * frame_h

    best: tuple[str, float] | None = None
    for end, ground_ft in RIM_GROUND_FT.items():
        ground_px = project_points(h_court_to_image, np.array([ground_ft]))[0]
        dx = abs(cx - ground_px[0])
        above = ground_px[1] - cy
        if dx > MAX_DX_FRAC * frame_w:
            continue
        if not (MIN_ABOVE_FRAC * frame_h <= above <= MAX_ABOVE_FRAC * frame_h):
            continue
        if best is None or dx < best[1]:
            best = (end, dx)
    if best is None:
        return None, False
    return best[0], True
