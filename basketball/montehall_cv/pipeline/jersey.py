"""Jersey number OCR (PARSeq, Apache-2.0 code+weights).

Zero-shot pretrained scene-text PARSeq until the jersey fine-tune lands
(design D9). The geometric pre-filter (bbox height) stands in for the
trained legibility gate; when trained gate weights are supplied
(JerseyLegibility, our from-scratch ResNet34 on the synth set), crops are
scored and only confidently-legible ones reach OCR. Reads are accepted only
as whole numeric strings 0-99 — the roster constraint proper comes at
binding time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

LEGIBILITY_INPUT = 64  # matches train_jersey's Resize((64, 64))

MIN_BBOX_HEIGHT_PX = 80
TORSO_TOP_FRAC = 0.12
TORSO_BOTTOM_FRAC = 0.55
TORSO_WIDTH_FRAC = 0.80


@dataclass(frozen=True)
class JerseyRead:
    text: str
    confidence: float


class JerseyOcr:
    def __init__(self, device: str = "cuda", batch_size: int = 64,
                 weights: Path | None = None) -> None:
        import torch

        self._torch = torch
        self._device = torch.device(device)
        model = torch.hub.load("baudm/parseq", "parseq", pretrained=True,
                               trust_repo=True)
        if weights is not None:
            # train_parseq fine-tune (same hub model, same preprocessing —
            # the contract in its docstring); state_dict loads as-is
            model.load_state_dict(
                torch.load(weights, map_location="cpu", weights_only=True)
            )
        self._model = model.to(self._device).eval()
        self._img_size = tuple(self._model.hparams.img_size)  # (h, w)
        self._batch_size = batch_size

    def read(self, crops: list[np.ndarray]) -> list[JerseyRead | None]:
        """RGB uint8 torso crops -> numeric reads (None where not a 0-99 number)."""
        torch = self._torch
        out: list[JerseyRead | None] = []
        for start in range(0, len(crops), self._batch_size):
            chunk = crops[start : start + self._batch_size]
            batch = torch.stack([self._preprocess(c) for c in chunk]).to(self._device)
            with torch.no_grad():
                probs = self._model(batch).softmax(-1)
            labels, confidences = self._model.tokenizer.decode(probs)
            for label, char_confs in zip(labels, confidences, strict=True):
                out.append(_accept(label, char_confs))
        return out

    def _preprocess(self, crop: np.ndarray):
        torch = self._torch
        h, w = self._img_size
        tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = torch.nn.functional.interpolate(
            tensor, size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)
        return (tensor - 0.5) / 0.5


class JerseyLegibility:
    """Trained legibility gate: p(legible) per torso crop.

    From-scratch ResNet34 (license-clean, matches ImageNet-init within one
    val image — 2026-07-08 A10G ablation). Preprocessing mirrors
    train_jersey's eval transform: resize 64x64 + ToTensor, no normalize.
    """

    def __init__(self, weights: Path, device: str = "cuda",
                 batch_size: int = 256) -> None:
        import torch
        from torchvision import models, transforms

        self._torch = torch
        self._device = torch.device(device)
        model = models.resnet34(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(weights, map_location="cpu",
                                         weights_only=True))
        self._model = model.to(self._device).eval()
        self._tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((LEGIBILITY_INPUT, LEGIBILITY_INPUT)),
            transforms.ToTensor(),
        ])
        self._batch_size = batch_size

    def scores(self, crops: list[np.ndarray]) -> np.ndarray:
        """RGB uint8 crops -> p(legible) float32 (n,)."""
        torch = self._torch
        out: list[np.ndarray] = []
        for start in range(0, len(crops), self._batch_size):
            chunk = crops[start : start + self._batch_size]
            batch = torch.stack([self._tf(c) for c in chunk]).to(self._device)
            with torch.no_grad():
                probs = self._model(batch).softmax(-1)[:, 1]
            out.append(probs.float().cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


MIN_CHAR_CONF = 0.55
MIN_SINGLE_DIGIT_CONF = 0.85


def _accept(label: str, char_confs) -> JerseyRead | None:
    """Whole numeric 0-99 reads only, with per-char confidence floors.

    Single-digit reads carry a higher bar: zero-shot scene-text models read
    jersey piping and limbs as '1'-like strokes, the legacy pipeline's
    dominant failure mode.
    """
    text = label.strip()
    if not text.isdigit() or not 1 <= len(text) <= 2:
        return None
    if int(text) > 99:
        return None
    confs = [float(c) for c in char_confs]
    if not confs or min(confs) < MIN_CHAR_CONF:
        return None
    confidence = float(np.prod(confs))
    if len(text) == 1 and confidence < MIN_SINGLE_DIGIT_CONF:
        return None
    return JerseyRead(text=str(int(text)), confidence=confidence)


def torso_crop(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray | None:
    """Chest/back region of a player bbox — where the number lives."""
    h_box = y2 - y1
    w_box = x2 - x1
    cx = (x1 + x2) / 2
    tx1 = cx - w_box * TORSO_WIDTH_FRAC / 2
    tx2 = cx + w_box * TORSO_WIDTH_FRAC / 2
    ty1 = y1 + h_box * TORSO_TOP_FRAC
    ty2 = y1 + h_box * TORSO_BOTTOM_FRAC
    h, w = image.shape[:2]
    xi1, yi1 = max(int(tx1), 0), max(int(ty1), 0)
    xi2, yi2 = min(int(tx2), w), min(int(ty2), h)
    if xi2 - xi1 < 12 or yi2 - yi1 < 12:
        return None
    return image[yi1:yi2, xi1:xi2]
