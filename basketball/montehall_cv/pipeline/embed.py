"""Appearance embeddings for tracklet crops.

CropEmbedder (SigLIP2, Apache-2.0): unsupervised team clustering (the
roboflow/sports recipe) and the appearance fallback in offline tracklet
association. NOT metric-learned — same-team players embed similarly.

ReidEmbedder (OSNet arch via torchreid MIT, OUR from-scratch weights):
metric-learned player identity — the association gate embedder when
weights are available (design D9). Inference preprocessing mirrors
train_reid*'s eval transform exactly: PIL resize 256x128 + ToTensor,
no mean/std normalization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_ID = "google/siglip2-base-patch16-224"
REID_INPUT_HW = (256, 128)


class CropEmbedder:
    def __init__(self, device: str = "cuda", batch_size: int = 64) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self._device = torch.device(device)
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        self._model = AutoModel.from_pretrained(MODEL_ID).to(self._device).eval().half()
        self._batch_size = batch_size

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """RGB uint8 crops -> L2-normalized (n, d) float32 embeddings."""
        torch = self._torch
        out: list[np.ndarray] = []
        for start in range(0, len(crops), self._batch_size):
            chunk = crops[start : start + self._batch_size]
            inputs = self._processor(images=chunk, return_tensors="pt")
            pixel_values = inputs["pixel_values"].half().to(self._device)
            with torch.no_grad():
                features = self._model.get_image_features(pixel_values=pixel_values)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            out.append(features.float().cpu().numpy())
        if not out:
            return np.empty((0, 0), dtype=np.float32)
        emb = np.concatenate(out)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return (emb / np.clip(norms, 1e-9, None)).astype(np.float32)


class ReidEmbedder:
    def __init__(self, weights: Path, arch: str = "osnet_x1_0",
                 device: str = "cuda", batch_size: int = 64) -> None:
        import torch
        from torchreid.models import build_model
        from torchvision import transforms

        self._torch = torch
        self._device = torch.device(device)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        # Head shape must match the checkpoint to load strictly; the head is
        # unused at inference (eval-mode forward returns features).
        n_classes = state["classifier.weight"].shape[0]
        model = build_model(arch, num_classes=n_classes, pretrained=False)
        model.load_state_dict(state)
        self._model = model.to(self._device).eval()
        self._tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(REID_INPUT_HW),
            transforms.ToTensor(),
        ])
        self._batch_size = batch_size

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """RGB uint8 crops -> L2-normalized (n, d) float32 embeddings."""
        torch = self._torch
        out: list[np.ndarray] = []
        for start in range(0, len(crops), self._batch_size):
            chunk = crops[start : start + self._batch_size]
            batch = torch.stack([self._tf(c) for c in chunk]).to(self._device)
            with torch.no_grad():
                features = self._model(batch)
            out.append(features.float().cpu().numpy())
        if not out:
            return np.empty((0, 0), dtype=np.float32)
        emb = np.concatenate(out)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return (emb / np.clip(norms, 1e-9, None)).astype(np.float32)


def crop_bbox(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray | None:
    """Clamped bbox crop; None if degenerate after clamping."""
    h, w = image.shape[:2]
    xi1, yi1 = max(int(x1), 0), max(int(y1), 0)
    xi2, yi2 = min(int(x2), w), min(int(y2), h)
    if xi2 - xi1 < 8 or yi2 - yi1 < 8:
        return None
    return image[yi1:yi2, xi1:xi2]
