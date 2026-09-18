"""Detection stage: Detector protocol + RF-DETR implementation.

RF-DETR pretrained COCO weights give person + sports-ball zero-shot, which
carries the skeleton until the basketball fine-tune lands (design D7/D9).
Classes are selected by NAME from the model's own id->name mapping so COCO
index conventions can't silently shift under us.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from montehall_cv.store.records import DetClass

CLASS_NAME_MAP: dict[str, DetClass] = {
    "person": DetClass.PERSON,
    "sports ball": DetClass.BALL,
}


@dataclass(frozen=True)
class FrameDetections:
    xyxy: np.ndarray  # (n,4) float32 pixel
    conf: np.ndarray  # (n,) float32
    cls: np.ndarray  # (n,) int8 DetClass values


class Detector(Protocol):
    def detect(self, images: list[np.ndarray]) -> list[FrameDetections]: ...


class RFDetrDetector:
    def __init__(
        self,
        threshold: float = 0.4,
        resolution: int | None = None,
        batch_size: int = 16,
        weights: Path | None = None,
    ) -> None:
        import torch
        from rfdetr import RFDETRMedium

        kwargs: dict = {}
        if resolution is not None:
            kwargs["resolution"] = resolution
        if weights is not None:
            kwargs["pretrain_weights"] = str(weights)
        self._model = RFDETRMedium(**kwargs)
        self._model.optimize_for_inference(batch_size=batch_size, dtype=torch.float16)
        self._batch_size = batch_size
        self._threshold = threshold
        # Fine-tuned checkpoints emit 0-BASED indices into the checkpoint's
        # own args.class_names (probed live 2026-07-08: ids [0,2] = player,
        # rim — the previous 1-based assumption silently dropped every player
        # and relabeled ball/rim, poisoning the v1 harvest). Read the names
        # from the checkpoint so the mapping can never drift again; pretrained
        # COCO emits 1-based COCO annotation ids.
        self._id_to_class = (
            _finetuned_class_map(weights)
            if weights is not None
            else _build_class_filter()
        )

    def detect(self, images: list[np.ndarray]) -> list[FrameDetections]:
        if not images:
            return []
        n_real = len(images)
        if n_real > self._batch_size:
            raise ValueError(f"got {n_real} images, compiled for batch {self._batch_size}")
        padded = images + [images[-1]] * (self._batch_size - n_real)
        results = self._model.predict(padded, threshold=self._threshold)
        if not isinstance(results, list):
            results = [results]
        return [self._convert(r) for r in results[:n_real]]

    def _convert(self, detections) -> FrameDetections:
        xyxy = np.asarray(detections.xyxy, dtype=np.float32)
        conf = np.asarray(detections.confidence, dtype=np.float32)
        class_ids = np.asarray(detections.class_id)
        mapped = np.array(
            [self._id_to_class.get(int(c), -1) for c in class_ids], dtype=np.int8
        )
        keep = mapped >= 0
        return FrameDetections(xyxy=xyxy[keep], conf=conf[keep], cls=mapped[keep])


FINETUNE_NAME_MAP: dict[str, DetClass] = {
    "player": DetClass.PERSON,
    "ball": DetClass.BALL,
    "rim": DetClass.RIM,
}


def _finetuned_class_map(weights: Path) -> dict[int, int]:
    """0-based index -> DetClass, from the checkpoint's own class_names.

    Lightning full checkpoints (last.ckpt) carry no args; fall back to the
    montehall fine-tune contract ['player', 'ball', 'rim'].
    """
    import torch

    names = ["player", "ball", "rim"]
    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        args = ckpt.get("args") if isinstance(ckpt, dict) else None
        stored = getattr(args, "class_names", None) if args is not None else None
        if stored:
            names = list(stored)
    except Exception:
        pass  # unreadable metadata -> contract fallback
    mapping = {
        idx: int(FINETUNE_NAME_MAP[str(name).lower()])
        for idx, name in enumerate(names)
        if str(name).lower() in FINETUNE_NAME_MAP
    }
    if not mapping:
        raise RuntimeError(f"no known class names in checkpoint {weights}: {names}")
    return mapping


def _build_class_filter() -> dict[int, int]:
    """Map predict()'s class ids to DetClass by name.

    predict() emits 1-based COCO annotation ids (person=1, sports ball=37) —
    verified live; these match rfdetr's COCO_CLASSES dict, NOT the 0-indexed
    model.class_names list.
    """
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES
    except ImportError:
        from rfdetr.util.coco_classes import COCO_CLASSES

    items = COCO_CLASSES.items() if isinstance(COCO_CLASSES, dict) else enumerate(COCO_CLASSES)
    mapping = {
        int(class_id): int(CLASS_NAME_MAP[str(name).lower()])
        for class_id, name in items
        if str(name).lower() in CLASS_NAME_MAP
    }
    if len(mapping) != len(CLASS_NAME_MAP):
        raise RuntimeError(f"COCO class map incomplete: resolved {mapping}")
    return mapping
