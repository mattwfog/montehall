"""Isolated detector experiments; importing this module does not load torch.

All adapters accept uint8 BGR images and return original-image pixel boxes.
Generic COCO person/ball predictions never acquire soccer-role labels.
"""

from __future__ import annotations

import importlib.metadata
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from soccerviz.core.assets import sha256

SOCCER_ROLES = {"player", "goalkeeper", "referee"}


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def validate_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input must be a uint8 BGR NumPy array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("Input must have nonempty H x W x 3 shape")


def normalize_detections(boxes, scores, names, *, mode, width, height, threshold):
    """Validate model output, clip to image bounds, and use an explicit class ontology."""
    if mode not in {"coco", "soccer"}:
        raise ValueError("class mode must be coco or soccer")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    boxes, scores = _array(boxes), _array(scores)
    if boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.ndim != 1:
        raise ValueError("Expected boxes Nx4 and scores N")
    if not (len(boxes) == len(scores) == len(names)):
        raise ValueError("Mismatched detection fields")
    detections = []
    for box, score, raw_name in zip(boxes, scores, names, strict=True):
        if not np.isfinite(box).all() or not math.isfinite(float(score)):
            raise ValueError("Detector returned a nonfinite box or score")
        if not 0 <= score <= 1:
            raise ValueError("Detector confidence is outside [0, 1]")
        if box[2] < box[0] or box[3] < box[1]:
            raise ValueError("Detector returned reversed xyxy coordinates")
        name = str(raw_name).strip().lower()
        if mode == "coco":
            if name in SOCCER_ROLES:
                raise ValueError("Soccer role returned in COCO mode; check checkpoint class mode")
            label = {"person": "person", "sports ball": "ball"}.get(name)
            role = None
        else:
            if name not in SOCCER_ROLES | {"ball", "person"}:
                raise ValueError(f"Unexpected soccer class {name!r}; check checkpoint class mode")
            label = "ball" if name == "ball" else "person"
            role = name if name in SOCCER_ROLES else None
        if label is None or score < threshold:
            continue
        clipped = np.clip(box.astype(float), [0, 0, 0, 0], [width, height, width, height])
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        item = {
            "bbox_xyxy": clipped.tolist(),
            "score": float(score),
            "label": label,
            "source_class": name,
        }
        if role:
            item["role"] = role
        detections.append(item)
    return detections


def nms(detections, iou_threshold=0.5):
    """Class-wise NMS for duplicate detections where overlapping tiles meet."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("NMS IoU threshold must be in [0, 1]")
    ordered = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep = []
    for item in ordered:
        box = np.asarray(item["bbox_xyxy"], dtype=float)
        duplicate = False
        for prior in keep:
            if prior["label"] != item["label"]:
                continue
            other = np.asarray(prior["bbox_xyxy"], dtype=float)
            intersection = np.maximum(
                0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2])
            ).prod()
            union = (box[2:] - box[:2]).prod() + (other[2:] - other[:2]).prod() - intersection
            if union > 0 and intersection / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            keep.append(item)
    return keep


def tile_regions(width, height):
    """The current pipeline's four overlapping 60% tiles, in source coordinates."""
    return [
        (x0, y0, x1, y1)
        for y0, y1 in [(0, max(1, int(height * 0.6))), (int(height * 0.4), height)]
        for x0, x1 in [(0, max(1, int(width * 0.6))), (int(width * 0.4), width)]
    ]


def _sync(device):
    if str(device).startswith("cuda") or str(device).isdigit():
        import torch

        torch.cuda.synchronize(device=int(device) if str(device).isdigit() else device)


def _versions(*names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _checkpoint(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


class UltralyticsDetector:
    def __init__(
        self,
        checkpoint,
        *,
        mode="soccer",
        device="cuda:0",
        resolution=1280,
        threshold=0.25,
        ball_checkpoint=None,
        tiled_ball=False,
        ball_selection="all",
    ):
        from ultralytics import YOLO

        if ball_checkpoint is not None and mode != "soccer":
            raise ValueError("A dedicated soccer ball checkpoint requires soccer mode")
        if tiled_ball and ball_checkpoint is None:
            raise ValueError("tiled_ball requires a dedicated ball checkpoint")
        if ball_selection not in {"all", "top1"}:
            raise ValueError("ball_selection must be all or top1")
        self.device, self.resolution, self.threshold = device, resolution, threshold
        self.mode, self.tiled_ball = mode, tiled_ball
        self.ball_selection = ball_selection
        checkpoint_info = _checkpoint(checkpoint)
        self.model = YOLO(str(checkpoint_info["path"]))
        self.ball_model = None
        checkpoints = {"detector": checkpoint_info}
        if ball_checkpoint is not None:
            checkpoints["ball"] = _checkpoint(ball_checkpoint)
            self.ball_model = YOLO(str(checkpoints["ball"]["path"]))
        self.metadata = {
            "name": "ultralytics",
            "class_mode": mode,
            "architecture": str(getattr(self.model.model, "yaml", {}).get("yaml_file", "unknown")),
            "versions": _versions("ultralytics", "torch", "torchvision", "numpy"),
            "checkpoints": checkpoints,
            "device": device,
            "resolution": resolution,
            "confidence_threshold": threshold,
            "input_color": "BGR",
            "tiled_ball": tiled_ball,
            "ball_selection": ball_selection,
            "ball_resolution": 640 if self.ball_model else None,
            "class_names": dict(self.model.names),
            "ball_class_names": dict(self.ball_model.names) if self.ball_model else None,
            "precision": "float32",
            "postprocess": "Ultralytics defaults; tile ball NMS IoU=0.5",
            "control_note": "Detector-only control adapted to a shared confidence threshold. "
            "Existing pipeline uses person0.30, ball0.25, and a single highest-score tiled ball. "
            "Ball selection all retains every NMS candidate; top1 matches existing selection.",
        }
        self.last_model_ms = 0.0

    def _predict(self, model, image, resolution, mode):
        _sync(self.device)
        start = time.perf_counter()
        result = model.predict(
            image,
            imgsz=resolution,
            conf=self.threshold,
            device=self.device,
            verbose=False,
            half=False,
        )[0]
        _sync(self.device)
        self.last_model_ms += (time.perf_counter() - start) * 1000
        boxes = result.boxes
        ids = _array(boxes.cls).astype(int)
        names = [result.names[int(i)] for i in ids]
        return normalize_detections(
            boxes.xyxy,
            boxes.conf,
            names,
            mode=mode,
            width=image.shape[1],
            height=image.shape[0],
            threshold=self.threshold,
        )

    def predict(self, image):
        validate_image(image)
        self.last_model_ms = 0.0
        height, width = image.shape[:2]
        result = self._predict(self.model, image, self.resolution, self.mode)
        transforms = [
            {
                "model": "detector",
                "crop_xyxy": [0, 0, width, height],
                "resize": "Ultralytics letterbox",
                "imgsz": self.resolution,
                "output_coordinates": "original-image pixels",
            }
        ]
        if self.ball_model is not None:
            result = [d for d in result if d["label"] != "ball"]
            regions = tile_regions(width, height) if self.tiled_ball else [(0, 0, width, height)]
            balls = []
            for x0, y0, x1, y1 in regions:
                crop = image[y0:y1, x0:x1]
                for detection in self._predict(self.ball_model, crop, 640, "soccer"):
                    if detection["label"] != "ball":
                        continue
                    detection["bbox_xyxy"] = (
                        np.asarray(detection["bbox_xyxy"]) + [x0, y0, x0, y0]
                    ).tolist()
                    balls.append(detection)
                transforms.append(
                    {
                        "model": "ball",
                        "crop_xyxy": [x0, y0, x1, y1],
                        "resize": "Ultralytics letterbox",
                        "imgsz": 640,
                        "output_coordinates": "original-image pixels",
                    }
                )
            balls = nms(balls)
            result.extend(balls[:1] if self.ball_selection == "top1" else balls)
        return result, transforms


class RFDETRDetector:
    def __init__(
        self,
        *,
        size="medium",
        checkpoint=None,
        mode="coco",
        device="cuda:0",
        resolution=None,
        threshold=0.25,
    ):
        import rfdetr

        if importlib.metadata.version("rfdetr") != "1.10.1":
            raise RuntimeError("This adapter is validated for pinned rfdetr==1.10.1")
        if size not in {"small", "medium"}:
            raise ValueError("RF-DETR size must be small or medium")
        if mode == "soccer" and checkpoint is None:
            raise ValueError("Soccer RF-DETR mode requires a soccer-trained local checkpoint")
        if mode not in {"coco", "soccer"}:
            raise ValueError("class mode must be coco or soccer")
        model_class = rfdetr.RFDETRSmall if size == "small" else rfdetr.RFDETRMedium
        config: dict[str, Any] = {"device": device}
        if resolution is not None:
            if resolution <= 0 or resolution % 32:
                raise ValueError("Small/Medium RF-DETR resolution must be positive/divisible by 32")
            config["resolution"] = resolution
        if checkpoint is not None:
            checkpoint_info = _checkpoint(checkpoint)
            self.model = model_class.from_checkpoint(checkpoint_info["path"], **config)
            if not isinstance(self.model, model_class):
                raise ValueError("Loaded RF-DETR checkpoint size differs from requested size")
        else:
            self.model = model_class(**config)
            checkpoint_info = _checkpoint(self.model.model_config.pretrain_weights)
        self.device, self.threshold, self.mode = device, threshold, mode
        self.resolution = int(self.model.model_config.resolution)
        self.metadata = {
            "name": f"rf-detr-{size}",
            "class_mode": mode,
            "training_domain": "COCO initialization; not soccer fine-tuned"
            if mode == "coco"
            else "user-supplied soccer checkpoint",
            "versions": _versions("rfdetr", "transformers", "supervision", "torch", "torchvision"),
            "checkpoints": {"detector": checkpoint_info},
            "device": device,
            "resolution": self.resolution,
            "confidence_threshold": threshold,
            "input_color": "RGB (converted from BGR)",
            "class_names": self.model.class_names,
            "precision": "float32",
            "optimized": False,
            "postprocess": "RF-DETR native predict",
            "no_object_filter": {
                "source_class": "__background__",
                "policy": "Exclude only RF-DETR's documented no-object sentinel before soccer mapping",
                "excluded_rows_including_warmup": 0,
            },
        }
        self.last_model_ms = 0.0
        self.last_background_count = 0

    def predict(self, image):
        validate_image(image)
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        _sync(self.device)
        start = time.perf_counter()
        result = self.model.predict(rgb, threshold=self.threshold, include_source_image=False)
        _sync(self.device)
        self.last_model_ms = (time.perf_counter() - start) * 1000
        names = result.data.get("class_name")
        if names is None:
            raise ValueError("RF-DETR1.10.1 output lacks explicit class_name metadata")
        names = np.asarray(names, dtype=object)
        if names.ndim != 1 or len(names) != len(result.xyxy):
            raise ValueError("RF-DETR class_name metadata is misaligned with boxes")
        background = names == "__background__"
        self.last_background_count = int(background.sum())
        self.metadata["no_object_filter"]["excluded_rows_including_warmup"] += (
            self.last_background_count
        )
        detections = normalize_detections(
            result.xyxy[~background],
            result.confidence[~background],
            names[~background],
            mode=self.mode,
            width=image.shape[1],
            height=image.shape[0],
            threshold=self.threshold,
        )
        return detections, [
            {
                "model": "detector",
                "crop_xyxy": [0, 0, image.shape[1], image.shape[0]],
                "resize": "RF-DETR square resize",
                "target_hw": [self.resolution, self.resolution],
                "scale_xy": [self.resolution / image.shape[1], self.resolution / image.shape[0]],
                "output_coordinates": "original-image pixels",
            }
        ]


def gpu_memory(device):
    """Process allocation peak, not the full shared/unified GPU memory usage."""
    if str(device).startswith("cuda") or str(device).isdigit():
        import torch

        device = int(device) if str(device).isdigit() else device
        return {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device_name": torch.cuda.get_device_name(device),
            "cuda_version": torch.version.cuda,
        }
    return None
