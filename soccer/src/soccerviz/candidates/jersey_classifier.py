"""Permissive whole-crop jersey-number classifier with an abstain class, at broadcast scale.

The digit-detector line died on scale: broadcast torsos are about 40 px wide and a
digit under 10 px, while every permissive digit box set is 90 px. This model reads
the whole torso crop as one of 100 numbers or `illegible`, and is trained with the
crop downscaled to broadcast torso heights (16 to 64 px) before it is fed at the
network input size, so the training scale matches the deployment scale by
construction. Sources, all CC BY 4.0: the Roboflow Universe
`taiseis-workspace/jersey-number` classification export (number tiles with
`no number` and `unreadble` classes) and the Pusan digit-box export (whole numbers
read left to right from its digit boxes; regions away from the digits give
number-free negatives). Code and ImageNet weights: timm, Apache-2.0.

Per-crop output keeps the full distribution so `vision.track_identity` can restrict
it to a lineup and aggregate it over a track; the per-crop contract mirrors
`jersey_digits.infer` so the 200-crop evaluator consumes it unchanged.
"""

from __future__ import annotations

import importlib.metadata
import json
import time
from pathlib import Path

import cv2
import numpy as np

from soccerviz.core.assets import sha256

NUM_NUMBERS = 100
ILLEGIBLE = NUM_NUMBERS  # class index of the abstain class
NUM_CLASSES = NUM_NUMBERS + 1
INPUT_SIZE = 128
BACKBONE = "resnet34"
PRETRAINED_TAG = "resnet34.a1_in1k"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TORSO_HEIGHT_RANGE = (16, 64)  # broadcast torso heights the crop is downscaled to in training
ILLEGIBLE_FOLDERS = {"no number", "unreadble"}
DROPPED_FOLDERS = {"00"}  # not a football number; 27 tiles
PUSAN_MAX_DIGITS = 2
PUSAN_MARGIN = (1.6, 1.4)  # crop half-extent as a multiple of the digit block's width, height
DEFAULT_THRESHOLDS = {"min_number_probability": 0.5, "min_legibility": 0.5}
TORSO_RULE = (1 / 6, 2 / 3)  # fraction of the person box kept: from the top sixth to two thirds
DETECTION_ROLES = ("player", "goalkeeper")
MIN_TORSO_PX = 12
RENDER_HEIGHT = 256  # torso is upscaled to this before a number is drawn, then degraded
RENDER_NUMBER_FRACTION = (0.22, 0.42)  # digit height as a fraction of the torso height
RENDER_CENTRE_Y = (0.28, 0.5)  # where the number sits on the back
SYNTHETIC_NUMBERS = tuple(range(1, 100))  # flat prior over football numbers; 0 is not used


def folder_records(export_dir, split):
    """Records of a Roboflow classification (`folder`) export split: path and class index."""
    folder = Path(export_dir) / split
    records = []
    for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        name = class_dir.name
        if name in DROPPED_FOLDERS:
            continue
        if name in ILLEGIBLE_FOLDERS:
            label = ILLEGIBLE
        elif name.isdigit() and 0 <= int(name) < NUM_NUMBERS:
            label = int(name)
        else:
            raise ValueError(f"{class_dir}: unexpected class folder")
        for image in sorted(class_dir.glob("*.jpg")):
            records.append({"image_path": str(image), "label": label, "source": "taiseis"})
    if not records:
        raise ValueError(f"{folder}: no class folders with images")
    return records


def pusan_records(export_dir, split):
    """Whole-number records from the Pusan digit-box export, plus number-free negatives.

    The number is the digit labels ordered left to right; images with more than
    `PUSAN_MAX_DIGITS` digits, or a leading zero on a two-digit number, are skipped.
    Each record carries the crop region in image pixels; a negative region is the
    same-sized window placed below the digits, where the shirt has no number.
    """
    folder = Path(export_dir) / split
    data = json.loads((folder / "_annotations.coco.json").read_text())
    names = {c["id"]: c["name"] for c in data["categories"]}
    images = {i["id"]: i for i in data["images"]}
    per_image = {}
    for annotation in data["annotations"]:
        if names[annotation["category_id"]].isdigit():
            per_image.setdefault(annotation["image_id"], []).append(annotation)
    records = []
    for image_id, digits in per_image.items():
        if len(digits) > PUSAN_MAX_DIGITS:
            continue
        digits = sorted(digits, key=lambda a: a["bbox"][0] + a["bbox"][2] / 2)
        text = "".join(names[a["category_id"]] for a in digits)
        if len(text) == 2 and text[0] == "0":
            continue
        image = images[image_id]
        boxes = np.array([a["bbox"] for a in digits], dtype=float)  # x, y, w, h
        x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
        x1, y1 = (boxes[:, 0] + boxes[:, 2]).max(), (boxes[:, 1] + boxes[:, 3]).max()
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        half_w, half_h = PUSAN_MARGIN[0] * w, PUSAN_MARGIN[1] * h
        region = _clip_region(cx, cy, half_w, half_h, image["width"], image["height"])
        below = _clip_region(cx, cy + 3.2 * h, half_w, half_h, image["width"], image["height"])
        path = str(folder / image["file_name"])
        records.append(
            {"image_path": path, "label": int(text), "region": region, "source": "pusan"}
        )
        if below[3] - below[1] >= 0.8 * (region[3] - region[1]):
            records.append(
                {
                    "image_path": path,
                    "label": ILLEGIBLE,
                    "region": below,
                    "source": "pusan-negative",
                }
            )
    if not records:
        raise ValueError(f"{folder}: no digit annotations")
    return records


def _clip_region(cx, cy, half_w, half_h, width, height):
    x0, x1 = max(0, round(cx - half_w)), min(width, round(cx + half_w))
    y0, y1 = max(0, round(cy - half_h)), min(height, round(cy + half_h))
    return [x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)]


def load_crop(record):
    """BGR crop of one record: the whole tile, or the record's region of the image."""
    image = cv2.imread(record["image_path"])
    if image is None:
        raise FileNotFoundError(record["image_path"])
    if "region" in record:
        x0, y0, x1, y1 = record["region"]
        image = image[y0:y1, x0:x1]
    if image.size == 0:
        raise ValueError(f"Empty crop for {record['image_path']}")
    return image


def detection_torso_records(export_dir, split, roles=DETECTION_ROLES):
    """Torso regions of every player and goalkeeper box in a Roboflow football detection export.

    These are real broadcast crops at deployment scale with no number label. Nearly
    all show no readable number (viewpoint, motion, distance), so they stand in for
    the illegible class as they are, and they are the backgrounds on which
    `render_number` draws synthetic positives; the one legible minority is label
    noise on the illegible side, stated in the doc.
    """
    folder = Path(export_dir) / split
    data = json.loads((folder / "_annotations.coco.json").read_text())
    names = {c["id"]: c["name"] for c in data["categories"]}
    images = {i["id"]: i for i in data["images"]}
    records = []
    for annotation in data["annotations"]:
        if names[annotation["category_id"]] not in roles:
            continue
        image = images[annotation["image_id"]]
        x, y, w, h = annotation["bbox"]
        y0, y1 = y + TORSO_RULE[0] * h, y + TORSO_RULE[1] * h
        region = _clip_region(
            (2 * x + w) / 2, (y0 + y1) / 2, w / 2, (y1 - y0) / 2, image["width"], image["height"]
        )
        if region[3] - region[1] < MIN_TORSO_PX or region[2] - region[0] < MIN_TORSO_PX:
            continue
        records.append(
            {
                "image_path": str(folder / image["file_name"]),
                "label": ILLEGIBLE,
                "region": region,
                "source": "broadcast-torso",
                "annotation_id": annotation["id"],
            }
        )
    if not records:
        raise ValueError(f"{folder}: no player boxes")
    return records


def cut_torso_pool(records, out):
    """Write each record's torso crop once as PNG; returns new records pointing at the files.

    Decoding a 1080p frame per item is too slow for a dataloader; the pool is cut
    once, shipped, and read like any other tile.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = {}
    pooled = []
    for record in records:
        path = out / f"{Path(record['image_path']).stem[:40]}-{record['annotation_id']}.png"
        if not path.exists():
            image = cache.get(record["image_path"])
            if image is None:
                image = cv2.imread(record["image_path"])
                if image is None:
                    raise FileNotFoundError(record["image_path"])
                cache = {record["image_path"]: image}
            x0, y0, x1, y1 = record["region"]
            if not cv2.imwrite(str(path), image[y0:y1, x0:x1]):
                raise RuntimeError(f"Failed to write {path}")
        pooled.append(
            {k: v for k, v in record.items() if k != "region"} | {"image_path": str(path)}
        )
    return pooled


def load_fonts(font_dir):
    """TrueType faces for rendering; PIL is imported here only."""
    from PIL import ImageFont

    paths = sorted(Path(font_dir).glob("*.ttf"))
    if not paths:
        raise FileNotFoundError(f"No .ttf fonts under {font_dir}")
    return [(str(p), ImageFont) for p in paths]


def _text_colours(torso_bgr, rng):
    """Light digits on dark kit, dark on light, with a random tint; an outline half the time."""
    luminance = float(cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2GRAY).mean())
    light = luminance < 128
    base = rng.integers(200, 256, 3) if light else rng.integers(0, 60, 3)
    if rng.random() < 0.3:  # coloured numbers: a saturated hue instead of white/black
        base = rng.integers(0, 256, 3)
    outline = None
    if rng.random() < 0.5:
        outline = rng.integers(0, 60, 3) if light else rng.integers(200, 256, 3)
    return tuple(int(v) for v in base), (
        None if outline is None else tuple(int(v) for v in outline)
    )


def render_number(torso_bgr, number, fonts, rng):
    """Draw `number` in a kit font onto a real torso crop; returns a BGR crop at RENDER_HEIGHT.

    The torso is upscaled first so the digits are drawn with real edges and then go
    through the same downscale degradation as every other crop. Perspective and
    rotation are mild; the number sits where a back number sits. The un-rendered
    torso is the matched illegible example, so the only cue separating the two
    classes is the digits themselves.
    """
    from PIL import Image, ImageDraw, ImageFont

    height, width = torso_bgr.shape[:2]
    scale = RENDER_HEIGHT / height
    canvas = cv2.resize(
        torso_bgr, (max(8, round(width * scale)), RENDER_HEIGHT), interpolation=cv2.INTER_CUBIC
    )
    image = Image.fromarray(canvas[:, :, ::-1])
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font_path, _ = fonts[int(rng.integers(len(fonts)))]
    digit_height = float(rng.uniform(*RENDER_NUMBER_FRACTION)) * RENDER_HEIGHT
    font = ImageFont.truetype(font_path, int(digit_height * 1.3))
    text = str(number)
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = x1 - x0, y1 - y0
    max_w = 0.9 * image.size[0]
    if text_w > max_w:
        font = ImageFont.truetype(font_path, max(8, int(font.size * max_w / text_w)))
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = x1 - x0, y1 - y0
    cx = image.size[0] / 2 + rng.uniform(-0.08, 0.08) * image.size[0]
    cy = float(rng.uniform(*RENDER_CENTRE_Y)) * RENDER_HEIGHT
    fill, outline = _text_colours(torso_bgr, rng)
    position = (cx - text_w / 2 - x0, cy - text_h / 2 - y0)
    if outline is not None:
        draw.text(
            position,
            text,
            font=font,
            fill=outline + (255,),
            stroke_width=max(1, int(digit_height * 0.06)),
            stroke_fill=outline + (255,),
        )
    draw.text(position, text, font=font, fill=fill + (255,))
    layer = layer.rotate(float(rng.uniform(-10, 10)), resample=Image.BICUBIC, center=(cx, cy))
    if rng.random() < 0.5:  # mild perspective: shear the layer horizontally
        shear = float(rng.uniform(-0.25, 0.25))
        layer = layer.transform(
            layer.size, Image.AFFINE, (1, shear, -shear * cy, 0, 1, 0), resample=Image.BICUBIC
        )
    alpha = float(rng.uniform(0.8, 1.0))
    layer = Image.fromarray((np.asarray(layer) * np.array([1, 1, 1, alpha])).astype(np.uint8))
    composed = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
    out = np.asarray(composed)[:, :, ::-1].copy()
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.5, 1.5)))
    return out


def degrade(crop, rng, torso_height_range=TORSO_HEIGHT_RANGE):
    """Broadcast-scale degradation: downscale to a torso height, blur, JPEG, jitter, upscale."""
    height, width = crop.shape[:2]
    target = int(rng.integers(torso_height_range[0], torso_height_range[1] + 1))
    scale = target / height
    small = cv2.resize(crop, (max(4, round(width * scale)), target), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        sigma = float(rng.uniform(0.3, 1.2))
        small = cv2.GaussianBlur(small, (0, 0), sigma)
    if rng.random() < 0.3:
        k = int(rng.integers(2, 5))
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k  # horizontal motion blur
        small = cv2.filter2D(small, -1, kernel)
    if rng.random() < 0.7:
        quality = int(rng.integers(30, 90))
        ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            small = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    alpha = float(rng.uniform(0.7, 1.3))
    beta = float(rng.uniform(-30, 30))
    small = np.clip(small.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # random colour cast, so the grayscale Pusan tiles are not a tell
        tint = rng.uniform(0.7, 1.3, size=3).astype(np.float32)
        small = np.clip(small.astype(np.float32) * tint, 0, 255).astype(np.uint8)
    return small


def jitter_crop(crop, rng, max_fraction=0.15, max_zoom_out=0.5):
    """Random crop or outward padding per side, so the number is neither centred nor a fixed size.

    Source tiles are tight on the number; a broadcast torso crop is not. Each side
    moves inward by up to `max_fraction` or outward by up to `max_zoom_out` of the
    crop size, with replicated borders standing in for the surrounding shirt.
    """
    height, width = crop.shape[:2]
    zoom = rng.random() < 0.6
    lo, hi = (-max_zoom_out, max_fraction) if zoom else (-max_fraction, max_fraction)
    dx0, dx1, dy0, dy1 = (int(rng.uniform(lo, hi) * s) for s in (width, width, height, height))
    padded = cv2.copyMakeBorder(crop, abs(dy0), abs(dy1), abs(dx0), abs(dx1), cv2.BORDER_REPLICATE)
    y0, x0 = abs(dy0) + dy0, abs(dx0) + dx0
    y1, x1 = y0 + height + dy1 - dy0, x0 + width + dx1 - dx0
    out = padded[max(0, y0) : max(1, y1), max(0, x0) : max(1, x1)]
    return out if out.size else crop


def preprocess(crop_bgr, size=INPUT_SIZE):
    """BGR uint8 crop → normalised CHW float32 at the square network input, aspect kept by padding."""
    height, width = crop_bgr.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(
        crop_bgr,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    y0, x0 = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    normalised = (canvas[:, :, ::-1].astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalised.transpose(2, 0, 1))


def build_network(pretrained=False):
    """timm ResNet-34 with a 101-way head. Torch is imported here only."""
    import timm

    return timm.create_model(
        PRETRAINED_TAG if pretrained else BACKBONE, pretrained=pretrained, num_classes=NUM_CLASSES
    )


def softmax(logits):
    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def read_distribution(probabilities, thresholds=DEFAULT_THRESHOLDS):
    """One crop's decision from its 101-way distribution: number, score, legibility, status."""
    probabilities = np.asarray(probabilities, dtype=float)
    legibility = float(1.0 - probabilities[ILLEGIBLE])
    number = int(np.argmax(probabilities[:NUM_NUMBERS]))
    score = float(probabilities[number])
    accepted = (
        score >= thresholds["min_number_probability"] and legibility >= thresholds["min_legibility"]
    )
    return {
        "pred_number": number,
        "pred_score": score,
        "legibility": legibility,
        "status": "provisional" if accepted else "abstain",
        "jersey_number": number if accepted else None,
        "reason": "whole_crop_classifier_no_named_identity"
        if accepted
        else "below_probability_or_legibility_floor_not_a_legibility_label",
    }


class JerseyClassifier:
    """Trained checkpoint; distributions(crops_bgr) → (n, 101) probabilities."""

    def __init__(self, checkpoint, device="cuda:0", batch_size=64):
        import torch

        self.torch, self.device, self.batch_size = torch, device, batch_size
        self.net = build_network(pretrained=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state.get("model", state), strict=True)
        self.net.to(device).eval()
        self.temperature = float(state.get("temperature", 1.0))
        self.metadata = {
            "name": f"timm-{BACKBONE}-jersey-classifier",
            "temperature": self.temperature,
            "checkpoints": {
                "jersey": {"path": str(Path(checkpoint).resolve()), "sha256": sha256(checkpoint)}
            },
            "classes": NUM_CLASSES,
            "input_size": INPUT_SIZE,
            "device": device,
            "versions": {p: importlib.metadata.version(p) for p in ("timm", "torch")},
        }

    def distributions(self, crops_bgr):
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(crops_bgr), self.batch_size):
                batch = np.stack(
                    [preprocess(c) for c in crops_bgr[start : start + self.batch_size]]
                )
                logits = (
                    self.net(self.torch.from_numpy(batch).to(self.device)).float().cpu().numpy()
                )
                outputs.append(softmax(logits / self.temperature))
        return np.concatenate(outputs) if outputs else np.zeros((0, NUM_CLASSES))


def infer(
    crop_manifest, checkpoint, out, device="cuda:0", thresholds=DEFAULT_THRESHOLDS, model=None
):
    """Read every crop of a jersey crop manifest; writes out/predictions.json and distributions.npz.

    A manifest crop may carry `torso_bounds_in_crop_xyxy` (the frozen 200-crop set records
    `model_torso_bounds_in_crop_xyxy`); when present the torso region is classified,
    otherwise the whole crop.
    """
    manifest = json.loads(Path(crop_manifest).read_text())
    root = Path(crop_manifest).parent
    model = model or JerseyClassifier(checkpoint, device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    crops, ids = [], []
    for row in manifest["crops"]:
        path = root / row["crop_path"]
        if sha256(path) != row["crop_sha256"]:
            raise ValueError(f"Crop hash mismatch: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable crop: {path}")
        bounds = row.get("torso_bounds_in_crop_xyxy") or row.get("model_torso_bounds_in_crop_xyxy")
        if bounds:
            x0, y0, x1, y1 = bounds
            image = image[y0:y1, x0:x1]
        crops.append(image)
        ids.append(row["crop_id"])
    probabilities = model.distributions(crops)
    results = [
        {"crop_id": crop_id, **read_distribution(p, thresholds)}
        for crop_id, p in zip(ids, probabilities)
    ]
    np.savez_compressed(
        out / "distributions.npz", crop_ids=np.array(ids), probabilities=probabilities
    )
    output = {
        "schema": "jersey-predictions/v1",
        "backend": "jersey-classifier",
        "crop_manifest_sha256": sha256(crop_manifest),
        "complete_crop_manifest": len(results) == len(manifest["crops"]),
        "ground_truth_used_in_inference": False,
        "classifier": model.metadata,
        "thresholds": dict(thresholds),
        "device": str(device),
        "inference_and_output_s": time.perf_counter() - started,
        "distributions": "distributions.npz",
        "abstention_policy": {
            **thresholds,
            "rule": "number probability and legibility (1 - p_illegible) both at or above their floors",
            "calibrated": False,
        },
        "predictions": results,
    }
    (out / "predictions.json").write_text(json.dumps(output, indent=2) + "\n")
    return output
