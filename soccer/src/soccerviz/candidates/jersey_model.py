"""Isolated Uncertainty-JNR inference on detector crops, with evaluation kept separate."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CHECKPOINT_SHA256 = "43eb17804e94e012883b37618d70fbab3fb190f58eb851843e38589a9cf80923"
UPSTREAM_REVISION = "f19d9cb90e1a67d5ffe44acbf69fb348fe6525e7"
UPSTREAM_FILE_HASHES = {
    "config.py": "b5f412ec5ec9897a5bd56d692ed6315742f25a4ba415dd67830e9e3247e03de0",
    "configs/base8_reid.yaml": "50eb3074945056101596d3ec79d474b809d6efddf805b9c3f1ea140540484ee8",
    "src/uncertainty_jnr/model.py": "05123bfdcd7c9d9157b9eed95df0eb29dab1dfff1550da626045dbc1b7dda1b1",
    "src/uncertainty_jnr/data.py": "b7db5ef91d297ba8df39956900af2696a4fa42c6284cb8189e6748ea679964e2",
    "src/uncertainty_jnr/augmentation.py": "de36baeb63f5aaa5ce392269000d0c0d5d1449911e23675ea26d5646ab541cdb",
    "src/uncertainty_jnr/utils.py": "511997f13f00ff3c0a45819250242106131f5e37b537efe9e6ba0f25b7dc29e0",
    "src/uncertainty_jnr/__init__.py": "219ec92e97b5038ce89a919dd5a57e53913bd4b46c83fd50b0f022d64d41922b",
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def verify_upstream(root):
    """Validate exported source archives without requiring deployment-time Git metadata."""
    actual = {name: file_sha(Path(root) / name) for name in UPSTREAM_FILE_HASHES}
    if actual != UPSTREAM_FILE_HASHES:
        raise ValueError("Upstream source/config hashes differ from pinned JNR revision")
    return actual


def crop_bounds(box, width, height):
    points = np.asarray(box, dtype=float)
    if points.shape != (4,) or not np.isfinite(points).all():
        raise ValueError("Crop box requires four finite original-pixel coordinates")
    x0, y0, x1, y1 = points
    if x1 <= x0 or y1 <= y0 or width <= 0 or height <= 0:
        raise ValueError("Crop or image dimensions are invalid")
    bounds = [
        max(0, math.floor(x0)),
        max(0, math.floor(y0)),
        min(width, math.ceil(x1)),
        min(height, math.ceil(y1)),
    ]
    if bounds[2] - bounds[0] < 2 or bounds[3] - bounds[1] < 6:
        raise ValueError("Crop too small or wholly outside source image")
    return bounds


def select_crops(manifest, predictions, limit=200):
    """Select uniformly within sequence using only detector output, never GT identities."""
    if limit < 1:
        raise ValueError("Crop limit must be positive")
    frames = {(r["sequence"], r["frame_id"]): r for r in manifest["frames"]}
    if len(frames) != len(manifest["frames"]):
        raise ValueError("Duplicate source frame")
    candidates = defaultdict(list)
    seen = set()
    for row in predictions["frames"]:
        key = row["sequence"], row["frame_id"]
        if key in seen or key not in frames:
            raise ValueError("Duplicate prediction frame or frame outside frozen manifest")
        seen.add(key)
        source = frames[key]
        if row.get("image_sha256") != source["sha256"]:
            raise ValueError("Detector source image hash differs from frozen image")
        for index, detection in enumerate(row["detections"]):
            if detection.get("label") != "person":
                continue
            # Soccer roles here are detector predictions, never source truth.
            if detection.get("role") not in (None, "player", "goalkeeper"):
                continue
            if not math.isfinite(detection["score"]) or not 0 <= detection["score"] <= 1:
                raise ValueError("Invalid detector confidence")
            bounds = crop_bounds(detection["bbox_xyxy"], source["width"], source["height"])
            candidates[key[0]].append(
                {
                    "crop_id": f"{key[0]}-{key[1]:06d}-d{index:03d}",
                    "sequence": key[0],
                    "frame_id": key[1],
                    "timestamp_s": source["timestamp_s"],
                    "source_image_sha256": source["sha256"],
                    "source_image_path": source["image_path"],
                    "source_width": source["width"],
                    "source_height": source["height"],
                    "detection_index": index,
                    "detector_score": detection["score"],
                    "predicted_role": detection.get("role"),
                    "predicted_bbox_xyxy": detection["bbox_xyxy"],
                    "crop_bounds_xyxy": bounds,
                }
            )
    if seen != set(frames):
        raise ValueError("Detector output does not cover the frozen manifest")
    if not candidates:
        raise ValueError("No eligible predicted person crops")
    selected = []
    budget = min(limit, sum(len(v) for v in candidates.values()))
    quotas = {name: 0 for name in candidates}
    while sum(quotas.values()) < budget:
        for name in sorted(candidates):
            if quotas[name] < len(candidates[name]) and sum(quotas.values()) < budget:
                quotas[name] += 1
    for name, group in sorted(candidates.items()):
        group.sort(key=lambda r: (r["frame_id"], r["predicted_bbox_xyxy"][0], r["detection_index"]))
        for index in np.linspace(0, len(group) - 1, quotas[name], dtype=int):
            selected.append(group[index])
    return selected, {name: len(group) for name, group in candidates.items()}


def resolve_path(value, prefixes=()):
    path = Path(value)
    for old, new in prefixes:
        try:
            return Path(new) / path.relative_to(old)
        except ValueError:
            pass
    return path


def prepare(manifest_path, predictions_path, out, limit=200, prefixes=()):
    import cv2

    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads(Path(manifest_path).read_text())
    predictions = json.loads(Path(predictions_path).read_text())
    if predictions.get("manifest_sha256") != file_sha(manifest_path):
        raise ValueError("Detector predictions are not bound to this frozen manifest")
    selected, counts = select_crops(manifest, predictions, limit)
    (out / "crops").mkdir(parents=True)
    for row in selected:
        image_path = resolve_path(row["source_image_path"], prefixes)
        if file_sha(image_path) != row["source_image_sha256"]:
            raise ValueError("Source image hash mismatch")
        image = cv2.imread(str(image_path))
        if image is None or image.shape[:2] != (row["source_height"], row["source_width"]):
            raise ValueError("Unreadable or resized source image")
        x0, y0, x1, y1 = row["crop_bounds_xyxy"]
        crop = image[y0:y1, x0:x1]
        path = out / "crops" / f"{row['crop_id']}.png"
        if not cv2.imwrite(str(path), crop):
            raise RuntimeError("Failed to save original-pixel crop")
        h, w = crop.shape[:2]
        row.update(
            crop_path=f"crops/{path.name}",
            crop_sha256=file_sha(path),
            crop_width=w,
            crop_height=h,
            model_torso_bounds_in_crop_xyxy=[0, h // 6, w, h - h // 3],
        )
    report = {
        "schema": "jersey-crops/v1",
        "crop_source": "predicted_detector_boxes",
        "manifest_sha256": file_sha(manifest_path),
        "detector_predictions_sha256": file_sha(predictions_path),
        "detector_backend": predictions["backend"],
        "candidate_counts": counts,
        "selected_crops": len(selected),
        "sampling": "equal sequence quotas then uniform indices over frame/x-sorted predicted detections",
        "ground_truth_used_in_sampling": False,
        "predicted_track_ids_available": False,
        "identity_sampling_limit": "Unavailable without predicted tracks; repeated people are correlated",
        "transform": "floor/ceil bbox bounds clipped to source; lossless PNG; upstream torso crop, cubic224x224, RGB/127.5-1",
        "crops": selected,
    }
    save_json(out / "crop-manifest.json", report)
    return report


def validate_number_output(number, score, uncertainty, probabilities, alphas=None):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (100,) or not np.isfinite(probabilities).all():
        raise ValueError("Expected 100 finite number probabilities")
    if (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1.0, atol=1e-4):
        raise ValueError("Dirichlet probabilities must sum to one")
    if isinstance(number, bool) or int(number) != number or not 0 <= number <= 99:
        raise ValueError("Jersey number must be an integer from 0 to 99")
    if int(number) != int(probabilities.argmax()) or not np.isclose(
        score, probabilities.max(), atol=1e-5
    ):
        raise ValueError("Predicted number or score contradicts full distribution")
    if not math.isfinite(uncertainty) or not 0 <= uncertainty <= 1:
        raise ValueError("Invalid uncertainty")
    if alphas is not None:
        alpha = np.asarray(alphas, dtype=float)
        if alpha.shape != (100,) or not np.isfinite(alpha).all() or (alpha < 1).any():
            raise ValueError("Invalid Dirichlet concentrations")
        if not np.allclose(probabilities, alpha / alpha.sum(), atol=1e-5):
            raise ValueError("Probabilities contradict Dirichlet concentrations")
        if not np.isclose(uncertainty, 100 / alpha.sum(), atol=1e-5):
            raise ValueError("Uncertainty contradicts Dirichlet concentration")


def decision(number, score, uncertainty, min_score=0.8, max_uncertainty=0.2):
    if not 0 <= min_score <= 1 or not 0 <= max_uncertainty <= 1:
        raise ValueError("Abstention thresholds must be in [0,1]")
    if score < min_score or uncertainty > max_uncertainty:
        return {
            "status": "abstain",
            "jersey_number": None,
            "reason": "insufficient_model_evidence_not_a_legibility_label",
        }
    return {
        "status": "provisional",
        "jersey_number": int(number),
        "reason": "uncalibrated_model_evidence_no_named_identity",
    }


def infer(crop_manifest, upstream, checkpoint, out, batch_size=8, device="cuda:0", limit=None):
    import torch

    upstream, checkpoint, out = Path(upstream).resolve(), Path(checkpoint), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("batch_size and limit must be positive")
    if file_sha(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint differs from the fixed downloaded SoccerNet ViT-B weights")
    upstream_hashes = verify_upstream(upstream)
    sys.path[:0] = [str(upstream), str(upstream / "src")]
    from config import Config
    from uncertainty_jnr.augmentation import get_val_transforms
    from uncertainty_jnr.data import SimpleImageDataset
    from uncertainty_jnr.model import TimmOCRModel
    from uncertainty_jnr.utils import load_checkpoint

    manifest = json.loads(Path(crop_manifest).read_text())
    root = Path(crop_manifest).parent
    rows = {r["crop_id"]: r for r in manifest["crops"]}
    for row in rows.values():
        path = root / row["crop_path"]
        if file_sha(path) != row["crop_sha256"]:
            raise ValueError("Crop hash mismatch")
        import cv2

        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (row["crop_height"], row["crop_width"]):
            raise ValueError("Missing or unreadable crop; zero-image fallback forbidden")
    config_path = upstream / "configs/base8_reid.yaml"
    config = Config.from_yaml(config_path)
    if config.model.uncertainty_head != "dirichlet":
        raise ValueError("This frozen worker evaluates the released Dirichlet checkpoint")
    dataset = SimpleImageDataset(
        root / "crops",
        target_size=config.data.target_size,
        transform=get_val_transforms(target_size=config.data.target_size),
    )
    if {p.stem for p in dataset.image_paths} != set(rows) or len(dataset) != len(rows):
        raise ValueError("Crop directory contains extra/missing images")
    if limit:
        dataset.image_paths = dataset.image_paths[:limit]
    model = TimmOCRModel(
        model_name=config.model.model_name,
        pretrained=False,
        classifier_type=config.model.classifier_type,
        embedding_type=config.model.embedding_type,
        per_digit_bias=config.model.per_digit_bias,
        uncertainty_head=config.model.uncertainty_head,
    )
    # Upstream compiled-key handling; strict=True forbids random uninitialized missing heads.
    load_checkpoint(model, checkpoint, torch.device("cpu"), strict=True)
    model.to(device).eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    results, full = [], {"crop_ids": [], "probs": [], "alphas": [], "all_logits": []}
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            outputs = model(batch["image"].to(device))
            probs = outputs.number_probs.float().cpu().numpy()
            alpha = (outputs.number_logits.float().exp() + 1).cpu().numpy()
            uncertainties = outputs.uncertainty.float().cpu().numpy().reshape(-1)
            logits = outputs.all_logits.float().cpu().numpy()
            for index, crop_id in enumerate(batch["image_path"]):
                number, score = int(probs[index].argmax()), float(probs[index].max())
                uncertainty = float(uncertainties[index])
                validate_number_output(number, score, uncertainty, probs[index], alpha[index])
                result = {
                    "crop_id": crop_id,
                    "pred_number": number,
                    "pred_score": score,
                    "uncertainty": uncertainty,
                    **decision(number, score, uncertainty),
                }
                results.append(result)
                for name, value in (
                    ("crop_ids", crop_id),
                    ("probs", probs[index]),
                    ("alphas", alpha[index]),
                    ("all_logits", logits[index]),
                ):
                    full[name].append(value)
            print(f"JNR: {len(results)}/{len(dataset)} crops", flush=True)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    out.mkdir(parents=True)
    np.savez_compressed(out / "distributions.npz", **{k: np.asarray(v) for k, v in full.items()})
    output = {
        "schema": "jersey-predictions/v1",
        "backend": "uncertainty-jnr-vitb-soccernet",
        "crop_manifest_sha256": file_sha(crop_manifest),
        "checkpoint_sha256": file_sha(checkpoint),
        "complete_crop_manifest": len(results) == len(rows),
        "ground_truth_used_in_inference": False,
        "upstream_commit": UPSTREAM_REVISION,
        "upstream_revision_verification": "strict expected source/config hashes; Git metadata not required",
        "upstream_file_hashes": upstream_hashes,
        "adapter_sha256": file_sha(__file__),
        "distributions_sha256": file_sha(out / "distributions.npz"),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "torchvision", "timm", "albumentations", "numpy")
        },
        "precision": "float32_no_autocast",
        "strict_checkpoint_loading": True,
        "device": str(device),
        "inference_and_output_s": time.perf_counter() - started,
        "abstention_policy": {
            "min_number_probability": 0.8,
            "max_dirichlet_uncertainty": 0.2,
            "calibrated": False,
        },
        "predictions": results,
    }
    save_json(out / "predictions.json", output)
    return output


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def jersey_label(value):
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text.isdigit() or not 0 <= int(text) <= 99:
        return None
    return int(text)


def evaluate(crop_manifest, predictions_path, truth_path, out, prefixes=()):
    crops = json.loads(Path(crop_manifest).read_text())
    predictions = json.loads(Path(predictions_path).read_text())
    truth = json.loads(Path(truth_path).read_text())
    if predictions["crop_manifest_sha256"] != file_sha(crop_manifest):
        raise ValueError("Predictions not bound to these detector crops")
    if truth["manifest_sha256"] != crops["manifest_sha256"]:
        raise ValueError("GT does not share the frozen image manifest")
    lookup, source_hashes = {}, {}
    for source in truth["sources"]:
        path = resolve_path(source["path"], prefixes)
        if file_sha(path) != source["sha256"]:
            raise ValueError("Ground-truth source hash mismatch")
        source_hashes[source["sequence"]] = file_sha(path)
        data = json.loads(path.read_text())
        frame_ids = {
            image["image_id"]: int(Path(image["file_name"]).stem) for image in data["images"]
        }
        for annotation in data["annotations"]:
            if annotation.get("supercategory") != "object":
                continue
            attrs = annotation.get("attributes", {})
            if attrs.get("role") not in ("player", "goalkeeper"):
                continue
            box = annotation["bbox_image"]
            key = source["sequence"], frame_ids[annotation["image_id"]]
            lookup.setdefault(key, []).append(
                {
                    "box": [box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]],
                    "number": jersey_label(attrs.get("jersey")),
                    "track_id": annotation["track_id"],
                    "annotation_id": annotation["id"],
                }
            )
    crop_map = {r["crop_id"]: r for r in crops["crops"]}
    used_truth, records = set(), []
    # Matching order is detector confidence and ID; jersey predictions never influence matches.
    pred_map = {r["crop_id"]: r for r in predictions["predictions"]}
    if len(pred_map) != len(predictions["predictions"]):
        raise ValueError("Duplicate jersey crop predictions")
    if set(pred_map) - set(crop_map):
        raise ValueError("Predictions include crops outside the frozen crop manifest")
    if len(crop_map) != len(crops["crops"]):
        raise ValueError("Duplicate crop IDs in frozen crop manifest")
    # Match ALL frozen crops independently of submitted outputs. A partial smoke run
    # must not shrink the denominator or alter which detection owns a GT match.
    for crop_id in sorted(crop_map, key=lambda k: (-crop_map[k]["detector_score"], k)):
        row = crop_map[crop_id]
        prediction = pred_map.get(
            crop_id,
            {
                "crop_id": crop_id,
                "pred_number": None,
                "jersey_number": None,
                "status": "abstain",
                "reason": "missing_prediction",
            },
        )
        key = row["sequence"], row["frame_id"]
        available = [g for g in lookup.get(key, []) if (key, g["annotation_id"]) not in used_truth]
        match = max(
            available, key=lambda g: iou(row["predicted_bbox_xyxy"], g["box"]), default=None
        )
        overlap = iou(row["predicted_bbox_xyxy"], match["box"]) if match else 0.0
        if overlap < 0.5:
            match = None
        if match:
            used_truth.add((key, match["annotation_id"]))
        gt_number = match["number"] if match else None
        records.append(
            {
                **prediction,
                "prediction_available": crop_id in pred_map,
                "sequence": row["sequence"],
                "frame_id": row["frame_id"],
                "matched_gt": match is not None,
                "match_iou": overlap,
                "gt_number": gt_number,
                "gt_track_id": match["track_id"] if match else None,
                "correct_number": prediction["jersey_number"] == gt_number
                if gt_number is not None and prediction["jersey_number"] is not None
                else None,
            }
        )
    labeled = [r for r in records if r["gt_number"] is not None]
    accepted_labeled = [r for r in labeled if r["jersey_number"] is not None]
    accepted = [r for r in records if r["jersey_number"] is not None]
    summary = {
        "expected_crops": len(crop_map),
        "submitted_prediction_crops": len(pred_map),
        "missing_prediction_crops": len(crop_map) - len(pred_map),
        "evaluated_crops": len(records),
        "matched_gt_crops": sum(r["matched_gt"] for r in records),
        "matched_number_labeled_crops": len(labeled),
        "accepted_crops": len(accepted),
        "abstained_crops": len(records) - len(accepted),
        "acceptance_coverage": len(accepted) / len(records) if records else None,
        "accepted_labeled_crops": len(accepted_labeled),
        "accepted_number_precision": sum(r["correct_number"] for r in accepted_labeled)
        / len(accepted_labeled)
        if accepted_labeled
        else None,
        "correct_accepted_number_coverage_of_labeled_crops": sum(
            r["correct_number"] for r in accepted_labeled
        )
        / len(labeled)
        if labeled
        else None,
        "forced_top1_accuracy_labeled_crops": sum(
            r.get("pred_number") == r["gt_number"] for r in labeled
        )
        / len(labeled)
        if labeled
        else None,
        "accepted_unmatched_or_number_unlabeled": sum(r["gt_number"] is None for r in accepted),
        "labeled_crops_by_sequence": dict(Counter(r["sequence"] for r in labeled)),
    }
    report = {
        "schema": "jersey-evaluation/v1",
        "summary": summary,
        "crop_manifest_sha256": file_sha(crop_manifest),
        "predictions_sha256": file_sha(predictions_path),
        "truth_sha256": file_sha(truth_path),
        "source_annotation_sha256": source_hashes,
        "evaluation_only_gt": True,
        "missing_output_policy": "Missing predictions are abstentions; matching and denominators use all frozen crops",
        "matching": "detector-score-ordered one-to-one same-frame IoU>=0.5",
        "limitations": [
            "GT jersey attributes identify a player's number, not its visibility or legibility in this crop",
            "Abstention coverage measured; unreadable-class accuracy unavailable without crop-legibility labels",
            "200 sampled predicted detections do not measure full-pipeline jersey recall",
            "Repeated people and nearby frames are correlated; no independent-sample confidence interval",
            "No predicted track IDs, temporal voting, roster matching or inferred player names",
            "Confidence thresholds fixed before evaluation and uncalibrated; development data only",
        ],
        "records": sorted(records, key=lambda r: r["crop_id"]),
    }
    save_json(out, report)
    return report


def tesseract(crop_manifest, out):
    """Same original predicted boxes and upstream torso crop; local OCR comparator."""
    import csv
    import io
    import shutil
    import tempfile

    import cv2

    executable = shutil.which("tesseract")
    if not executable:
        raise RuntimeError("Tesseract executable unavailable")
    data = json.loads(Path(crop_manifest).read_text())
    results = []
    with tempfile.TemporaryDirectory(prefix="soccerviz-jersey-") as tmp:
        image_path = Path(tmp) / "torso.png"
        for row in data["crops"]:
            path = Path(crop_manifest).parent / row["crop_path"]
            if file_sha(path) != row["crop_sha256"]:
                raise ValueError("Crop hash mismatch")
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("Unreadable source crop")
            x0, y0, x1, y1 = row["model_torso_bounds_in_crop_xyxy"]
            torso = image[y0:y1, x0:x1]
            gray = cv2.cvtColor(cv2.resize(torso, None, fx=4, fy=4), cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(image_path), gray)
            output = subprocess.run(
                [
                    executable,
                    str(image_path),
                    "stdout",
                    "--psm",
                    "7",
                    "-c",
                    "tessedit_char_whitelist=0123456789",
                    "tsv",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            tokens = [
                (jersey_label(r.get("text")), float(r["conf"]))
                for r in csv.DictReader(io.StringIO(output.stdout), delimiter="\t")
                if jersey_label(r.get("text")) is not None and float(r["conf"]) >= 0
            ]
            number, confidence = max(tokens, key=lambda r: r[1], default=(None, None))
            accepted = number is not None and confidence >= 80
            results.append(
                {
                    "crop_id": row["crop_id"],
                    "pred_number": number,
                    "ocr_confidence": confidence,
                    "jersey_number": number if accepted else None,
                    "status": "provisional" if accepted else "abstain",
                }
            )
    report = {
        "schema": "jersey-predictions/v1",
        "backend": "tesseract_psm7_digits",
        "crop_manifest_sha256": file_sha(crop_manifest),
        "ground_truth_used_in_inference": False,
        "transform": "same upstream torso region;4x linear resize;grayscale;PSM7 digits",
        "acceptance_threshold": 80,
        "complete_crop_manifest": True,
        "predictions": results,
    }
    save_json(out, report)
    return report
