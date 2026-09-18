"""Expose completed candidate experiments in the existing workbench catalog."""

import argparse
import json
from pathlib import Path

from soccerviz.datasets.dataset_catalog import DatasetCatalog


def register(root):
    catalog = DatasetCatalog(root / "artifacts/harness")
    benchmark = "artifacts/vision-benchmark"
    entries = [
        (
            "Soccer YOLOv8x control",
            "vision-detector",
            f"{benchmark}/yolov8x-soccer-ap-results.json",
            [f"{benchmark}/yolov8x-soccer-ap-predictions.json"],
        ),
        (
            "RF-DETR Medium COCO, 576 pixels",
            "vision-detector",
            f"{benchmark}/rf-medium-coco-results.json",
            [f"{benchmark}/rf-medium-coco-predictions.json"],
        ),
        (
            "RF-DETR Medium COCO, 960 pixels",
            "vision-detector",
            f"{benchmark}/rf-medium-960-results.json",
            [f"{benchmark}/rf-medium-960-predictions.json"],
        ),
        (
            "WASB soccer temporal ball model",
            "vision-ball",
            f"{benchmark}/wasb-peak-results.json",
            [f"{benchmark}/wasb-peak-predictions.json"],
        ),
    ]
    for name, prefix, model_folder, checkpoint in (
        ("YOLO26 Medium COCO, 640 pixels", "yolo26m-coco", None, None),
        (
            "RF-DETR Medium soccer pilot, five epochs",
            "rf-medium-soccer-pilot",
            "artifacts/models/rf-detr-soccer-pilot",
            "checkpoint_best_total.pth",
        ),
        (
            "YOLO26 Medium soccer pilot, five epochs",
            "yolo26m-soccer-pilot",
            "artifacts/models/yolo26m-soccer-pilot",
            "best.pt",
        ),
    ):
        files = [f"{benchmark}/{prefix}-predictions.json"]
        if model_folder:
            files += [
                f"{model_folder}/{checkpoint}",
                f"{model_folder}/training-evidence.json",
                f"{model_folder}/run-config.json",
                "artifacts/detector-training/manifest.json",
                "artifacts/detector-training/protocol.json",
            ]
        entries.append((name, "vision-detector", f"{benchmark}/{prefix}-results.json", files))
    entries.append(
        (
            "RF-DETR Medium Roboflow CC BY v10 pilot, five epochs",
            "vision-detector",
            f"{benchmark}/rf-medium-roboflow-v10-results.json",
            [
                f"{benchmark}/rf-medium-roboflow-v10-predictions.json",
                "artifacts/models/rf-detr-roboflow-v10/checkpoint_best_total.pth",
                "artifacts/models/rf-detr-roboflow-v10/training-evidence.json",
                "artifacts/models/rf-detr-roboflow-v10/run-config.json",
                "artifacts/detector-training-roboflow-v10/manifest.json",
                "artifacts/detector-training-roboflow-v10/protocol.json",
            ],
        )
    )
    for name, prefix in (("YOLOv8x", "yolov8x"), ("RF-DETR Medium", "rf-medium")):
        entries.append(
            (
                f"{name} with fixed anonymous tracker",
                "vision-tracking",
                f"{benchmark}/{prefix}-tracking-verified-results.json",
                [f"{benchmark}/{prefix}-tracking-verified-results-tracks.json"],
            )
        )
    for name, prefix in (
        ("RF-DETR Medium, 960 pixels", "rf-medium-960"),
        ("YOLO26 Medium COCO", "yolo26m-coco"),
        ("RF-DETR soccer pilot", "rf-medium-soccer-pilot"),
        ("YOLO26 soccer pilot", "yolo26m-soccer-pilot"),
        ("RF-DETR Roboflow CC BY v10 pilot", "rf-medium-roboflow-v10"),
    ):
        entries.append(
            (
                f"{name} with fixed anonymous tracker",
                "vision-tracking",
                f"{benchmark}/{prefix}-tracking-results.json",
                [f"{benchmark}/{prefix}-tracking-results-tracks.json"],
            )
        )
    for name, prefix in (("YOLO pitch control", "yolo"), ("PnLCalib candidate", "pnl")):
        entries.append(
            (
                name,
                "vision-calibration",
                f"{benchmark}/calibration/{prefix}-results.json",
                [
                    f"{benchmark}/calibration/{prefix}-predictions.json",
                    f"{benchmark}/calibration/protocol.json",
                ],
            )
        )
    entries.append(
        (
            "Uncertainty-JNR public soccer checkpoint",
            "vision-jersey",
            "artifacts/integrations/jersey-model-full/evaluation.json",
            [
                "artifacts/integrations/jersey-model-full/predictions.json",
                "artifacts/integrations/jersey-model-full/distributions.npz",
                "artifacts/integrations/jersey-model-crops/crop-manifest.json",
            ],
        )
    )
    for game in (1, 2):
        folder = f"artifacts/integrations/formation-metrica-game-{game}"
        entries.append(
            (
                f"UnravelSports EFPI, Metrica game {game}",
                "provider-formation",
                f"{folder}/report.json",
                [f"{folder}/frame-assignments.parquet", f"{folder}/frame-eligibility.parquet"],
            )
        )
    result = []
    for name, kind, report, files in entries:
        paths = [root / p for p in files]
        if kind.startswith("vision-"):
            paths += [root / benchmark / "manifest.json", root / benchmark / "protocol.json"]
        record = catalog.register(
            name, kind, "Development experiment; no model promotion", root / report, paths
        )
        verification = catalog.verify(record["id"])
        if not verification["valid"]:
            raise ValueError(f"Catalog verification failed: {name}")
        result.append({"name": name, "id": record["id"], "verified": True})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(register(args.root.resolve()), indent=2))
