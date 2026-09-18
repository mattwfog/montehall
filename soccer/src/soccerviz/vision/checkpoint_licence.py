"""Licence provenance of every checkpoint the video workflow may load, keyed by SHA256.

The copyleft rule (2026-09-07; docs/guides/public-data.md, "Licence census")
makes anything trained on GPL/AGPL/non-commercial data or run through copyleft
code an evaluation-only artefact. A checkpoint can only run in the workflow once
its SHA256 is registered here with its source and class, so a replacement copied
into the image under an old filename cannot inherit the old file's class.
"""

from __future__ import annotations

SHIPPING = "shipping"
EVALUATION_ONLY = "evaluation-only"
NO_CHECKPOINTS = "no-checkpoints"

SOCCERNET_TERMS = "GPL-3.0 plus SoccerNet non-commercial terms (training data)"
ULTRALYTICS_TERMS = "AGPL-3.0 runtime (Ultralytics)"

CHECKPOINTS = {
    "4e5cb479a57e48fc13bc69361bf7321d5f09d6097dcf97c95433b3fe1b176f18": {
        "file": "rf-soccer.pth (workflow default until 2026-09-08; retired)",
        "source": "RF-DETR Medium fine-tuned on 150 SoccerNet SN-GSR-2024 frames, "
        "corpus manifest 828750c5…",
        "licence": SOCCERNET_TERMS,
        "class": EVALUATION_ONLY,
    },
    "e5a740c733863ef1ca98372314d16322d0ca9110e1c85eed6de808e8f432fd9e": {
        "file": "rf-soccer.pth (workflow default since 2026-09-08; "
        "artifacts/models/rf-roboflow-v10-e50/checkpoint_best_total.pth)",
        "source": "RF-DETR Medium fine-tuned 50 epochs on the Roboflow Universe "
        "football-players-detection v10 corpus, manifest 568e29f9…",
        "licence": "Apache-2.0 code; CC BY 4.0 training data (attribution)",
        "class": SHIPPING,
    },
    "749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8": {
        "file": "rf-detr-medium.pth",
        "source": "Roboflow RF-DETR Medium COCO release",
        "licence": "Apache-2.0",
        "class": SHIPPING,
    },
    "02b42ab6daa21d5e8356b3da9707714f60be558dec8fbb5e865523f31daa2a0b": {
        "file": "yolo26-soccer.pt",
        "source": "YOLO26m fine-tuned on 150 SoccerNet SN-GSR-2024 frames",
        "licence": f"{ULTRALYTICS_TERMS}; {SOCCERNET_TERMS}",
        "class": EVALUATION_ONLY,
    },
    "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7": {
        "file": "yolo26m.pt",
        "source": "Ultralytics assets release v8.4.0",
        "licence": ULTRALYTICS_TERMS,
        "class": EVALUATION_ONLY,
    },
    "75b09c377fbf9d0791d23f6cfb689f5aed6eaa43a6818bd1fb884cf7507fffaf": {
        "file": "football-player-detection.pt",
        "source": "roboflow/sports demo weights (YOLOv8), no licence stated for the weights",
        "licence": f"{ULTRALYTICS_TERMS}; weights unstated",
        "class": EVALUATION_ONLY,
    },
    "678fbad05134f19c5094cb8d273812ec9c6691228180d46832551ecf99ed2912": {
        "file": "football-ball-detection.pt",
        "source": "roboflow/sports demo weights (YOLOv8), no licence stated for the weights",
        "licence": f"{ULTRALYTICS_TERMS}; weights unstated",
        "class": EVALUATION_ONLY,
    },
    "28f68f7c4056d6d9b137efd2e7ab5f3c494039380c63831649126ced25628b36": {
        "file": "football-pitch-detection.pt",
        "source": "roboflow/sports demo weights (YOLOv8), no licence stated for the weights",
        "licence": f"{ULTRALYTICS_TERMS}; weights unstated",
        "class": EVALUATION_ONLY,
    },
    "7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113": {
        "file": "pnl-keypoints.pt",
        "source": "PnLCalib release, pre-trained on SoccerNet-Calibration",
        "licence": f"GPL-2.0 code (PnLCalib); {SOCCERNET_TERMS}",
        "class": EVALUATION_ONLY,
    },
    "d72f4ed71734a2e3df9fa084f666e9b8adaef21bf69bac8952d6d3f970ff7455": {
        "file": "pnl-lines.pt",
        "source": "PnLCalib release, pre-trained on SoccerNet-Calibration",
        "licence": f"GPL-2.0 code (PnLCalib); {SOCCERNET_TERMS}",
        "class": EVALUATION_ONLY,
    },
    "528e1b26048a03ce51678f464f1092034c7126064d9ec16042b4ea6bf8a4f869": {
        "file": "pitch-heatmap.pt candidate (artifacts/models/pitch-hrnet-w32-v18-vis/best.pt; "
        "not the workflow default)",
        "source": "timm HRNet-W32 heatmap calibrator, ImageNet init, 60 epochs on the "
        "Roboflow Universe football-field-detection v18 export (255 train images)",
        "licence": "Apache-2.0 code (timm); CC BY 4.0 training data (attribution)",
        "class": SHIPPING,
    },
    "5f3e7b12715b948cf170abaf5c6c752c541f31b26761d2a817f7e62103fa11c7": {
        "file": "pitch-heatmap.pt candidate (artifacts/models/pitch-hrnet-w32-v18-lines/best.pt; "
        "not the workflow default)",
        "source": "timm HRNet-W32 heatmap calibrator with 20 line/conic channels rendered from "
        "each training image's own fitted homography, ImageNet init, 60 epochs on the "
        "Roboflow Universe football-field-detection v18 export (255 train images)",
        "licence": "Apache-2.0 code (timm); CC BY 4.0 training data (attribution)",
        "class": SHIPPING,
    },
    "12585310aedc5669e4d8bab20adc3bc56425e3f0695226673968f33877bbb3bf": {
        "file": "jersey-classifier.pt candidate (artifacts/models/jersey-resnet34-v1/best.pt; "
        "not in the workflow)",
        "source": "timm ResNet-34 whole-crop jersey classifier, ImageNet init, 12 epochs on the "
        "Roboflow taiseis-workspace jersey-number v1 tiles and Pusan digit-box numbers",
        "licence": "Apache-2.0 code (timm); CC BY 4.0 training data (attribution)",
        "class": SHIPPING,
    },
    "e83a76b32e875a4f817c4cc505a6dd50cfe14d9856fd83f58822446d413f8a9e": {
        "file": "jersey-classifier.pt candidate (artifacts/models/jersey-resnet34-synth/best.pt; "
        "not in the workflow)",
        "source": "timm ResNet-34 whole-crop jersey classifier, ImageNet init, 12 epochs on numbers "
        "rendered in OFL fonts over Roboflow football-players-detection v10 torsos plus the "
        "taiseis-workspace and Pusan numbers",
        "licence": "Apache-2.0 code (timm); CC BY 4.0 training data (attribution); OFL 1.1 fonts",
        "class": SHIPPING,
    },
    "43eb17804e94e012883b37618d70fbab3fb190f58eb851843e38589a9cf80923": {
        "file": "jersey-vitb.pt",
        "source": "Uncertainty-JNR ViT-B release, trained on SoccerNet JNR and ReID",
        "licence": f"CC BY-NC-SA 4.0 code (Uncertainty-JNR); {SOCCERNET_TERMS}",
        "class": EVALUATION_ONLY,
    },
}


def classify_checkpoint(digest):
    """Return the registered provenance record for one checkpoint SHA256."""
    record = CHECKPOINTS.get(digest)
    if record is None:
        raise ValueError(
            f"Checkpoint {digest[:12]}… is not registered in "
            "soccerviz.vision.checkpoint_licence.CHECKPOINTS; record its source and "
            "licence class before it can run in the workflow"
        )
    return dict(record)


def licence_report(model_sha256):
    """Classify a role→SHA256 map. Shipping only when every checkpoint ships."""
    checkpoints = {role: classify_checkpoint(digest) for role, digest in model_sha256.items()}
    classes = {record["class"] for record in checkpoints.values()}
    overall = (
        NO_CHECKPOINTS if not classes else SHIPPING if classes == {SHIPPING} else EVALUATION_ONLY
    )
    return {
        "class": overall,
        "blocking": sorted(
            role for role, record in checkpoints.items() if record["class"] != SHIPPING
        ),
        "checkpoints": checkpoints,
    }


def require_shipping(report):
    """Raise unless every loaded checkpoint is cleared for shipping."""
    if report["class"] != SHIPPING:
        blocked = (
            ", ".join(
                f"{role} ({report['checkpoints'][role]['file']}: {report['checkpoints'][role]['licence']})"
                for role in report["blocking"]
            )
            or "no checkpoints loaded"
        )
        raise ValueError(f"Workflow models are {report['class']}, not shipping: {blocked}")
