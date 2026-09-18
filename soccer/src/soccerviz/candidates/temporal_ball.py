"""Pinned WASB temporal heatmap inference with explicit source-frame context.

The upstream model, affine transform, image normalization, component centroid
postprocessor and tracker are used directly. GPU dependencies load lazily.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from soccerviz.core.assets import sha256

UPSTREAM_COMMIT = "923462cacdeb3353b84ddebdedb3f4b7a8553b0f"
UPSTREAM_SOURCE_SHA256 = "5c8e692d79d39244c60d17a4e7372227411d64514f7532690efe7330ca7b464f"
SOCCER_CHECKPOINT_SHA256 = "d0369572807c2baf751880d6cdf3cce9fc6283fa8d153f18af6baf4e64d2646c"


def source_fingerprint(upstream):
    """Hash real source bytes; validated macOS resource sidecars are not Python.

    AppleDouble is identified by both its sidecar filename and binary magic,
    and only omitted when its corresponding real file exists. Ordinary extra,
    missing, or changed source files still alter the pinned aggregate hash.
    """
    root = Path(upstream) / "src"
    files = []
    for path in root.rglob("*"):
        if path.suffix not in {".py", ".yaml"}:
            continue
        if path.name.startswith("._") and path.with_name(path.name[2:]).is_file():
            with path.open("rb") as handle:
                if handle.read(4) == b"\x00\x05\x16\x07":
                    continue
        files.append(path)
    files.sort()
    payload = "".join(f"{p.relative_to(root)} {sha256(p)}\n" for p in files)
    return hashlib.sha256(payload.encode()).hexdigest()


def frame_key(frame):
    return str(frame["sequence"]), int(frame["frame_id"])


def contiguous(previous, current):
    """Both identifiers and PTS-derived cadence must agree; never compress gaps."""
    for frame in (previous, current):
        if not math.isfinite(float(frame["fps"])) or frame["fps"] <= 0:
            raise ValueError("Frames require a positive finite source fps")
        if not math.isfinite(float(frame["timestamp_s"])):
            raise ValueError("Frame timestamp must be finite")
    return (
        previous["sequence"] == current["sequence"]
        and int(current["frame_id"]) == int(previous["frame_id"]) + 1
        and math.isclose(previous["fps"], current["fps"], rel_tol=1e-6)
        and math.isclose(
            current["timestamp_s"] - previous["timestamp_s"],
            1.0 / current["fps"],
            abs_tol=1e-6,
            rel_tol=1e-5,
        )
        and (previous["width"], previous["height"]) == (current["width"], current["height"])
        and previous.get("shot_id") == current.get("shot_id")
        and not current.get("cut_before", False)
    )


def plan_windows(frames, *, cut_before=(), unavailable=()):
    """Nonoverlapping native Step3 windows, resetting at every known discontinuity.

    Every input frame gets a status. Tail frames in segments shorter than a full
    triple stay unpredicted, rather than repeating frames or reaching across cuts.
    """
    if len({frame_key(f) for f in frames}) != len(frames):
        raise ValueError("Duplicate sequence/frame identifiers")
    cuts, unavailable = set(cut_before), set(unavailable)
    windows, statuses, pending = [], {}, []
    segment = 0

    def flush():
        for index in pending:
            statuses[index] = "incomplete_context"
        pending.clear()

    for index, frame in enumerate(frames):
        key = frame_key(frame)
        if key in unavailable:
            flush()
            statuses[index] = "image_unavailable"
            segment += 1
            continue
        boundary = (
            index == 0
            or key in cuts
            or frame_key(frames[index - 1]) in unavailable
            or not contiguous(frames[index - 1], frame)
        )
        if boundary:
            flush()
            segment += 1
        pending.append(index)
        if len(pending) == 3:
            windows.append({"indices": list(pending), "segment": segment})
            for member in pending:
                statuses[member] = "ready"
            pending.clear()
    flush()
    return windows, statuses


def frame_histogram(image):
    small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def histogram_cut_distance(left, right):
    return float(cv2.compareHist(left, right, cv2.HISTCMP_BHATTACHARYYA))


def native_candidates(result, width, height, heatmap_threshold=0.5):
    """Retain native centroid/mass; attach selected component peak as bounded score.

    A component's summed heatmap mass is the upstream ranking score, and can
    exceed1. It is preserved separately; it is not presented as confidence.
    """
    heatmap = np.asarray(result["hm"])
    if not np.isfinite(heatmap).all() or np.any(heatmap < 0) or np.any(heatmap > 1):
        raise ValueError("WASB produced a nonfinite or invalid sigmoid heatmap")
    _, labels = cv2.connectedComponents((heatmap > heatmap_threshold).astype(np.uint8))
    component_ids = sorted(set(np.unique(labels)) - {0})
    if len(component_ids) != len(result["xys"]) or len(component_ids) != len(result["scores"]):
        raise ValueError("Native WASB connected-component output is misaligned")
    candidates = []
    for component, center, mass in zip(component_ids, result["xys"], result["scores"], strict=True):
        center = np.asarray(center, dtype=float)
        mass = float(mass)
        if center.shape != (2,) or not np.isfinite(center).all() or not math.isfinite(mass):
            raise ValueError("Native WASB output contains invalid center or score")
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            continue
        candidates.append(
            {
                "label": "ball",
                "center_xy": center.tolist(),
                "score": float(heatmap[labels == component].max()),
                "native_blob_score": mass,
                "score_kind": "component_peak_sigmoid_uncalibrated",
            }
        )
    return candidates


class WASBDetector:
    def __init__(self, upstream, checkpoint, *, device="cuda:0", tracker="peak"):
        upstream = Path(upstream).resolve()
        checkpoint = Path(checkpoint).resolve()
        fingerprint = source_fingerprint(upstream)
        if fingerprint != UPSTREAM_SOURCE_SHA256:
            raise ValueError(
                f"WASB source tree differs from pinned upstream revision: "
                f"expected {UPSTREAM_SOURCE_SHA256}, observed {fingerprint}, "
                f"root {upstream}"
            )
        if sha256(checkpoint) != SOCCER_CHECKPOINT_SHA256:
            raise ValueError("WASB checkpoint differs from the pinned public soccer weights")
        if tracker not in {"peak", "online"}:
            raise ValueError("tracker must be peak or online")
        source_dir = upstream / "src"
        for name in ("models", "detectors", "utils", "dataloaders", "datasets", "trackers"):
            existing = sys.modules.get(name)
            if existing and not Path(existing.__file__).resolve().is_relative_to(source_dir):
                raise RuntimeError(f"Conflicting top-level module {name}; use an isolated worker")
        sys.path.insert(0, str(source_dir))
        import torch
        from dataloaders import build_img_transforms, get_transform
        from detectors.postprocessor import TracknetV2Postprocessor
        from models.hrnet import HRNet
        from omegaconf import OmegaConf
        from trackers.intra_frame_peak import IntraFramePeakTracker
        from trackers.online import OnlineTracker

        self.torch, self.device = torch, device
        config_dir = source_dir / "configs"
        self.cfg = OmegaConf.create(
            {
                name: OmegaConf.load(config_dir / path)
                for name, path in {
                    "model": "model/wasb.yaml",
                    "detector": "detector/tracknetv2.yaml",
                    "dataloader": "dataloader/default.yaml",
                    "tracker": "tracker/online.yaml",
                }.items()
            }
        )
        self.model = HRNet(self.cfg["model"])
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        del state
        self.model.to(device).eval()
        _, self.transform = build_img_transforms(self.cfg)
        self.get_transform = get_transform
        self.postprocessor = TracknetV2Postprocessor(self.cfg)
        self.tracker = (
            IntraFramePeakTracker(self.cfg) if tracker == "peak" else OnlineTracker(self.cfg)
        )
        self.input_wh = (self.cfg["model"]["inp_width"], self.cfg["model"]["inp_height"])
        self.last_timing_ms = {}
        self.metadata = {
            "name": "WASB-soccer",
            "upstream": "https://github.com/nttcom/WASB-SBDT",
            "upstream_commit": UPSTREAM_COMMIT,
            "source_sha256": UPSTREAM_SOURCE_SHA256,
            "checkpoint_sha256": SOCCER_CHECKPOINT_SHA256,
            "checkpoint_path": str(checkpoint),
            "device": device,
            "model": OmegaConf.to_container(self.cfg["model"], resolve=True),
            "postprocessor": OmegaConf.to_container(
                self.cfg["detector"]["postprocessor"], resolve=True
            ),
            "tracker": "upstream IntraFramePeakTracker"
            if tracker == "peak"
            else "upstream OnlineTracker max_disp300 source pixels",
            "scope": "Native WASB detector with explicit tracker selection; peak is detector-only "
            "and differs from the paper's default online displacement gate.",
            "preprocessing": "Upstream RGB affine warp,512x288,ImageNet normalization,9 channels",
            "frames_in": 3,
            "frames_out": 3,
            "window_step": 3,
            "precision": "float32",
            "score_kind": "component_peak_sigmoid_uncalibrated",
            "score_note": "Native weighted blob mass ranks candidates and is preserved separately. "
            "Component peak is a bounded heatmap activation, not calibrated confidence. "
            "Native heatmap threshold0.5 is fixed before evaluation.",
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "numpy", "opencv-python", "hydra-core")
            },
        }

    def _sync(self):
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def reset_tracker(self):
        self.tracker.refresh()

    def select(self, candidates):
        native = [
            {"xy": np.array(c["center_xy"]), "score": c["native_blob_score"]} for c in candidates
        ]
        selected = self.tracker.update(native)
        if not selected["visi"]:
            return []
        for candidate in candidates:
            if np.allclose(candidate["center_xy"], [selected["x"], selected["y"]]):
                return [candidate]
        raise ValueError("Native tracker selected a center absent from native detections")

    def predict(self, images):
        from PIL import Image

        if len(images) != 3:
            raise ValueError("WASB requires exactly3 consecutive frames")
        shape = images[0].shape
        if any(im.dtype != np.uint8 or im.shape != shape for im in images):
            raise ValueError("WASB frames must be equal-shaped uint8 BGR images")
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError("WASB frames require HxWx3 BGR shape")
        started = time.perf_counter()
        affine = self.get_transform(images[0], self.input_wh)
        inverse = self.get_transform(images[0], self.input_wh, inv=1)
        tensors = []
        for image in images:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            warped = cv2.warpAffine(rgb, affine, self.input_wh, flags=cv2.INTER_LINEAR)
            tensors.append(self.transform(Image.fromarray(warped)))
        inputs = self.torch.cat(tensors, dim=0).unsqueeze(0).to(self.device)
        self._sync()
        preprocessed = time.perf_counter()
        with self.torch.inference_mode():
            predictions = self.model(inputs)
            self._sync()
            forwarded = time.perf_counter()
            processed = self.postprocessor.run(
                predictions, {0: self.torch.tensor(inverse, dtype=self.torch.float64).unsqueeze(0)}
            )
        candidates = [
            native_candidates(processed[0][index][0], shape[1], shape[0]) for index in range(3)
        ]
        finished = time.perf_counter()
        self.last_timing_ms = {
            "preprocess_and_transfer": (preprocessed - started) * 1000,
            "forward": (forwarded - preprocessed) * 1000,
            "postprocess": (finished - forwarded) * 1000,
            "total": (finished - started) * 1000,
        }
        return candidates, {
            "source_to_input_affine": affine.tolist(),
            "input_to_source_affine": inverse.tolist(),
            "input_hw": [self.input_wh[1], self.input_wh[0]],
            "output_coordinates": "original-image pixels",
        }
