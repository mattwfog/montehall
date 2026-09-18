"""Ball-flight fitting: WASB peaks -> rim-relative parabola -> launch point.

Canonical home of the machinery scripts/flight_fit_probe.py and
scripts/wasb_shot_window_probe.py pioneered (08-03 stage-1/2 solves); the
probe scripts import from here. Track-before-detect: fit the family of
smooth trajectories that END AT THE RIM over WASB heatmap peaks expressed
in rim-relative, rim-width-normalized court feet (camera motion cancels),
score by integrated energy, gate on inliers/span/endpoint. A passing fit
yields the LAUNCH POINT — the shooter's location.

⚠ WASB weights are NEVER auto-discovered (the 08-24 default-path trap):
callers pass the checkpoint explicitly.
⚠ Decode is PyAV, never cv2.VideoCapture (AV1 silently read-fails —
mine_broadcast.py:274 trap).
"""

from __future__ import annotations

import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

WINDOW_S = (-3.0, 1.0)
PEAK_MIN = 0.2
CLS_PERSON = 0  # DetClass space (⚠ was wrongly 1=BALL until 2026-08-03)
CLS_RIM = 3
RIM_CONF_MIN = 0.3
RIM_TRUE_FT = 1.5
RIM_NEAR_ARRIVE_FT = 3.0
FLIGHT_SPAN_MIN_S = 0.4
FLIGHT_LOOKBACK_S = 1.6
INLIER_FT = 1.5
MIN_INLIERS = 6
LAUNCH_PERSON_FT = 6.0
BATCH = 8
WASB_W, WASB_H = 512, 288


def load_wasb(wasb_root: Path, weights: Path):
    import torch
    import yaml
    from omegaconf import OmegaConf  # hrnet.py reads cfg.MODEL.EXTRA

    sys.path.insert(0, str(wasb_root / "src"))
    from models import build_model  # wasb src

    model_cfg = yaml.safe_load(
        (wasb_root / "src/configs/model/wasb.yaml").read_text())
    model = build_model(OmegaConf.create({"model": model_cfg}))
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(dev).eval(), dev, model_cfg


def heatmap_of(preds):
    """Normalize the model output to a [3, H, W] numpy heatmap stack.
    Sigmoid ONLY when logits leave [0,1] — an unconditional sigmoid
    double-squashes already-normalized outputs and shifts PEAK_MIN."""
    import torch

    if isinstance(preds, dict):
        preds = preds[sorted(preds.keys())[0]]
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    hm = preds[0] if preds.dim() == 4 else preds
    if float(hm.max()) > 1.0 or float(hm.min()) < 0.0:
        hm = torch.sigmoid(hm)
    return hm.detach().cpu().numpy()


def fit_flight(t: np.ndarray, x: np.ndarray, y: np.ndarray,
               scores: np.ndarray, t_arr: float,
               min_inliers: int = MIN_INLIERS,
               span_min: float = FLIGHT_SPAN_MIN_S) -> dict | None:
    """Fit x,y ~ quadratic(t) over peaks preceding an arrival time.

    Returns fit dict when the inlier set passes count/span/endpoint
    gates. Coordinates are rim-relative feet (rim at origin).
    """
    sel = (t <= t_arr + 0.05) & (t >= t_arr - FLIGHT_LOOKBACK_S)
    if sel.sum() < min_inliers:
        return None
    ts, xs, ys, ws = t[sel], x[sel], y[sel], scores[sel]
    inliers = np.ones(ts.shape[0], dtype=bool)
    coef_x = coef_y = None
    for _ in range(3):  # LSQ -> residual gate -> refit
        if inliers.sum() < min_inliers:
            return None
        basis = np.stack([np.ones_like(ts), ts - t_arr,
                          (ts - t_arr) ** 2], axis=1)
        w = ws * inliers
        coef_x, *_ = np.linalg.lstsq(basis * w[:, None], xs * w, rcond=None)
        coef_y, *_ = np.linalg.lstsq(basis * w[:, None], ys * w, rcond=None)
        res = np.hypot(basis @ coef_x - xs, basis @ coef_y - ys)
        inliers = res <= INLIER_FT
    n_in = int(inliers.sum())
    if n_in < min_inliers:
        return None
    span = float(ts[inliers].max() - ts[inliers].min())
    end_dist = float(np.hypot(coef_x[0], coef_y[0]))  # value at t_arr
    if span < span_min or end_dist > 2.0:
        return None
    t_launch = float(ts[inliers].min())
    dt = t_launch - t_arr
    launch = (float(coef_x[0] + coef_x[1] * dt + coef_x[2] * dt * dt),
              float(coef_y[0] + coef_y[1] * dt + coef_y[2] * dt * dt))
    return {"t_arr": round(t_arr, 2), "t_launch": round(t_launch, 2),
            "n_inliers": n_in, "span_s": round(span, 2),
            "energy": round(float(ws[inliers].sum()), 2),
            "launch_ft": [round(v, 1) for v in launch]}


def best_flight(peaks: list[tuple], rim_dist: np.ndarray,
                min_inliers: int = MIN_INLIERS,
                span_min: float = FLIGHT_SPAN_MIN_S) -> dict | None:
    """Try each near-rim arrival candidate; keep the highest-energy fit."""
    t = np.array([p[0] for p in peaks])
    x = np.array([p[1] for p in peaks])
    y = np.array([p[2] for p in peaks])
    s = np.array([p[3] for p in peaks])
    arrivals = t[rim_dist <= RIM_NEAR_ARRIVE_FT]
    best = None
    for t_arr in arrivals[:8]:
        fit = fit_flight(t, x, y, s, float(t_arr),
                         min_inliers=min_inliers, span_min=span_min)
        if fit and (best is None or fit["energy"] > best["energy"]):
            best = fit
    return best


class DetIndex:
    """Detections sorted by ts with class-sliced window reads."""

    def __init__(self, table) -> None:
        dts = table.column("ts_ms").to_numpy(zero_copy_only=False)
        order = np.argsort(dts, kind="stable")
        self.ts = dts[order]
        self.cls = table.column("cls").to_numpy(zero_copy_only=False)[order]
        self.conf = table.column("conf").to_numpy(zero_copy_only=False)[order]
        self.x1 = table.column("x1").to_numpy(zero_copy_only=False)[order]
        self.y1 = table.column("y1").to_numpy(zero_copy_only=False)[order]
        self.x2 = table.column("x2").to_numpy(zero_copy_only=False)[order]
        self.y2 = table.column("y2").to_numpy(zero_copy_only=False)[order]

    def window(self, lo_ms: float, hi_ms: float, cls_want: int):
        """(ts, cx, cy, w, bottom) for one class inside [lo, hi)."""
        i = bisect_left(self.ts, lo_ms)
        j = bisect_left(self.ts, hi_ms)
        sel = self.cls[i:j] == cls_want
        if cls_want == CLS_RIM:
            sel &= self.conf[i:j] >= RIM_CONF_MIN
        return (self.ts[i:j][sel],
                (self.x1[i:j][sel] + self.x2[i:j][sel]) / 2,
                (self.y1[i:j][sel] + self.y2[i:j][sel]) / 2,
                self.x2[i:j][sel] - self.x1[i:j][sel],
                self.y2[i:j][sel])


def flight_fits_for_anchors(
    video: Path,
    anchors_ms: list[int],
    det: DetIndex,
    wasb_root: Path,
    weights: Path,
    window_s: tuple[float, float] = WINDOW_S,
    min_inliers: int = MIN_INLIERS,
    span_min: float = FLIGHT_SPAN_MIN_S,
    log=None,
) -> list[dict]:
    """One row per anchor: {"anchor_ms", "frames", "peaks", "fit"?,
    "launch_person_ft"?}. Dense WASB over the window, peaks ->
    rim-relative feet -> best_flight, plus the fake-rim null control.
    """
    import av
    import cv2
    import torch

    model, dev, model_cfg = load_wasb(wasb_root, weights)
    wh = (model_cfg["inp_width"], model_cfg["inp_height"])

    container = av.open(str(video))
    stream = container.streams.video[0]
    tb = stream.time_base
    fps = float(stream.average_rate or 30.0)
    stride = 2 if fps > 45 else 1
    native_scale = stream.width / WASB_W

    rows: list[dict] = []
    n_fit = n_null = 0
    for n_done, a_ms in enumerate(anchors_ms):
        t0 = a_ms / 1000.0
        lo_s, hi_s = t0 + window_s[0], t0 + window_s[1]
        container.seek(int(max(lo_s, 0.0) / tb), stream=stream)
        frames, times = [], []
        seen = 0
        for frame in container.decode(stream):
            stamp = frame.pts if frame.pts is not None else frame.dts
            if stamp is None:
                continue
            ts_s = float(stamp * tb)
            if ts_s < lo_s - 0.02:
                continue
            if ts_s > hi_s:
                break
            if seen % stride == 0:
                img = cv2.resize(frame.to_ndarray(format="rgb24"), wh)
                img = (img.astype(np.float32) / 255 - IMAGENET_MEAN) \
                    / IMAGENET_STD
                frames.append(img.transpose(2, 0, 1))
                times.append(ts_s)
            seen += 1
        row: dict = {"anchor_ms": int(a_ms), "frames": len(frames)}
        if len(frames) >= min_inliers + 2:
            triplets = np.stack([
                np.concatenate(frames[i:i + 3], axis=0)
                for i in range(len(frames) - 2)])
            mid_t = times[1:-1]
            peaks = []
            with torch.no_grad():
                for i in range(0, triplets.shape[0], BATCH):
                    batch = torch.from_numpy(triplets[i:i + BATCH]).to(dev)
                    preds = model(batch)
                    for k in range(batch.shape[0]):
                        hm = heatmap_of(
                            preds[k:k + 1] if not isinstance(preds, dict)
                            else {s: v[k:k + 1] for s, v in preds.items()})
                        mid = hm[min(1, hm.shape[0] - 1)]
                        score = float(mid.max())
                        if score < PEAK_MIN:
                            continue
                        yx = np.unravel_index(int(mid.argmax()), mid.shape)
                        peaks.append((mid_t[i + k],
                                      float(yx[1]) * native_scale,
                                      float(yx[0]) * native_scale, score))
            r_ts, r_cx, r_cy, r_w, _ = det.window(
                lo_s * 1000, hi_s * 1000, CLS_RIM)
            rel = []
            if r_ts.size >= 3 and peaks:
                for (pt, px, py, sc) in peaks:
                    k = np.argmin(np.abs(r_ts - pt * 1000))
                    if abs(r_ts[k] - pt * 1000) > 500:
                        continue
                    fpp = RIM_TRUE_FT / max(float(r_w[k]), 1e-6)
                    rel.append((pt, (px - float(r_cx[k])) * fpp,
                                (py - float(r_cy[k])) * fpp, sc))
            row["peaks"] = len(rel)
            if len(rel) >= min_inliers:
                rim_dist = np.array([np.hypot(p[1], p[2]) for p in rel])
                fit = best_flight(rel, rim_dist,
                                  min_inliers=min_inliers, span_min=span_min)
                fake = np.array([np.hypot(p[1] - 20.0, p[2]) for p in rel])
                null_fit = best_flight(
                    [(p[0], p[1] - 20.0, p[2], p[3]) for p in rel], fake,
                    min_inliers=min_inliers, span_min=span_min)
                if fit:
                    n_fit += 1
                    row["fit"] = fit
                if null_fit:
                    n_null += 1
                    row["null_fit"] = True
        rows.append(row)
        if log and (n_done + 1) % 25 == 0:
            log(f"flight: {n_done + 1}/{len(anchors_ms)} fits={n_fit} "
                f"null={n_null}")
    container.close()
    return rows
