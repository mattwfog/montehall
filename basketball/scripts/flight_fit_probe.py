"""Track-before-detect flight fitting — stage-2 join of the 08-03 solves.

Thin probe over the canonical machinery in pipeline/flight.py (moved there
2026-08-25, structural fix A — the shot_attribution stage consumes the same
code). Per anchor: dense WASB over the window, peaks -> rim-relative feet,
robust quadratic constrained to end at the rim, launch point + person-at-
launch check, fake-rim null control.

Usage (inside cvbench, GPU):
    python scripts/flight_fit_probe.py --video "<file>" \
        --game-dir /work/out-harvest/cal_fsu_v3real \
        --weights /work/weights/wasb/wasb_basketball_best.pth.tar \
        --out /work/models/ball-v1/flight_cal.json
    python scripts/flight_fit_probe.py --selftest   # synthetic fit check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from montehall_cv.brain.dataset import BIN_MS
from montehall_cv.brain.eval_slots import _shot_anchors
from montehall_cv.pipeline.flight import (
    CLS_PERSON,
    CLS_RIM,
    DetIndex,
    LAUNCH_PERSON_FT,
    RIM_TRUE_FT,
    fit_flight,
    flight_fits_for_anchors,
)
from montehall_cv.store.artifacts import read_stage


def selftest() -> None:
    rng = np.random.default_rng(3)
    t = np.arange(-1.2, 0.05, 1 / 30)
    x = -15.0 * t  # arrives at rim x=0 at t=0
    y = -0.5 * (-16.0) * t * t + 8.0 * t  # quadratic, ends ~0
    xn = x + rng.normal(0, 0.4, t.shape)
    yn = y + rng.normal(0, 0.4, t.shape)
    scores = np.full(t.shape, 0.8)
    fit = fit_flight(t, xn, yn, scores, 0.0)
    assert fit is not None, "selftest: fit not found"
    lx, ly = fit["launch_ft"]
    true_launch = (x[0], y[0])
    err = np.hypot(lx - true_launch[0], ly - true_launch[1])
    assert err < 2.0, f"selftest: launch err {err:.1f}ft"
    noise = rng.uniform(-30, 30, (40, 2))
    bad = fit_flight(np.linspace(-1.5, 0, 40), noise[:, 0], noise[:, 1],
                     np.full(40, 0.8), 0.0)
    assert bad is None, "selftest: fit hallucinated on uniform noise"
    print("SELFTEST_OK", json.dumps(fit))


def _launch_person_ft(row: dict, det: DetIndex) -> float | None:
    """Nearest person-det distance (ft) to the fitted launch point."""
    fit = row.get("fit")
    if not fit:
        return None
    lt = fit["t_launch"]
    lx, ly = fit["launch_ft"]
    r_ts, r_cx, r_cy, r_w, _ = det.window(
        (lt - 0.6) * 1000, (lt + 0.6) * 1000, CLS_RIM)
    p_ts, p_cx, _p_cy, _p_w, p_bot = det.window(
        (lt - 0.15) * 1000, (lt + 0.15) * 1000, CLS_PERSON)
    if not r_ts.size or not p_ts.size:
        return None
    k = int(np.argmin(np.abs(r_ts - lt * 1000)))
    fpp = RIM_TRUE_FT / max(float(r_w[k]), 1e-6)
    px_ft = (p_cx - float(r_cx[k])) * fpp
    py_ft = (p_bot - float(r_cy[k])) * fpp
    return float(np.hypot(px_ft - lx, py_ft - ly).min())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--video", type=Path)
    ap.add_argument("--game-dir", type=Path)
    ap.add_argument("--wasb-root", type=Path, default=Path("/work/wasb"))
    ap.add_argument("--weights", type=Path, required=False)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--anchors-json", type=Path, default=None,
                    help="external anchors for PBP-less games: JSON list of "
                         '{"ts_ms", "jersey"} (same file shot_sheet_probe '
                         "takes)")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    if args.anchors_json:
        ext = json.loads(args.anchors_json.read_text())
        anchors_ms = [int(a["ts_ms"]) // BIN_MS * BIN_MS for a in ext]
    else:
        tokens = read_stage(args.game_dir / "tokens").to_pylist()
        anchors_ms = [b * BIN_MS for b, _j in _shot_anchors(tokens)]

    det = DetIndex(read_stage(args.game_dir / "detections"))
    rows = flight_fits_for_anchors(
        args.video, anchors_ms, det, args.wasb_root,
        args.weights or args.wasb_root / "wasb_basketball_best.pth.tar",
        log=lambda m: print(f"FLIGHT-FIT: {m}", flush=True))

    n_fit = n_null = n_launch_person = 0
    for row in rows:
        row["anchor_bin"] = row["anchor_ms"] // BIN_MS  # legacy consumers
        if row.get("fit"):
            n_fit += 1
            d = _launch_person_ft(row, det)
            if d is not None and d <= LAUNCH_PERSON_FT:
                n_launch_person += 1
                row["launch_person_ft"] = round(d, 1)
        n_null += bool(row.get("null_fit"))

    n = max(len(anchors_ms), 1)
    out = {
        "game": args.game_dir.name, "anchors": len(anchors_ms),
        "flight_fit_frac": round(n_fit / n, 4),
        "null_fit_frac": round(n_null / n, 4),
        "launch_person_frac_of_fits": (round(n_launch_person / n_fit, 4)
                                       if n_fit else None),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print("FLIGHT_FIT_RESULT " + json.dumps(
        {k: out[k] for k in ("game", "anchors", "flight_fit_frac",
                             "null_fit_frac",
                             "launch_person_frac_of_fits")}), flush=True)


if __name__ == "__main__":
    main()
