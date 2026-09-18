"""Launch-point ranker score: flight-fit launch position -> nearest body
-> that body's sheet read (from a shot_sheet_probe JSON with candidates).

Composes three existing artifacts, no new inference:
  * flight_fit_probe --out JSON       (per-anchor t_launch + launch_ft)
  * shot_sheet_probe --out JSON       (per-anchor candidate reads by track)
  * game_dir detections + a binding stage (rim + body boxes)

Prints per-anchor picks and the top-1 line. n is small by design — this is
the ranker A/B for hand-labeled external-anchor games (HS clip).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from shot_sheet_probe import _load_binding  # noqa: E402

from montehall_cv.store.artifacts import read_stage  # noqa: E402

CLS_RIM = 3
RIM_TRUE_FT = 1.5
RIM_CONF_MIN = 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game_dir", type=Path)
    ap.add_argument("--flight", type=Path, required=True)
    ap.add_argument("--sheets", type=Path, required=True)
    ap.add_argument("--binding-stage", type=Path, required=True)
    ap.add_argument("--launch-window-ms", type=int, default=200)
    args = ap.parse_args()

    flight = json.loads(args.flight.read_text())
    sheets = json.loads(args.sheets.read_text())
    b = _load_binding(args.binding_stage)

    det = read_stage(args.game_dir / "detections")
    dts = det.column("ts_ms").to_numpy(zero_copy_only=False)
    dcls = det.column("cls").to_numpy(zero_copy_only=False)
    dconf = det.column("conf").to_numpy(zero_copy_only=False)
    dx1 = det.column("x1").to_numpy(zero_copy_only=False)
    dx2 = det.column("x2").to_numpy(zero_copy_only=False)
    dy1 = det.column("y1").to_numpy(zero_copy_only=False)
    dy2 = det.column("y2").to_numpy(zero_copy_only=False)
    rim = (dcls == CLS_RIM) & (dconf >= RIM_CONF_MIN)

    n = hits = fitted = 0
    for row, per in zip(flight["rows"], sheets["per_anchor"]):
        n += 1
        truth = per["truth"]
        fit = row.get("fit")
        if not fit:
            print(f"anchor {per['anchor']} truth {truth}: NO FIT")
            continue
        fitted += 1
        lt_ms = fit["t_launch"] * 1000.0
        lx, ly = fit["launch_ft"]

        sel = rim & (np.abs(dts - lt_ms) < 600)
        if not sel.any():
            print(f"anchor {per['anchor']} truth {truth}: no rim at launch")
            continue
        k = int(np.argmin(np.abs(dts[sel] - lt_ms)))
        r_cx = float(((dx1[sel] + dx2[sel]) / 2)[k])
        r_cy = float(((dy1[sel] + dy2[sel]) / 2)[k])
        fpp = RIM_TRUE_FT / max(float((dx2[sel] - dx1[sel])[k]), 1e-6)

        win = np.abs(b["ts_ms"] - lt_ms) <= args.launch_window_ms
        if not win.any():
            print(f"anchor {per['anchor']} truth {truth}: no bodies at launch")
            continue
        bx = (b["x1"][win] + b["x2"][win]) / 2
        bb = b["y2"][win]
        tid = b["track_id"][win]
        d_ft = np.hypot((bx - r_cx) * fpp - lx, (bb - r_cy) * fpp - ly)

        read_of = {c["track_id"]: c["number"] for c in per["candidates"]}
        best = sorted(zip(d_ft, tid), key=lambda t: t[0])
        pick = None
        for dist, t in best:
            num = read_of.get(int(t))
            if num:
                pick = (num, round(float(dist), 1), int(t))
                break
        ok = pick is not None and pick[0] == truth
        hits += ok
        print(f"anchor {per['anchor']} truth {truth}: pick {pick} "
              f"{'HIT' if ok else 'miss'}")

    print(json.dumps({
        "anchors": n, "fitted": fitted,
        "top1_launch_rank": round(hits / max(n, 1), 4),
        "top1_of_fitted": round(hits / max(fitted, 1), 4),
    }))


if __name__ == "__main__":
    main()
