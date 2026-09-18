"""Team-filtered shooter ranking over existing shot-sheet artifacts.

The candidate pool mixes both teams (both may even share numbers — the
HS clip has a 21 and a 23 on EACH side), so every colorblind ranker
is capped. Product-legitimate fix: jersey COLOR is customer-supplied
config in deployment (job params carry light/dark shirt colors), so:

  1. classify every candidate white/dark from its own torso crops
     (median luminance split — no model),
  2. infer the SHOOTING team per anchor as the color of the bodies
     nearest the ball in the release window (majority of top 3),
  3. drop other-color candidates, then rank as before:
     launch-point pick when a flight fit exists, release-window
     ball-proximity otherwise.

Zero new VLM spend: reads come from the shot_sheet_probe JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from montehall_cv.brain.dataset import BIN_MS  # noqa: E402
from montehall_cv.brain.eval_slots import SHOT_WINDOW_BINS  # noqa: E402
from montehall_cv.pipeline.jersey import torso_crop  # noqa: E402
from montehall_cv.pipeline.shot_attribution import (  # noqa: E402
    ball_dets as _ball_dets,
    ball_proximity as _ball_proximity,
    load_binding as _load_binding,
    window_candidates as _window_candidates,
)
from montehall_cv.store.artifacts import read_stage  # noqa: E402
from montehall_cv.pipeline.video import decode_frames  # noqa: E402

CLS_RIM = 3
RIM_TRUE_FT = 1.5


def _luminance(crop: np.ndarray) -> float:
    return float(np.median(crop.astype(np.float32).mean(axis=2)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game_dir", type=Path)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--sheets", type=Path, required=True)
    ap.add_argument("--flight", type=Path, required=True)
    ap.add_argument("--binding-stage", type=Path, required=True)
    ap.add_argument("--anchors-json", type=Path, required=True)
    ap.add_argument("--release-ms", type=str, default="-2500,-800")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ext = json.loads(args.anchors_json.read_text())
    anchors = [(int(a["ts_ms"]) // BIN_MS, int(a["jersey"])) for a in ext]
    given_team = [a.get("team") for a in ext]  # optional per-anchor
    # "white"/"dark" override: the oracle-team ceiling (in deployment,
    # team comes from customer config + possession, not inference)
    sheets = json.loads(args.sheets.read_text())
    flight = json.loads(args.flight.read_text())
    binding = _load_binding(args.binding_stage)
    balls = _ball_dets(args.game_dir)
    r_lo_off, r_hi_off = (int(x) for x in args.release_ms.split(","))

    det = read_stage(args.game_dir / "detections")
    dts = det.column("ts_ms").to_numpy(zero_copy_only=False)
    dcls = det.column("cls").to_numpy(zero_copy_only=False)
    dconf = det.column("conf").to_numpy(zero_copy_only=False)
    dx1 = det.column("x1").to_numpy(zero_copy_only=False)
    dx2 = det.column("x2").to_numpy(zero_copy_only=False)
    dy1 = det.column("y1").to_numpy(zero_copy_only=False)
    dy2 = det.column("y2").to_numpy(zero_copy_only=False)
    rim_sel = (dcls == CLS_RIM) & (dconf >= 0.5)

    # ---- one decode pass: 2 crops per candidate for color ----
    plan: dict[int, list[tuple]] = {}
    cand_rows: dict[tuple[int, int], list] = {}
    for ai, (bin_idx, _t) in enumerate(anchors):
        lo = (bin_idx - SHOT_WINDOW_BINS) * BIN_MS
        hi = (bin_idx + SHOT_WINDOW_BINS) * BIN_MS
        wc = _window_candidates(binding, lo, hi)
        for tid, rows in wc.items():
            rows.sort(key=lambda r: -r[0])
            cand_rows[(ai, tid)] = rows
            for row in rows[:2]:
                plan.setdefault(row[1], []).append(((ai, tid), row[2]))
    lums: dict[tuple[int, int], list[float]] = {}
    remaining = set(plan)
    for frame in decode_frames(args.video):
        if frame.frame_idx in plan:
            for key, bbox in plan[frame.frame_idx]:
                crop = torso_crop(frame.image, *bbox)
                if crop is not None:
                    lums.setdefault(key, []).append(_luminance(crop))
            remaining.discard(frame.frame_idx)
            if not remaining:
                break
    all_lum = sorted(np.median(v) for v in lums.values())
    split = float(np.median(all_lum))  # bimodal white/dark -> global split
    color_of = {k: ("white" if np.median(v) >= split else "dark")
                for k, v in lums.items()}

    per = []
    hits = 0
    for ai, (bin_idx, truth) in enumerate(anchors):
        a_ms = bin_idx * BIN_MS
        reads = {c["track_id"]: c["number"]
                 for c in sheets["per_anchor"][ai]["candidates"]}
        cands = {tid: rows for (a, tid), rows in cand_rows.items() if a == ai}

        def rel_prox(rows: list) -> float:
            sub = [r for r in rows
                   if a_ms + r_lo_off <= r[3] < a_ms + r_hi_off]
            return _ball_proximity(sub, balls) if sub else float("inf")

        by_rel = sorted(cands.items(), key=lambda kv: rel_prox(kv[1]))
        # shooting team = majority color of 3 most ball-proximate bodies
        top_colors = [color_of.get((ai, tid)) for tid, _ in by_rel[:3]]
        top_colors = [c for c in top_colors if c]
        team = max(set(top_colors), key=top_colors.count) if top_colors \
            else None
        if given_team[ai]:
            team = given_team[ai]

        keep = [tid for tid, _ in by_rel
                if team is None or color_of.get((ai, tid)) == team]

        pick = None
        fit = flight["rows"][ai].get("fit")
        if fit:
            lt_ms = fit["t_launch"] * 1000.0
            lx, ly = fit["launch_ft"]
            rsel = rim_sel & (np.abs(dts - lt_ms) < 600)
            if rsel.any():
                k = int(np.argmin(np.abs(dts[rsel] - lt_ms)))
                r_cx = float(((dx1[rsel] + dx2[rsel]) / 2)[k])
                r_cy = float(((dy1[rsel] + dy2[rsel]) / 2)[k])
                fpp = RIM_TRUE_FT / max(float((dx2[rsel] - dx1[rsel])[k]),
                                        1e-6)
                scored = []
                for tid in keep:
                    near = [r for r in cands[tid]
                            if abs(r[3] - lt_ms) <= 200]
                    if not near or not reads.get(tid):
                        continue
                    bx = np.array([(r[2][0] + r[2][2]) / 2 for r in near])
                    bb = np.array([r[2][3] for r in near])
                    d = np.hypot((bx - r_cx) * fpp - lx,
                                 (bb - r_cy) * fpp - ly)
                    scored.append((float(d.min()), tid))
                if scored:
                    scored.sort()
                    pick = ("launch", scored[0][1], reads[scored[0][1]])
        if pick is None:
            for tid in keep:
                if reads.get(tid):
                    pick = ("release", tid, reads[tid])
                    break
        ok = pick is not None and pick[2] == str(truth)
        hits += ok
        per.append({"anchor": ai, "truth": str(truth), "team": team,
                    "kept": len(keep), "of": len(cands),
                    "pick": pick, "hit": ok})
        print(f"anchor {ai} truth {truth} team={team} "
              f"kept {len(keep)}/{len(cands)} pick={pick} "
              f"{'HIT' if ok else 'miss'}")

    result = {"anchors": len(anchors), "top1_teamfiltered":
              round(hits / max(len(anchors), 1), 4), "per_anchor": per}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("anchors", "top1_teamfiltered")}))


if __name__ == "__main__":
    main()
