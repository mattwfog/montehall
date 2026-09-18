"""v10 residuals probe — the two opens the 08-25 close named, row-observed.

1. ARRIVAL OVER-GENERATION (spine precision .282, 39 shots / 11 truths;
   the held-ball hypothesis was REFUTED by the flight gate's 40->39):
   census every spine shot, classify each extra by what the rows show —
   clustered around a matched truth (rebound-tap swarm), inside a VLM
   window / FT-flagged window (the FT/untimed-real class), or isolated
   (with the ±1.2s track context: sources, descent, rim distance).

2. MAKE/MISS RESIDUAL (5/8; truth-makes at 22200/116800/153200 predicted
   miss): per scored truth row, every channel's verdict side by side —
   box_events final, VLM verdict_made, spine made_geom — plus the
   rim-pass row evidence (above/below-mouth in-x counts vs the static
   rim) for the mismatches.

Read-only over existing stage artifacts; no inference.

Usage (inside cvbench):
    python scripts/spine_residuals_probe.py /work/out-harvest/hs_clip_20260824 \
        --truth /work/hs_clip_truth_20260824.json \
        --out /work/models/quark-v1/spine_residuals_v10.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from montehall_cv.pipeline.ball_track import (
    RIM_PASS_WINDOW_MS,
    RIM_PASS_X_FRAC,
    static_rims,
)
from montehall_cv.store.artifacts import read_stage, stage_complete

# local import of the sibling probe's loaders (scripts/ is not a package)
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "spine_coverage_probe", Path(__file__).parent / "spine_coverage_probe.py")
if _spec is None or _spec.loader is None:
    raise ImportError("spine_coverage_probe.py not found beside this probe")
_scp = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_scp)

MATCH_PAD_MS = 3000  # = score_e2e_vs_anchors.MATCH_PAD_MS
CONTEXT_MS = 1200


def greedy_match(truth_ts: list[int], shot_ts: list[int]) -> dict[int, int]:
    """truth ts -> shot ts, one-to-one nearest-first (the scorer's rule)."""
    pairs = sorted((abs(t - s), t, s) for t in truth_ts for s in shot_ts
                   if abs(t - s) <= MATCH_PAD_MS)
    used_t, used_s, out = set(), set(), {}
    for _d, t, s in pairs:
        if t in used_t or s in used_s:
            continue
        out[t] = s
        used_t.add(t)
        used_s.add(s)
    return out


def track_context(track: list[dict], ts: int) -> dict:
    win = [r for r in track if abs(r["ts_ms"] - ts) <= CONTEXT_MS]
    conf = [r for r in win if r["source"] != "interp"]
    before = [r for r in conf if r["ts_ms"] < ts][-6:]
    ys = [r["y"] for r in before]
    return {
        "rows": len(win),
        "confirmed": len(conf),
        "by_source": {s: sum(1 for r in win if r["source"] == s)
                      for s in ("det", "wasb", "both", "interp")},
        "descending_into": (len(ys) >= 2 and ys[-1] > ys[0]),
        "pre_rows_y": [round(y, 1) for y in ys],
    }


def rim_pass_rows(track: list[dict], ts: int, rim: tuple | None) -> dict:
    """The evidence rim_pass_verdict judges: rows around ts vs the mouth."""
    if rim is None:
        return {"rim": None}
    x1, y1, x2, y2 = rim
    cx, w = (x1 + x2) / 2, (x2 - x1)
    lo, hi = ts + RIM_PASS_WINDOW_MS[0], ts + RIM_PASS_WINDOW_MS[1]
    rows = [r for r in track if lo <= r["ts_ms"] <= hi]
    in_x = [r for r in rows if abs(r["x"] - cx) <= RIM_PASS_X_FRAC * w]
    return {
        "rim": [round(v, 1) for v in rim],
        "rows_in_window": len(rows),
        "rows_in_x": len(in_x),
        "above_mouth_in_x": sum(1 for r in in_x if r["y"] < y1),
        "below_rim_in_x": sum(1 for r in in_x if r["y"] > y2),
        "trace": [{"ts": r["ts_ms"], "x": round(r["x"], 1),
                   "y": round(r["y"], 1), "src": r["source"]}
                  for r in rows][:40],
    }


def nearest_static_rim(rims_static: list[tuple], x: float) -> tuple | None:
    if not rims_static:
        return None
    return min(rims_static, key=lambda b: abs((b[0] + b[2]) / 2 - x))


def classify_extra(shot: dict, matched_truth: dict[int, int],
                   windows: list[dict], ft_eids: set[int],
                   track: list[dict]) -> dict:
    ts = shot["ts_ms"]
    near_matched_truth = [t for t in matched_truth
                          if abs(t - ts) <= MATCH_PAD_MS]
    in_windows = [w for w in windows
                  if w["ts_start_ms"] - MATCH_PAD_MS <= ts
                  <= w["ts_end_ms"] + MATCH_PAD_MS]
    in_ft = [w["event_id"] for w in in_windows if w["event_id"] in ft_eids]
    if near_matched_truth:
        kind = "cluster_around_matched_truth"
    elif in_ft:
        kind = "ft_window"
    elif in_windows:
        kind = "vlm_window_no_truth"
    else:
        kind = "isolated"
    return {
        "ts_ms": ts,
        "release_ts_ms": shot.get("release_ts_ms"),
        "track_id": shot.get("track_id"),
        "made_geom": shot.get("made_geom"),
        "class": kind,
        "near_matched_truth_ts": near_matched_truth,
        "window_eids": [w["event_id"] for w in in_windows],
        "ft_window_eids": in_ft,
        "context": track_context(track, ts) if kind == "isolated" else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game_dir", type=Path)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    jd = args.game_dir

    truth = json.loads(args.truth.read_text())["timed_events"]
    track = _scp.load_track(jd)
    rims_by_frame = _scp.load_rims(jd)
    rims_static = static_rims(rims_by_frame)
    events = read_stage(jd / "ball_events").to_pylist()
    shots = sorted((r for r in events if r["kind"] == "shot"),
                   key=lambda r: r["ts_ms"])
    windows = [w for w in read_stage(jd / "shot_events").to_pylist()
               if w["verdict_attempt"]]
    all_windows = read_stage(jd / "shot_events").to_pylist()
    ft_eids = set()
    if stage_complete(jd / "ft_events"):
        ft_eids = {int(r["event_id"])
                   for r in read_stage(jd / "ft_events").to_pylist()
                   if r["is_ft"]}

    truth_ts = [int(t["ts_ms"]) for t in truth]
    match = greedy_match(truth_ts, [s["ts_ms"] for s in shots])
    matched_shot_ts = set(match.values())

    extras = [classify_extra(s, match, all_windows, ft_eids, track)
              for s in shots if s["ts_ms"] not in matched_shot_ts]
    dts = [b["ts_ms"] - a["ts_ms"] for a, b in zip(shots, shots[1:])]

    # ---- part 2: make/miss channels per scored truth row
    fgm_ts = set()
    if stage_complete(jd / "box_events"):
        fgm_ts = {r["ts_ms"] for r in read_stage(jd / "box_events").to_pylist()
                  if r["event_type"] in ("FGM", "FTM")}
    wmatch = greedy_match(truth_ts,
                          [w["ts_start_ms"] for w in windows])
    win_by_start = {w["ts_start_ms"]: w for w in windows}
    shot_by_ts = {s["ts_ms"]: s for s in shots}
    make_rows = []
    for t in truth:
        if t.get("uncertain"):
            continue
        wstart = wmatch.get(int(t["ts_ms"]))
        w = win_by_start.get(wstart) if wstart is not None else None
        spine = shot_by_ts.get(match.get(int(t["ts_ms"])))
        pred = (wstart in fgm_ts) if wstart is not None else None
        row = {
            "truth_ts": t["ts_ms"], "truth_made": t["made"],
            "pred_made": pred,
            "vlm_verdict_made": (w or {}).get("verdict_made"),
            "vlm_confidence": (w or {}).get("confidence"),
            "spine_shot_ts": spine["ts_ms"] if spine else None,
            "spine_made_geom": spine.get("made_geom") if spine else None,
        }
        if pred != t["made"] and spine is not None:
            rim = nearest_static_rim(rims_static, spine["x"])
            row["rim_pass_evidence"] = rim_pass_rows(
                track, spine["ts_ms"], rim)
        make_rows.append(row)

    report = {
        "game_dir": str(jd),
        "spine_shots": len(shots),
        "matched_to_truth": len(matched_shot_ts),
        "extras": len(extras),
        "extras_by_class": {k: sum(1 for e in extras if e["class"] == k)
                            for k in ("cluster_around_matched_truth",
                                      "ft_window", "vlm_window_no_truth",
                                      "isolated")},
        "inter_shot_dt_ms_under_3s": sum(1 for d in dts if d < 3000),
        "static_rims": [[round(v, 1) for v in b] for b in rims_static],
        "extras_rows": extras,
        "make_miss_rows": make_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("extras_rows", "make_miss_rows")},
                     indent=2))
    print(json.dumps(report["make_miss_rows"], indent=2))


if __name__ == "__main__":
    main()
