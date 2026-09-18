"""Shot-window contact sheets: read the shooter's jersey by pooling every
view of him from before, during and after the shot.

2026-08-17 direction. The existing contact sheet
(pipeline/contact_sheet.py) already tiles many crops into one image and
asks the VLM once -- but it keys on TRACKLET, so it inherits the tracker's
fragmentation: a shooter chopped into five segments yields five partial
sheets, and a shooter the tracker lost at the shot yields none. That join
failure is why mass shot attribution sits at 1.2% (466/37,698 correct,
eval-burn/aggregate.json 2026-08-03).

Keying the pool on the SHOT ANCHOR instead removes the tracker from the
critical path for reading, and lets the read be scored directly against
ground truth: every anchor carries shooter_jersey from the PBP alignment.

Per shooting anchor (+-SHOT_WINDOW_BINS around it) this:
  1. takes each candidate body alive in the window from the binding stage
     (default: quark v1 -- the approved arm, 2026-08-05),
  2. pools that body's tallest observed boxes across the whole window
     (before + during + after) into ONE sheet, and
  3. asks the VLM once per candidate, roster-constrained, cached.

Scored two ways, both honest:
  * RECALL  -- is the true jersey among the candidate reads at all?
    (isolates READING from candidate selection)
  * TOP-1   -- does the single best-ranked candidate read the true jersey?
    (the product number)

Reads are cached by content (VlmCache), so re-runs never re-buy calls.

Usage (inside cvbench):
    python scripts/shot_sheet_probe.py /work/out-harvest/cal_fsu_v3real \
        --video "/work/pairing/videos/...8Bb95rASehA....mp4" \
        --binding-stage /work/models/quark-v1/cal_full/quark_binding \
        --out /work/models/quark-v1/shotsheet_cal.json \
        --max-anchors 20
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from montehall_cv.brain.dataset import BIN_MS
from montehall_cv.brain.eval_slots import SHOT_WINDOW_BINS, _shot_anchors
from montehall_cv.pipeline.contact_sheet import (
    CROPS_PER_SHEET,
    build_sheet,
    _read_sheet,
)
from montehall_cv.pipeline.run_jersey import _collect_crops
from montehall_cv.pipeline.shot_attribution import (
    ball_dets as _ball_dets,
    ball_proximity as _ball_proximity,
    load_binding as _load_binding,
    window_candidates as _window_candidates,
)
from montehall_cv.store.artifacts import read_stage
from montehall_cv.store.vlm_cache import VlmCache

MAX_CANDIDATES = 6      # per anchor; 0 = every body in the window
KEY_PATH = Path("/root/.anthropic_key")


def _api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key.strip()
    if KEY_PATH.exists():
        return KEY_PATH.read_text().strip()
    raise SystemExit("no ANTHROPIC_API_KEY and no /root/.anthropic_key")


def _build_plan(anchors: list, binding: dict, balls: dict | None,
                max_candidates: int,
                release_ms: tuple[int, int] | None = None) -> tuple[dict, dict]:
    """One decode pass for every anchor: frame_idx -> [(cand_key, bbox)]."""
    plan: dict[int, list[tuple]] = defaultdict(list)
    cand_of: dict[int, tuple] = {}   # cand_key -> (anchor_i, track_id)
    next_key = 0
    for ai, (bin_idx, _jersey) in enumerate(anchors):
        lo = (bin_idx - SHOT_WINDOW_BINS) * BIN_MS
        hi = (bin_idx + SHOT_WINDOW_BINS) * BIN_MS
        cands = _window_candidates(binding, lo, hi)
        if balls is None:
            ranked = sorted(cands.items(), key=lambda kv: -len(kv[1]))
        elif release_ms is not None:
            # Anchors sit at the shot's RESULT (rim), so whole-window ball
            # proximity surfaces rebounders. Rank by proximity in the
            # RELEASE window only (anchor+release_ms[0] .. anchor+
            # release_ms[1]); a candidate with no boxes there sorts last.
            a_ms = bin_idx * BIN_MS
            r_lo, r_hi = a_ms + release_ms[0], a_ms + release_ms[1]

            def _release_prox(rows: list[tuple]) -> float:
                sub = [r for r in rows if r_lo <= r[3] < r_hi]
                return _ball_proximity(sub, balls) if sub else float("inf")

            ranked = sorted(cands.items(),
                            key=lambda kv: _release_prox(kv[1]))
        else:  # nearest-to-ball first: the shooter held it before release
            ranked = sorted(cands.items(),
                            key=lambda kv: _ball_proximity(kv[1], balls))
        keep = ranked if max_candidates <= 0 else ranked[:max_candidates]
        for rank, (tid, rows) in enumerate(keep):
            rows.sort(key=lambda r: -r[0])          # tallest first
            key = next_key
            next_key += 1
            cand_of[key] = (ai, tid, rank)
            for row in rows[:CROPS_PER_SHEET]:
                plan[row[1]].append((key, row[2]))
    return dict(plan), cand_of


def _rank_key(c: dict) -> tuple:
    """Selection rank first (ball proximity), VLM confidence as tiebreak."""
    return (c["rank"], -(c.get("confidence") or 0.0))


def _score(anchors: list, cand_of: dict, reads: dict) -> dict:
    """Recall (truth anywhere) and top-1 (best-ranked candidate)."""
    by_anchor: dict[int, list[dict]] = defaultdict(list)
    for key, verdict in reads.items():
        ai, tid, rank = cand_of[key]
        by_anchor[ai].append({"track_id": tid, "rank": rank, **verdict})
    n = hit_any = hit_top1 = attempted = 0
    per: list[dict] = []
    for ai, (_bin, truth) in enumerate(anchors):
        n += 1
        cands = sorted(by_anchor.get(ai, []), key=_rank_key)
        numbers = [c["number"] for c in cands if c["number"]]
        found = str(truth) in numbers
        top1 = bool(numbers) and numbers[0] == str(truth)
        if numbers:
            attempted += 1
        hit_any += found
        hit_top1 += top1
        per.append({"anchor": ai, "truth": str(truth),
                    "reads": numbers, "found": found, "top1": top1,
                    "candidates": [{"rank": c["rank"],
                                    "track_id": c["track_id"],
                                    "number": c["number"],
                                    "confidence": c.get("confidence")}
                                   for c in cands]})
    return {
        "anchors": n,
        "anchors_with_any_read": attempted,
        "recall_truth_among_reads": round(hit_any / max(n, 1), 4),
        "top1_precision_all_anchors": round(hit_top1 / max(n, 1), 4),
        "top1_precision_of_attempted": round(hit_top1 / max(attempted, 1), 4),
        "per_anchor": per,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game_dir", type=Path)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--binding-stage", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES,
                    help="bodies pooled per anchor; 0 = every body in the "
                         "window (isolates selection from reading)")
    ap.add_argument("--rank", choices=("ball", "crops"), default="ball",
                    help="ball = nearest to a detected ball in the window "
                         "(homography-free); crops = readable-crop supply")
    ap.add_argument("--rank-release", type=str, default=None,
                    metavar="LO_MS,HI_MS",
                    help="with --rank ball: restrict proximity to the "
                         "release window, offsets in ms relative to the "
                         "anchor (e.g. '-2500,-1000')")
    ap.add_argument("--max-anchors", type=int, default=0,
                    help="0 = every shooting anchor; else first N (cost gate)")
    ap.add_argument("--anchors-json", type=Path, default=None,
                    help="external anchors for games with no PBP tokens: a "
                         'JSON list of {"ts_ms": int, "jersey": int} objects '
                         "(hand-labeled truth); roster = those jerseys")
    ap.add_argument("--cache", type=Path,
                    default=Path("/work/models/vlm_cache_shotsheet.sqlite"))
    args = ap.parse_args()

    if args.anchors_json:
        ext = json.loads(args.anchors_json.read_text())
        anchors = [(int(a["ts_ms"]) // BIN_MS, int(a["jersey"])) for a in ext]
    else:
        tokens = read_stage(args.game_dir / "tokens").to_pylist()
        anchors = _shot_anchors(tokens)
    if args.max_anchors:
        anchors = anchors[:args.max_anchors]
    if not anchors:
        raise SystemExit("no shooting anchors with readable truth")

    # Roster vocabulary from this game's own PBP jerseys -- what a real
    # deployment has from the box score. Caveat: truth is always in the
    # vocabulary, so this measures reading, not open-set identification.
    roster = {str(j) for _b, j in anchors}

    binding = _load_binding(args.binding_stage)
    balls = _ball_dets(args.game_dir) if args.rank == "ball" else None
    release_ms = None
    if args.rank_release:
        lo_s, hi_s = args.rank_release.split(",")
        release_ms = (int(lo_s), int(hi_s))
    plan, cand_of = _build_plan(anchors, binding, balls, args.max_candidates,
                                release_ms=release_ms)
    print(f"anchors={len(anchors)} candidates={len(cand_of)} "
          f"frames_to_decode={len(plan)}", flush=True)

    crops, meta = _collect_crops(args.video, plan)
    by_key: dict[int, list] = defaultdict(list)
    for crop, (key, _frame_idx) in zip(crops, meta):
        by_key[key].append(crop)
    print(f"crops={len(crops)} sheets={len(by_key)}", flush=True)

    import anthropic
    client = anthropic.Anthropic(api_key=_api_key())
    cache = VlmCache(args.cache)
    reads: dict[int, dict] = {}
    for i, (key, crop_list) in enumerate(sorted(by_key.items())):
        reads[key] = _read_sheet(client, build_sheet(crop_list), roster, cache)
        if (i + 1) % 25 == 0:
            print(f"  read {i + 1}/{len(by_key)} sheets", flush=True)

    report = {
        "game_dir": str(args.game_dir),
        "binding_stage": str(args.binding_stage),
        "window_bins": SHOT_WINDOW_BINS,
        "crops_per_sheet": CROPS_PER_SHEET,
        "max_candidates_per_anchor": args.max_candidates or "all",
        "ranker": args.rank,
        "roster_size": len(roster),
        "sheets_read": len(reads),
        **_score(anchors, cand_of, reads),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    summary = {k: v for k, v in report.items() if k != "per_anchor"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
