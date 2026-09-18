"""Per-candidate ranking features + weak labels for the learned shot ranker.

Structural fix B (2026-08-25): the corpus carries 37,698 PBP-labeled shot
anchors — free supervision for the selection problem (truth reaches the
candidate set at 1.0/.58/.65 but top-1 sits at .50/.07/.15). This extracts,
per (anchor, quark candidate), the GEOMETRY features shot_attribution
computes at inference (candidate_features — one canonical definition) plus
a weak label: does the candidate's jersey read match the PBP shooter?

Label bridge: contact_reads carries only (track_id, number) in ByteTrack id
space; quark segments carry no reads. Both bind the SAME raw detections, so
a per-window IoU match between quark candidate rows and localized cls-0
rows (which carry ByteTrack track_id) transfers numbers near-exactly.

Anchors with no positive candidate are dropped (nothing to supervise —
the conditional ranking task is "given truth present, surface it first",
matching the measured recall->top-1 product framing). Drop counts are
reported, never silent.

Usage (inside cvbench, CPU):
    python scripts/extract_rank_features.py /work/out-extract/eid_X \
        --out /work/models/ranker-v1/features/eid_X.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from montehall_cv.brain.dataset import BIN_MS
from montehall_cv.brain.eval_slots import _shot_anchors
from montehall_cv.pipeline.shot_attribution import (
    BRIDGE_IOU_MIN,
    WINDOW_MS,
    _iou,
    ball_dets,
    candidate_features,
    load_binding,
    localized_person_index as _localized_index,
    window_candidates,
)
from montehall_cv.store.artifacts import read_stage, stage_complete

READ_CONF_MIN = 0.6


def _read_map(game_dir: Path) -> dict[int, str]:
    """bytetrack track_id -> trusted contact-sheet number."""
    out: dict[int, str] = {}
    for row in read_stage(game_dir / "contact_reads").to_pylist():
        if row["number"] and (row["confidence"] or 0.0) >= READ_CONF_MIN:
            out[int(row["track_id"])] = str(row["number"])
    return out


def _candidate_number(rows: list[tuple], loc_idx: dict,
                      reads: dict[int, str]) -> str | None:
    """Majority read number bridged onto one quark candidate's rows."""
    votes: Counter = Counter()
    for row in rows:
        for bt_id, bbox in loc_idx.get(row[1], ()):
            if bt_id in reads and _iou(row[2], bbox) >= BRIDGE_IOU_MIN:
                votes[reads[bt_id]] += 1
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game_dir", type=Path)
    ap.add_argument("--binding-stage", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    game_dir = args.game_dir
    stage = args.binding_stage or game_dir / "quark_binding"
    if args.out.exists():
        print(f"SKIP_DONE {game_dir.name}")
        return
    for need in (stage, game_dir / "tokens", game_dir / "contact_reads",
                 game_dir / "localized"):
        if not stage_complete(need):
            print(f"SKIP_MISSING {game_dir.name} {need.name}")
            return

    anchors = _shot_anchors(read_stage(game_dir / "tokens").to_pylist())
    binding = load_binding(stage)
    balls = ball_dets(game_dir)
    loc_idx = _localized_index(game_dir)
    reads = _read_map(game_dir)

    rows_out: list[dict] = []
    n_dropped = 0
    for ai, (bin_idx, truth) in enumerate(anchors):
        a_ms = bin_idx * BIN_MS
        cands = window_candidates(binding, a_ms - WINDOW_MS, a_ms + WINDOW_MS)
        if not cands:
            n_dropped += 1
            continue
        feats = candidate_features(cands, balls, a_ms, fit=None, det=None)
        numbered = {tid: _candidate_number(cands[tid], loc_idx, reads)
                    for tid in cands}
        labels = {tid: numbered[tid] == str(truth)
                  for tid in cands if numbered[tid]}
        if not any(labels.values()):
            n_dropped += 1
            continue
        for tid, f in feats.items():
            rows_out.append({
                "anchor": ai,
                "track_id": tid,
                "label": bool(labels.get(tid, False)),
                "number": numbered[tid],
                "release_prox_px": (None if f["release_prox_px"] == float("inf")
                                    else f["release_prox_px"]),
                "whole_prox_px": (None if f["whole_prox_px"] == float("inf")
                                  else f["whole_prox_px"]),
                "n_obs": f["n_obs"],
                "max_h_px": f["max_h_px"],
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "game": game_dir.name,
        "anchors_total": len(anchors),
        "anchors_used": len({r["anchor"] for r in rows_out}),
        "anchors_dropped_no_positive": n_dropped,
        "rows": rows_out,
    }))
    print(f"FEATURES_OK {game_dir.name} anchors_used="
          f"{len({r['anchor'] for r in rows_out})} rows={len(rows_out)}",
          flush=True)


if __name__ == "__main__":
    main()
