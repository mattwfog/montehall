"""Anchors-truth JSON from an aligned broadcast game — the corpus bridge.

score_e2e_vs_anchors.py is the ONE scorer that writes SCORECARD.json
(2026-08-25). Broadcast games already have machine truth — ESPN PBP
aligned to video time by eval/pbp_align — so this converts a
`pbp_alignment` artifact into the same truth format the hand-labeled
HS-clip files use (scripts/hs_clip*_truth_*.json), and broadcast
footage scores through the same scorer onto the same scorecard.

Conventions carried from the hand-truth files:
  - FT plays (FT_TYPES — the ESPN type name 'MadeFreeThrow' covers
    makes AND misses; scoring_play is the outcome) land in
    untimed_truth, matching the HS-clip files' FT convention, with their
    video timestamps noted in the description.
  - team is OMITTED: PBP truth is home/away while the pipeline claims
    appearance clusters; there is no ratified mapping — the scorer
    skips team accuracy on rows without a team.
  - match_pad_ms=5000: a PBP clock stamp lags the actual release by up
    to ~6 s (training/mine_broadcast.WINDOW_PRE_S), so the hand-truth
    default +-3 s is too tight for PBP-derived timestamps; +-5 s is the
    action-spotting convention the July broadcast evals reported at.
  - box_truth: game-total FGA/FGM/FTA/FTM/AST for the scorer's box
    level (the retired three-level pbp_score's unique capability).

CLI (one game, inside cvbench):
    python -m montehall_cv.eval.pbp_truth /work/align/eid_401706868 \
        --out /work/truth/eid_401706868_truth.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

MATCH_PAD_MS = 5000
# ESPN type is 'MadeFreeThrow' (no space) for makes AND misses — the type
# name lies; scoring_play carries the actual outcome (verified over all 45
# FT rows of game02, 2026-07-13).
FT_TYPES = ("freethrow",)
# a confident-clock-read gap this long marks non-live footage (replay,
# cut, break) — mine_broadcast.NEGATIVE_GAP_S, the July miner's negative
# sampler, same convention
LIVE_GAP_S = 8.0


def _read_parquet_rows(stage_dir: Path) -> list[dict]:
    import pyarrow.parquet as pq

    parts = sorted(glob.glob(str(stage_dir / "part-*.parquet")))
    rows: list[dict] = []
    for part in parts:
        rows.extend(pq.read_table(part).to_pylist())
    return rows


def _is_ft(play: dict) -> bool:
    t = (play.get("type_text") or "").lower().replace(" ", "")
    return any(ft in t for ft in FT_TYPES)


def live_spans_from_reads(reads: list[dict],
                          gap_s: float = LIVE_GAP_S) -> list[list[float]]:
    """Confident clock reads -> [[start_s, end_s], ...] live-play spans.

    The game clock is only on screen during live play; a gap in confident
    reads longer than gap_s is a replay/cut/break span.
    """
    times = sorted(r["video_t"] for r in reads
                   if (r.get("conf") or 0) >= 0.85
                   and r.get("clock_s") is not None)
    spans: list[list[float]] = []
    for t in times:
        if spans and t - spans[-1][1] <= gap_s:
            spans[-1][1] = t
        else:
            spans.append([t, t])
    return [s for s in spans if s[1] > s[0]]


def truth_from_alignment(align_dir: Path, start_s: float = 0.0,
                         end_s: float | None = None) -> dict:
    plays = _read_parquet_rows(align_dir / "pbp_alignment")
    if not plays:
        raise RuntimeError(
            f"no aligned plays under {align_dir} — run pbp_align first")
    spans = live_spans_from_reads(
        _read_parquet_rows(align_dir / "clock_reads"))
    return truth_from_plays(plays, align_dir.name, start_s=start_s,
                            end_s=end_s, live_spans_s=spans)


def truth_from_plays(plays: list[dict], name: str, start_s: float = 0.0,
                     end_s: float | None = None,
                     live_spans_s: list[list[float]] | None = None) -> dict:
    # segment mode (the cheap layer): truth for an ffmpeg-cut
    # [start_s, end_s) of the source video, timestamps shifted to the
    # segment's clock so the cut file scores like any other clip
    if start_s or end_s is not None:
        hi = float("inf") if end_s is None else end_s
        plays = [{**p, "video_t": p["video_t"] - start_s}
                 for p in plays if start_s <= p["video_t"] < hi]
        if live_spans_s:
            live_spans_s = [
                [max(a - start_s, 0.0), b - start_s]
                for a, b in live_spans_s
                if b > start_s and a < hi]
            if end_s is not None:
                live_spans_s = [[a, min(b, end_s - start_s)]
                                for a, b in live_spans_s]
        name = (f"{name} segment [{start_s:g}s, "
                f"{'end' if end_s is None else f'{end_s:g}s'}) of source")
    shots = sorted((p for p in plays if p["shooting_play"]),
                   key=lambda r: r["video_t"])
    timed: list[dict] = []
    untimed: list[dict] = []
    for p in shots:
        jersey = p.get("shooter_jersey")
        made = bool(p["scoring_play"])
        # ESPN score_value is the attempt's worth even on a miss; the
        # hand-truth convention scores 0 points on a miss.
        points = int(p["score_value"] or 0) if made else 0
        if _is_ft(p):
            untimed.append({
                "what": (f"FT {'make' if made else 'miss'} by "
                         f"#{jersey or '?'} at video_t={p['video_t']:.1f}s "
                         f"({p['text']})"),
                "points": points,
            })
            continue
        timed.append({
            "ts_ms": round(p["video_t"] * 1000),
            "jersey": (int(jersey)
                       if jersey and str(jersey).isdigit() else None),
            "made": made,
            "points": points,
            "play_id": p["play_id"],
        })
    fgs = [p for p in shots if not _is_ft(p)]
    fts = [p for p in shots if _is_ft(p)]
    return {
        "comment": (f"PBP-derived truth from {name}: ESPN plays "
                    f"aligned to video clock by eval/pbp_align "
                    f"({len(timed)} timed FG attempts, {len(untimed)} FTs "
                    f"untimed per the hand-truth convention). Machine "
                    f"truth, not hand labels — timestamps carry PBP clock "
                    f"lag, hence match_pad_ms."),
        "match_pad_ms": MATCH_PAD_MS,
        # non-live footage (replays/cuts, no clock on screen) — the
        # scorer masks continuous-ball coverage to these spans
        "live_spans_ms": [[round(a * 1000), round(b * 1000)]
                          for a, b in (live_spans_s or [])],
        "timed_events": timed,
        "untimed_truth": untimed,
        # game-total truth for the scorer's box level (the retired
        # pbp_score's unique capability, ported 2026-08-26)
        "box_truth": {
            "fga": len(fgs),
            "fgm": sum(1 for p in fgs if p["scoring_play"]),
            "fta": len(fts),
            "ftm": sum(1 for p in fts if p["scoring_play"]),
            "ast": sum(1 for p in plays
                       if "assist" in (p.get("text") or "").lower()),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("align_dir", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--start-s", type=float, default=0.0,
                    help="segment mode: source video_t the cut starts at")
    ap.add_argument("--end-s", type=float, default=None,
                    help="segment mode: source video_t the cut ends at")
    args = ap.parse_args()
    doc = truth_from_alignment(args.align_dir, start_s=args.start_s,
                               end_s=args.end_s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v)
                      for k, v in doc.items()}, indent=2))


if __name__ == "__main__":
    main()
