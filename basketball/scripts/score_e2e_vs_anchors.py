"""Score the FULL pipeline output against hand truth — no hand anchors in.

The end-to-end judge (structural fix A's honesty gate): every prior HS
number was attribution-given-hand-anchor; this scores what the system
emits from raw video — shot_events windows, shot_attribution picks,
box_events verdicts — against a truth file of timed events.

Matching: a truth event claims the nearest attempt window whose
[ts_start-pad, ts_end+pad] covers it (greedy by distance, one-to-one).

Reported, in order of product weight:
  detection   attempt recall / precision over timed truth
  attribution end-to-end named-shooter accuracy (truth rows with a jersey)
  make/miss   verdict accuracy on matched windows (box_events, i.e. after
              any scoreboard reconciliation)

Usage (inside cvbench):
    python scripts/score_e2e_vs_anchors.py /work/out-harvest/hs_clip_20260824 \
        --truth /work/hs_clip_truth_20260824.json --out /work/models/quark-v1/e2e_hsclip.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from montehall_cv.store.artifacts import read_stage, stage_complete

MATCH_PAD_MS = 3000  # hand-truth default; a truth file may override via
#                      match_pad_ms (PBP-derived truth carries clock lag —
#                      eval/pbp_truth writes 5000)


def _match(truth: list[dict], windows: list[dict],
           pad_ms: int) -> dict[int, int | None]:
    """truth index -> event_id (one-to-one, nearest first)."""
    pairs = []
    for ti, t in enumerate(truth):
        for w in windows:
            lo = w["ts_start_ms"] - pad_ms
            hi = w["ts_end_ms"] + pad_ms
            if lo <= t["ts_ms"] <= hi:
                mid = (w["ts_start_ms"] + w["ts_end_ms"]) / 2
                pairs.append((abs(t["ts_ms"] - mid), ti, w["event_id"]))
    pairs.sort()
    used_t: set[int] = set()
    used_w: set[int] = set()
    out: dict[int, int | None] = {ti: None for ti in range(len(truth))}
    for _d, ti, eid in pairs:
        if ti in used_t or eid in used_w:
            continue
        out[ti] = eid
        used_t.add(ti)
        used_w.add(eid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game_dir", type=Path)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scorecard", type=Path, default=None,
                    help="append this run's headline numbers to the "
                         "checked-in SCORECARD.json (anti-goal-hack gate, "
                         "ratified 2026-08-25: comparison claims cite "
                         "scorecard entries, written ONLY by this scorer)")
    args = ap.parse_args()
    truth_doc = json.loads(args.truth.read_text())
    truth = truth_doc["timed_events"]
    pad_ms = int(truth_doc.get("match_pad_ms", MATCH_PAD_MS))

    windows = [w for w in read_stage(args.game_dir / "shot_events").to_pylist()
               if w["verdict_attempt"]]
    match = _match(truth, windows, pad_ms)

    picks: dict[int, tuple[str, float]] = {}
    person_picks: dict[int, dict] = {}
    if stage_complete(args.game_dir / "shot_attribution"):
        for row in read_stage(args.game_dir / "shot_attribution").to_pylist():
            if not row["picked"]:
                continue
            if row["number"]:  # number = pre-fill HINT (person-only era)
                picks[int(row["event_id"])] = (
                    str(row["number"]), float(row["read_confidence"] or 0.0))
            person_picks[int(row["event_id"])] = {
                "person_id": row["person_id"], "team": row["team"]}

    made_of: dict[int, bool] = {}
    if stage_complete(args.game_dir / "box_events"):
        fgm_ts = set()
        for r in read_stage(args.game_dir / "box_events").to_pylist():
            if r["event_type"] in ("FGM", "FTM"):
                fgm_ts.add(r["ts_ms"])
        for w in windows:
            made_of[w["event_id"]] = w["ts_start_ms"] in fgm_ts
    else:  # pre-boxscore: raw adjudicator verdicts
        for w in windows:
            made_of[w["event_id"]] = bool(w["verdict_made"])

    per = []
    n_det = n_named_truth = n_named_hit = n_made_scored = n_made_hit = 0
    for ti, t in enumerate(truth):
        eid = match[ti]
        row = {"ts_ms": t["ts_ms"], "jersey": t.get("jersey"),
               "made_truth": t["made"], "event_id": eid}
        if eid is not None:
            n_det += 1
            pick = picks.get(eid)
            row["pick"] = pick
            pp = person_picks.get(eid)
            if pp is not None:
                row["person_pick"] = pp
            if t.get("jersey") is not None:
                n_named_truth += 1
                row["named_hit"] = (pick is not None
                                    and pick[0] == str(t["jersey"]))
                n_named_hit += row["named_hit"]
            if not t.get("uncertain"):
                n_made_scored += 1
                row["made_pred"] = made_of.get(eid)
                row["made_hit"] = made_of.get(eid) == t["made"]
                n_made_hit += row["made_hit"]
        per.append(row)

    # ---- PERSON-SPACE metrics (people-only ruling, 2026-08-25): the
    # machine claims persons; jersey accuracy is a POST-NAMING metric.
    person_block = {}
    if person_picks:
        team_scored = team_hit = 0
        for ti, t in enumerate(truth):
            eid = match[ti]
            pp = person_picks.get(eid) if eid is not None else None
            if pp is None or not t.get("team") or pp["team"] is None:
                continue
            team_scored += 1
            team_hit += pp["team"] == t["team"]
        # consistency: truth rows of the SAME (jersey, team) must land on
        # the same person
        groups: dict[tuple, set] = {}
        for ti, t in enumerate(truth):
            eid = match[ti]
            pp = person_picks.get(eid) if eid is not None else None
            if pp is None or t.get("jersey") is None or pp["person_id"] is None:
                continue
            groups.setdefault((str(t["jersey"]), t.get("team")),
                              set()).add(pp["person_id"])
        multi = {k: v for k, v in groups.items() if len(v) >= 1}
        consistent = sum(1 for v in groups.values() if len(v) == 1)
        # oracle one-touch naming: the best unique person->jersey
        # assignment the truth itself allows (simulates the coach naming
        # each person once from a perfect sheet)
        votes: dict[int, dict] = {}
        for ti, t in enumerate(truth):
            eid = match[ti]
            pp = person_picks.get(eid) if eid is not None else None
            if pp is None or t.get("jersey") is None or pp["person_id"] is None:
                continue
            votes.setdefault(pp["person_id"], {}).setdefault(
                str(t["jersey"]), 0)
            votes[pp["person_id"]][str(t["jersey"])] += 1
        ranked = sorted(
            ((n, pid, j) for pid, js in votes.items()
             for j, n in js.items()), reverse=True)
        naming: dict[int, str] = {}
        used: set[str] = set()
        for _n, pid, j in ranked:
            if pid not in naming and j not in used:
                naming[pid] = j
                used.add(j)
        post_scored = post_hit = 0
        for ti, t in enumerate(truth):
            eid = match[ti]
            pp = person_picks.get(eid) if eid is not None else None
            if pp is None or t.get("jersey") is None:
                continue
            post_scored += 1
            post_hit += (pp["person_id"] is not None
                         and naming.get(pp["person_id"]) == str(t["jersey"]))
        person_block = {
            "person_claims": len(person_picks),
            "team_scored": team_scored,
            "team_hits": team_hit,
            "team_accuracy": round(team_hit / max(team_scored, 1), 4),
            "person_groups": {f"{j}/{tm}": sorted(v)
                              for (j, tm), v in multi.items()},
            "person_consistency": round(
                consistent / max(len(groups), 1), 4),
            "oracle_naming": {str(p): j for p, j in naming.items()},
            "post_naming_scored": post_scored,
            "post_naming_hits": post_hit,
            "post_naming_accuracy": round(
                post_hit / max(post_scored, 1), 4),
        }

    matched_eids = {e for e in match.values() if e is not None}
    extra = [w["event_id"] for w in windows
             if w["event_id"] not in matched_eids]

    # box level (ported from the retired eval/pbp_score, 2026-08-26):
    # game-total deltas vs a truth doc that carries box_truth (the
    # PBP-derived files do; hand-truth files without it skip this block)
    box_block = {}
    box_truth = truth_doc.get("box_truth")
    if box_truth and stage_complete(args.game_dir / "box_score"):
        stack = {
            key: sum(int(r.get(key) or 0) for r in
                     read_stage(args.game_dir / "box_score").to_pylist())
            for key in box_truth
        }
        box_block = {
            "box_truth": box_truth,
            "box_stack": stack,
            "box_deltas": {k: stack[k] - box_truth[k] for k in box_truth},
        }

    # continuous ball coverage (design decision 2026-08-26: shots alone are
    # not enough — the brain must track the ball THROUGHOUT
    # the film). Coverage of the whole clip by the spine, not just at
    # truth shots. Denominator = the extraction's observed frame span
    # (max detections frame_idx + 1 — artifacts carry no video length).
    # Coverage is presence, NOT correctness: a wrong track counts; track
    # correctness needs its own truth source (open).
    ball_block = {}
    if (stage_complete(args.game_dir / "ball_track")
            and stage_complete(args.game_dir / "detections")):
        det_frames = read_stage(args.game_dir / "detections")\
            .column("frame_idx").to_pylist()
        frames_total = (max(det_frames) + 1) if det_frames else 0
        track = sorted(
            read_stage(args.game_dir / "ball_track").to_pylist(),
            key=lambda r: r["frame_idx"])
        confirmed = {r["frame_idx"] for r in track
                     if r["source"] != "interp"}
        tracked = {r["frame_idx"] for r in track}
        gaps_ms = []
        if track:
            prev_ms = 0.0
            for r in track:
                gaps_ms.append(r["ts_ms"] - prev_ms)
                prev_ms = r["ts_ms"]
            # trailing gap: untracked tail, at the track's own ms/frame
            span_ms = max(r["ts_ms"] for r in track)
            gaps_ms.append(
                (frames_total - 1 - track[-1]["frame_idx"])
                * (span_ms / max(track[-1]["frame_idx"], 1)))
        ball_block = {
            "ball_frames_total": frames_total,
            "ball_frames_confirmed": len(confirmed),
            "ball_frames_tracked": len(tracked),
            "ball_coverage_confirmed": round(
                len(confirmed) / max(frames_total, 1), 4),
            "ball_coverage_tracked": round(
                len(tracked) / max(frames_total, 1), 4),
            "ball_longest_gap_ms": int(max(gaps_ms)) if gaps_ms else None,
            "ball_segments": len({r["seg_id"] for r in track}),
        }
        # live-masked coverage: PBP-derived truth carries live_spans_ms
        # (clock-on-screen spans) — replays/cuts have no trackable ball
        # and deflate the raw number. Hand-truth gym clips (continuous
        # live) carry no spans and skip this.
        live_spans = truth_doc.get("live_spans_ms") or []
        if live_spans and track:
            span_ms = max(r["ts_ms"] for r in track)
            ms_per_frame = span_ms / max(track[-1]["frame_idx"], 1)
            ts_of = {r["frame_idx"]: r["ts_ms"] for r in track}

            def _live(f: int) -> bool:
                t = ts_of.get(f, f * ms_per_frame)
                return any(a <= t <= b for a, b in live_spans)

            live_total = sum(1 for f in range(frames_total) if _live(f))
            live_confirmed = sum(1 for f in confirmed if _live(f))
            ball_block["ball_live_frames_total"] = live_total
            ball_block["ball_coverage_live"] = round(
                live_confirmed / max(live_total, 1), 4)

    # ball-spine detection channel (brick 4): spine shots vs the same
    # truth, independent of run_shots' VLM windows
    spine_block = {}
    if stage_complete(args.game_dir / "ball_events"):
        spine_shots = sorted(
            r["ts_ms"] for r in
            read_stage(args.game_dir / "ball_events").to_pylist()
            if r["kind"] == "shot")
        pairs = sorted(
            (abs(t["ts_ms"] - s), ti, s)
            for ti, t in enumerate(truth) for s in spine_shots
            if abs(t["ts_ms"] - s) <= pad_ms)
        st, ss = set(), set()
        for _d, ti, s in pairs:
            if ti in st or s in ss:
                continue
            st.add(ti)
            ss.add(s)
        spine_block = {
            "spine_shots": len(spine_shots),
            "spine_detection_recall": round(len(st) / max(len(truth), 1), 4),
            "spine_detection_precision": round(
                len(ss) / max(len(spine_shots), 1), 4),
        }
    out = {
        "game_dir": str(args.game_dir),
        "match_pad_ms": pad_ms,
        "timed_truth": len(truth),
        "attempt_windows": len(windows),
        "detection_recall": round(n_det / max(len(truth), 1), 4),
        "detection_precision": round(
            len(matched_eids) / max(len(windows), 1), 4),
        "unmatched_windows": extra,
        "e2e_named_truth_rows": n_named_truth,
        "e2e_named_hits": n_named_hit,
        "e2e_named_accuracy": round(n_named_hit / max(n_named_truth, 1), 4),
        "make_miss_scored": n_made_scored,
        "make_miss_hits": n_made_hit,
        "make_miss_accuracy": round(n_made_hit / max(n_made_scored, 1), 4),
        "made_source": ("box_events"
                        if stage_complete(args.game_dir / "box_events")
                        else "shot_events_raw"),
        **person_block,
        **ball_block,
        **spine_block,
        **box_block,
        "per_event": per,
        "untimed_truth_not_scored": truth_doc.get("untimed_truth", []),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    if args.scorecard is not None:
        from datetime import datetime, timezone

        card = (json.loads(args.scorecard.read_text())
                if args.scorecard.exists() else
                {"rule": "Written ONLY by score_e2e_vs_anchors.py. Any "
                         "comparison or capability claim cites an entry "
                         "here by eval name — unmeasured claims don't "
                         "ship (2026-08-25).",
                 "entries": []})
        card["entries"].append({
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "eval": args.out.name,
            "game_dir": str(args.game_dir),
            "e2e_named": f"{out['e2e_named_hits']}/"
                         f"{out['e2e_named_truth_rows']}",
            "detection_recall": out["detection_recall"],
            "detection_precision": out["detection_precision"],
            "spine_detection_recall": out.get("spine_detection_recall"),
            "spine_detection_precision": out.get("spine_detection_precision"),
            "ball_coverage_confirmed": out.get("ball_coverage_confirmed"),
            "ball_coverage_live": out.get("ball_coverage_live"),
            "ball_longest_gap_ms": out.get("ball_longest_gap_ms"),
            "make_miss_accuracy": out["make_miss_accuracy"],
            "team_accuracy": out.get("team_accuracy"),
            "post_naming_accuracy": out.get("post_naming_accuracy"),
        })
        args.scorecard.write_text(json.dumps(card, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("per_event",)}, indent=2))


if __name__ == "__main__":
    main()
