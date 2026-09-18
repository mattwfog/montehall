"""Full-chain driver: video -> box score in one command.

Runs every stage in order, skipping any whose _SUCCESS exists (each stage
is individually resumable). LLM stages (rim, shots, harness) need
ANTHROPIC_API_KEY and are skipped with a warning if it's absent — the
perception stages still run, which is exactly what the multi-video label
harvest needs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def run_all(
    video: Path,
    out_root: Path,
    job_id: str,
    court_weights: Path,
    detector_weights: Path | None = None,
    reid_weights: Path | None = None,
    legibility_weights: Path | None = None,
    ocr_weights: Path | None = None,
    roster_map: Path | None = None,
    roster_job_id: str | None = None,
    attr_roster: Path | None = None,
    wasb_root: Path | None = None,
    wasb_weights: Path | None = None,
    ranker_weights: Path | None = None,
    long_gap: bool = False,
    batch: int = 16,
    render: bool = False,
    render_scale: float = 1.0,
) -> dict:
    from montehall_cv.pipeline import (
        run_boxscore,
        run_court,
        run_events,
        run_extract_fast,
        run_freethrows,
        run_jersey,
        run_plays,
        run_possession,
        run_rim,
        run_shots,
        run_team_assoc,
    )

    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    started = time.monotonic()
    results: dict = {}

    results["extract"] = run_extract_fast.run(
        video, out_root, job_id, batch_size=batch, weights=detector_weights
    )
    results["court"] = run_court.run(video, out_root, job_id, court_weights)
    # jersey BEFORE association: trusted reads exist when the long-gap pass
    # runs, so the jersey veto bites on fresh runs, not just re-runs
    results["jersey"] = run_jersey.run(
        video, out_root, job_id, legibility_weights=legibility_weights,
        ocr_weights=ocr_weights,
    )
    results["team_assoc"] = run_team_assoc.run(
        video, out_root, job_id, reid_weights=reid_weights, long_gap=long_gap
    )
    results["possession"] = run_possession.run(out_root, job_id)
    results["events"] = run_events.run(out_root, job_id)
    if wasb_weights is not None:
        # THE BALL SPINE (ball-first ruling 2026-08-03): continuous fused
        # ball track + hold/release/rim events; shot_attribution reads the
        # release holder as the shooter wherever the spine covers a shot
        from montehall_cv.pipeline import ball_track

        results["ball_track"] = ball_track.run(
            video, out_root, job_id,
            wasb_root=wasb_root, wasb_weights=wasb_weights,
        )
    if have_key:
        # run_shots windows off detector RIM boxes when the fine-tuned
        # detector produced them; the VLM rim-locate only runs as fallback
        if _has_detector_rims(out_root / job_id):
            results["rim"] = {"skipped": True, "reason": "detector rim class present"}
        else:
            results["rim"] = run_rim.run(video, out_root, job_id)
        results["shots"] = run_shots.run(video, out_root, job_id)
        # NO scoreboard stage (design decision 2026-08-25: the board is
        # never part of the algo — made/miss is trajectory > VLM >
        # movement veto, all vision)
        # geometric FT verdicts must exist before attribution: unclassified
        # free throws score as two-point field goals
        results["freethrows"] = run_freethrows.run(out_root, job_id)
        # composed shooter attribution (candidates -> sheets -> team ->
        # launch ranking); boxscore consumes its picks, superseding the
        # event_identity read path (structural fix A, 2026-08-25)
        from montehall_cv.pipeline import shot_attribution

        attr_numbers = None
        if attr_roster is not None:
            attr_numbers = {str(n) for n in json.loads(attr_roster.read_text())}
        results["attribution"] = shot_attribution.run(
            video, out_root, job_id,
            roster_numbers=attr_numbers,
            wasb_root=wasb_root, wasb_weights=wasb_weights,
            ranker_weights=ranker_weights,
        )
        results["boxscore"] = run_boxscore.run(
            out_root, job_id, roster_map=roster_map, roster_job_id=roster_job_id,
            video=video, ocr_weights=ocr_weights,
        )
        from montehall_cv.harness import run_harness

        results["harness"] = run_harness.run(out_root, job_id)

        # Aggregated-view jersey suggestions (C1): one cached VLM read per
        # tracklet contact sheet; surfaced as suggested_number on unnamed
        # rows — a suggestion never takes a stat line (0-fabrication holds)
        from montehall_cv.pipeline import contact_sheet

        results["contact_sheet"] = _run_contact_sheet(
            video, out_root, job_id, roster_map, roster_job_id,
            legibility_weights, contact_sheet,
        )
    else:
        results["skipped_llm_stages"] = {
            "reason": "ANTHROPIC_API_KEY not set",
            "stages": ["rim", "shots", "boxscore", "harness"],
        }

    # After the LLM block: plays join shot evidence when it exists and
    # degrade to trajectories+events (zones null) in harvest mode.
    results["plays"] = run_plays.run(out_root, job_id)

    if render:
        from montehall_cv.pipeline.render import run as run_render

        results["render"] = run_render(video, out_root, job_id, scale=render_scale)

    # Evidence sheets for the coach naming flow (A6) — CPU, needs box_score;
    # skips itself in harvest mode (no box_score without the LLM stages).
    from montehall_cv.pipeline import run_evidence

    results["evidence"] = run_evidence.run(
        video, out_root, job_id, legibility_weights=legibility_weights
    )

    # Overlay keyframe tracks for the in-video naming prompts (B1).
    from montehall_cv.pipeline import run_tracks

    results["tracks"] = run_tracks.run(video, out_root, job_id)

    results["total_wall_seconds"] = round(time.monotonic() - started, 1)
    return results


def _run_contact_sheet(video, out_root, job_id, roster_map, roster_job_id,
                       legibility_weights, contact_sheet) -> dict:
    from montehall_cv.store.artifacts import stage_complete

    if stage_complete(out_root / job_id / "contact_reads"):
        return {"skipped": True, "reason": "stage already complete"}
    try:
        return contact_sheet.run(
            video, out_root, job_id, roster_map=roster_map,
            roster_job_id=roster_job_id,
            legibility_weights=legibility_weights,
        )
    except Exception as exc:  # suggestions are optional — never fail the job
        return {"error": str(exc)[:300]}


def _has_detector_rims(job_dir: Path) -> bool:
    from montehall_cv.store.artifacts import read_stage, stage_complete
    from montehall_cv.store.records import DetClass

    if not stage_complete(job_dir / "localized"):
        return False
    rim = int(DetClass.RIM)
    return any(
        row["cls"] == rim for row in read_stage(job_dir / "localized").to_pylist()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--court-weights", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, default=None)
    parser.add_argument("--reid-weights", type=Path, default=None,
                        help="OSNet checkpoint: metric-learned association gate")
    parser.add_argument("--legibility-weights", type=Path, default=None,
                        help="trained jersey legibility gate")
    parser.add_argument("--ocr-weights", type=Path, default=None,
                        help="fine-tuned PARSeq state_dict (train_parseq)")
    parser.add_argument("--roster-map", type=Path, default=None,
                        help="roster JSON; enables roster-filtered attribution")
    parser.add_argument("--roster-job-id", default=None,
                        help="roster map key when it differs from --job-id")
    parser.add_argument("--attr-roster", type=Path, default=None,
                        help="JSON list of jersey numbers (BOTH teams) as "
                             "shot-attribution read vocabulary")
    parser.add_argument("--wasb-root", type=Path, default=None)
    parser.add_argument("--wasb-weights", type=Path, default=None,
                        help="explicit WASB checkpoint enabling launch-point "
                             "ranking in shot attribution")
    parser.add_argument("--ranker-weights", type=Path, default=None,
                        help="learned shot-attribution ranker JSON")
    parser.add_argument("--long-gap", action="store_true",
                        help="ReID-only merges across motion-unbridgeable gaps")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--render", action="store_true",
                        help="render the annotated overlay video (CPU)")
    parser.add_argument("--render-scale", type=float, default=1.0)
    args = parser.parse_args()
    summary = run_all(
        args.video, args.out, args.job_id, args.court_weights,
        detector_weights=args.detector_weights, reid_weights=args.reid_weights,
        legibility_weights=args.legibility_weights, ocr_weights=args.ocr_weights,
        roster_map=args.roster_map, roster_job_id=args.roster_job_id,
        attr_roster=args.attr_roster, wasb_root=args.wasb_root,
        wasb_weights=args.wasb_weights, ranker_weights=args.ranker_weights,
        long_gap=args.long_gap,
        batch=args.batch,
        render=args.render, render_scale=args.render_scale,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
