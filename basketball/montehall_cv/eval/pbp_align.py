"""Align ESPN play-by-play to broadcast video time via the game clock.

Dense clock reads -> period segmentation (clock jumping UP >4 min = new
period) -> (period, clock)->video-time index -> per-play alignment rows.
Writes two resumable artifacts under <out>/: `clock_reads` and
`pbp_alignment` (parquet + _SUCCESS, ArtifactWriter contract).

CLI (one game):
    python -m montehall_cv.eval.pbp_align --video g.mp4 --pbp g_pbp.json --out out/g
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa

from montehall_cv.eval.clock_ocr import ClockReader, iter_clock_crops, parse_clock
from montehall_cv.store.artifacts import SUCCESS_MARKER, ArtifactWriter

CONF_FLOOR = 0.85
MAX_CLOCK_S = 20 * 60
PERIOD_JUMP_S = 240
MATCH_TOLERANCE_S = 8.0
# The game clock never increases within a period; +jitter above this is a
# misread, and a candidate period break must be confirmed by the reads that
# follow it (the game06 false period-3 split: one spiked read re-perioded
# the whole second half).
CLOCK_JITTER_S = 2.0
BREAK_LOOKAHEAD = 3

CLOCK_READS_SCHEMA = pa.schema(
    [
        ("video_t", pa.float64()),
        ("text", pa.string()),
        ("conf", pa.float32()),
        ("clock_s", pa.float64()),
        ("period", pa.int32()),
    ]
)

ALIGNMENT_SCHEMA = pa.schema(
    [
        ("play_id", pa.string()),
        ("sequence", pa.int64()),
        ("period", pa.int32()),
        ("clock_s", pa.float64()),
        ("video_t", pa.float64()),
        ("type_text", pa.string()),
        ("text", pa.string()),
        ("shooting_play", pa.bool_()),
        ("scoring_play", pa.bool_()),
        ("score_value", pa.int32()),
        ("athlete_ids", pa.list_(pa.string())),
        ("shooter_jersey", pa.string()),
        ("coord_x", pa.float64()),
        ("coord_y", pa.float64()),
        ("align_gap_s", pa.float64()),
    ]
)


def build_timeline(reads) -> list[tuple[int, float, float]]:
    """Confident reads -> [(period, clock_s, video_t)] with period breaks.

    Monotonic-within-period validation: the clock only counts down inside a
    period, so a read that jumps UP is either a period break or a misread.
    A break is accepted only when the majority of the next BREAK_LOOKAHEAD
    confident reads are also in the new-period regime; an unconfirmed spike
    is dropped. Smaller in-period increases (> CLOCK_JITTER_S) are dropped
    as misreads.
    """
    confident = [
        r for r in reads
        if r.conf >= CONF_FLOOR and r.clock_s is not None and r.clock_s <= MAX_CLOCK_S
    ]

    def _sustained(i: int, floor: float, margin: float) -> bool:
        """Majority of the next BREAK_LOOKAHEAD confident reads exceed floor+margin."""
        followers = confident[i + 1 : i + 1 + BREAK_LOOKAHEAD]
        if not followers:
            return False
        confirming = sum(1 for f in followers if f.clock_s - floor > margin)
        return confirming * 2 >= len(followers)

    timeline: list[tuple[int, float, float]] = []
    period, prev = 1, None
    for i, r in enumerate(confident):
        if prev is not None:
            delta = r.clock_s - prev
            if delta > PERIOD_JUMP_S:
                if not _sustained(i, prev, PERIOD_JUMP_S):
                    continue  # unconfirmed spike: misread, not a period break
                period += 1
            elif delta > CLOCK_JITTER_S and not _sustained(i, prev, CLOCK_JITTER_S):
                continue  # isolated in-period increase: clock never goes up
                # (a sustained sub-jump increase is a scorer correction or an
                # unmarked restart — kept, same period, matching old behavior)
        prev = r.clock_s
        timeline.append((period, r.clock_s, r.video_t))
    return timeline


def video_time(
    timeline_by_period: dict[int, list[tuple[float, float]]], period: int, clock_s: float
) -> tuple[float, float] | None:
    """Nearest confident read in the period; None past MATCH_TOLERANCE_S."""
    entries = timeline_by_period.get(period)
    if not entries:
        return None
    best_t, best_gap = None, float("inf")
    for eck, t in entries:
        gap = abs(eck - clock_s)
        if gap < best_gap:
            best_t, best_gap = t, gap
    if best_t is None or best_gap > MATCH_TOLERANCE_S:
        return None
    return best_t, best_gap


def jersey_map(summary: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for team in summary.get("boxscore", {}).get("players") or []:
        for grp in team.get("statistics", []):
            for a in grp.get("athletes", []):
                athlete = a.get("athlete") or {}
                jersey = a.get("jersey") or athlete.get("jersey")
                if athlete.get("id") and jersey:
                    out[str(athlete["id"])] = str(jersey)
    return out


def align_game(video: Path, pbp_json: Path, out_dir: Path, device: str = "cpu") -> dict:
    reads_dir = out_dir / "clock_reads"
    align_dir = out_dir / "pbp_alignment"
    if (align_dir / SUCCESS_MARKER).exists():
        return json.loads((out_dir / "alignment_report.json").read_text())

    reader = ClockReader(device=device)
    crops: list = []
    times: list[float] = []
    reads = []
    for crop, t in iter_clock_crops(str(video)):
        crops.append(crop)
        times.append(t)
        if len(crops) >= 256:
            reads.extend(reader.read(crops, times))
            crops, times = [], []
    if crops:
        reads.extend(reader.read(crops, times))

    timeline = build_timeline(reads)
    period_of: dict[float, int] = {t: p for p, _, t in timeline}
    reads_writer = ArtifactWriter(reads_dir, CLOCK_READS_SCHEMA)
    reads_writer.add_many(
        {
            "video_t": r.video_t,
            "text": r.text,
            "conf": r.conf,
            "clock_s": r.clock_s,
            "period": period_of.get(r.video_t),
        }
        for r in reads
    )
    reads_writer.close()
    (reads_dir / SUCCESS_MARKER).touch()

    by_period: dict[int, list[tuple[float, float]]] = {}
    for p, ck, t in timeline:
        by_period.setdefault(p, []).append((ck, t))

    summary = json.loads(pbp_json.read_text())
    jerseys = jersey_map(summary)
    plays = summary.get("plays") or []
    writer = ArtifactWriter(align_dir, ALIGNMENT_SCHEMA)
    aligned = shots = shots_with_athlete = 0
    for i, play in enumerate(plays):
        clock_s = parse_clock((play.get("clock") or {}).get("displayValue", ""))
        period = (play.get("period") or {}).get("number")
        if clock_s is None or period is None:
            continue
        match = video_time(by_period, period, clock_s)
        if match is None:
            continue
        vt, gap = match
        aligned += 1
        shooting = bool(play.get("shootingPlay"))
        participants = [
            str((p.get("athlete") or {}).get("id"))
            for p in play.get("participants") or []
            if (p.get("athlete") or {}).get("id")
        ]
        if shooting:
            shots += 1
            if participants:
                shots_with_athlete += 1
        coord = play.get("coordinate") or {}
        writer.add(
            {
                "play_id": str(play.get("id", i)),
                "sequence": i,
                "period": period,
                "clock_s": clock_s,
                "video_t": vt,
                "type_text": (play.get("type") or {}).get("text", ""),
                "text": play.get("text", ""),
                "shooting_play": shooting,
                "scoring_play": bool(play.get("scoringPlay")),
                "score_value": int(play.get("scoreValue") or 0),
                "athlete_ids": participants,
                "shooter_jersey": jerseys.get(participants[0]) if participants else None,
                "coord_x": float(coord["x"]) if "x" in coord else None,
                "coord_y": float(coord["y"]) if "y" in coord else None,
                "align_gap_s": gap,
            }
        )
    writer.close()
    (align_dir / SUCCESS_MARKER).touch()

    report = {
        "video": str(video),
        "plays_total": len(plays),
        "plays_aligned": aligned,
        "shot_events_aligned": shots,
        "shots_with_athlete": shots_with_athlete,
        "confident_reads": len(timeline),
        "reads_total": len(reads),
        "periods": sorted({p for p, _, _ in timeline}),
        "rostered_athletes": len(jerseys),
    }
    (out_dir / "alignment_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--pbp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    report = align_game(args.video, args.pbp, args.out, device=args.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
