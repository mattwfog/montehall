"""Stage artifacts -> one ordered observation-token stream per game.

Reads whatever primitive stages exist for a game (density tiers are
expected: an align-only eid emits clock/cut/pbp tokens; a full pipeline
job adds player/ball/rim/jersey/score channels; a name_calls stage adds
audio anchors) and writes `tokens/` (ArtifactWriter parts + _SUCCESS)
plus `tokens_meta.json`. Model verdicts never enter the stream
(contract §3).

CLI (one game — align-only eid dir, or a full job dir with its align):
    python -m montehall_cv.brain.tokenize_game --game-dir align/eid_401824880
    python -m montehall_cv.brain.tokenize_game \
        --game-dir out-eval-v3/clemson_duke_acc26 --align-dir align/game02

Batch over every aligned game (existence-skip, resumable):
    python -m montehall_cv.brain.tokenize_game --align-root /work/align
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from montehall_cv.brain.tokens import (
    CH_BALL,
    CH_BALL_CONTROL,
    CH_CLOCK,
    CH_CUT,
    CH_JERSEY_READ,
    CH_NAME_CALL,
    CH_PBP_ANCHOR,
    CH_PLAYER,
    CH_RIM,
    CH_SCORE_DELTA,
    CLOCK_CONF_FLOOR,
    CUT_GAP_S,
    META_NAME,
    PLAYER_GRID_MS,
    RIM_GRID_MS,
    make_token,
    sort_key,
    write_meta,
)
from montehall_cv.store.artifacts import (
    ArtifactWriter,
    read_stage,
    stage_complete,
)
from montehall_cv.store.records import DetClass
from montehall_cv.store.schemas import TOKENS_SCHEMA


_token = make_token


def _grid_sample(rows: list[dict], grid_ms: int, key_fn) -> list[dict]:
    """First row per (key, time-bucket); rows must be ts-sorted."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in rows:
        bucket = (key_fn(r), r["ts_ms"] // grid_ms)
        if bucket in seen:
            continue
        seen.add(bucket)
        out.append(r)
    return out


def _clock_tokens(game_key: str, align_dir: Path) -> list[dict]:
    reads = read_stage(align_dir / "clock_reads").to_pylist()
    confident = sorted(
        (r for r in reads
         if (r.get("conf") or 0) >= CLOCK_CONF_FLOOR and r.get("clock_s") is not None),
        key=lambda r: r["video_t"],
    )
    tokens = [
        _token(
            game_key, round(r["video_t"] * 1000), CH_CLOCK,
            value=float(r["clock_s"]), conf=float(r["conf"]),
            period=int(r["period"]) if r.get("period") is not None else None,
        )
        for r in confident
    ]
    for prev, cur in zip(confident, confident[1:]):
        gap = cur["video_t"] - prev["video_t"]
        if gap > CUT_GAP_S:
            tokens.append(
                _token(game_key, round(prev["video_t"] * 1000), CH_CUT,
                       value=float(gap))
            )
    return tokens


def _pbp_tokens(game_key: str, align_dir: Path) -> list[dict]:
    plays = read_stage(align_dir / "pbp_alignment").to_pylist()
    out = []
    for p in plays:
        payload = {
            "play_id": p.get("play_id"),
            "text": p.get("text"),
            "shooting_play": p.get("shooting_play"),
            "scoring_play": p.get("scoring_play"),
            "athlete_ids": list(p.get("athlete_ids") or []),
            "shooter_jersey": p.get("shooter_jersey"),
            "coord_x": p.get("coord_x"),
            "coord_y": p.get("coord_y"),
            "align_gap_s": p.get("align_gap_s"),
        }
        out.append(
            _token(
                game_key, round(p["video_t"] * 1000), CH_PBP_ANCHOR,
                text=p.get("type_text"),
                value=float(p.get("score_value") or 0),
                period=int(p["period"]) if p.get("period") is not None else None,
                payload_json=json.dumps(payload),
            )
        )
    return out


def _position_tokens(game_key: str, job_dir: Path) -> list[dict]:
    rows = sorted(read_stage(job_dir / "positions").to_pylist(),
                  key=lambda r: r["ts_ms"])
    players = [r for r in rows if r["cls"] == int(DetClass.PERSON)
               and r.get("entity_id") is not None]
    balls = [r for r in rows if r["cls"] == int(DetClass.BALL)]
    out = [
        _token(game_key, r["ts_ms"], CH_PLAYER,
               entity_id=r["entity_id"], team_cluster=r.get("team_cluster"),
               court_x=r["court_x"], court_y=r["court_y"],
               conf=r.get("court_conf"))
        for r in _grid_sample(players, PLAYER_GRID_MS, lambda r: r["entity_id"])
    ]
    out += [
        _token(game_key, r["ts_ms"], CH_BALL,
               court_x=r["court_x"], court_y=r["court_y"],
               conf=r.get("court_conf"))
        for r in _grid_sample(balls, PLAYER_GRID_MS, lambda _r: None)
    ]
    return out


def _ball_control_tokens(game_key: str, job_dir: Path) -> list[dict]:
    rows = sorted(read_stage(job_dir / "ball_controls").to_pylist(),
                  key=lambda r: r["ts_ms"])
    sampled = _grid_sample(rows, PLAYER_GRID_MS, lambda r: r["entity_id"])
    return [
        _token(game_key, r["ts_ms"], CH_BALL_CONTROL,
               entity_id=r["entity_id"], team_cluster=r.get("team_cluster"),
               court_x=r["court_x"], court_y=r["court_y"],
               value=r.get("dist_ft"),
               payload_json=json.dumps({"interpolated": bool(r.get("interpolated"))})
               if r.get("interpolated") is not None else None)
        for r in sampled
    ]


def _rim_tokens(game_key: str, job_dir: Path) -> list[dict]:
    rows = sorted(
        (r for r in read_stage(job_dir / "localized").to_pylist()
         if r["cls"] == int(DetClass.RIM)),
        key=lambda r: r["ts_ms"],
    )
    return [
        _token(game_key, r["ts_ms"], CH_RIM,
               court_x=r.get("court_x"), court_y=r.get("court_y"),
               conf=r.get("conf"))
        for r in _grid_sample(rows, RIM_GRID_MS, lambda _r: None)
    ]


def _frame_time_index(job_dir: Path):
    """track_id -> linear frame->ts interpolator from tracklet spans."""
    spans = {
        r["track_id"]: (r["frame_start"], r["frame_end"],
                        r["ts_start_ms"], r["ts_end_ms"])
        for r in read_stage(job_dir / "tracklets").to_pylist()
    }

    def at(track_id: int, frame_idx: int | None) -> int | None:
        span = spans.get(track_id)
        if span is None:
            return None
        f0, f1, t0, t1 = span
        if frame_idx is None:
            return t0
        if f1 <= f0:
            return t0
        frac = min(max((frame_idx - f0) / (f1 - f0), 0.0), 1.0)
        return round(t0 + frac * (t1 - t0))

    return at


def _jersey_tokens(game_key: str, job_dir: Path) -> list[dict]:
    ts_at = _frame_time_index(job_dir)
    out = []
    for r in read_stage(job_dir / "identity").to_pylist():
        t_ms = ts_at(r["track_id"], r.get("bound_at_frame"))
        if t_ms is None:
            continue
        out.append(
            _token(game_key, t_ms, CH_JERSEY_READ,
                   text=r["candidate"], conf=r.get("prob"),
                   payload_json=json.dumps({
                       "track_id": r["track_id"],
                       "stage": r.get("bound_at_stage"),
                   }))
        )
    return out


def _contact_read_tokens(game_key: str, job_dir: Path) -> list[dict]:
    """Per-track contact-sheet jersey verdicts as jersey_read tokens.

    The identity stage's accepted-read gate passes ~4% of available
    evidence (Cal: 42 identity rows vs 915 non-null contact reads,
    measured 2026-08-01); this widens the channel with the per-tracklet
    VLM verdicts, stage-marked in the payload, conf carried as-is.
    Emitted at track start (frame_idx=None -> ts_start): post-hoc
    analysis semantics — the verdict aggregates the whole tracklet."""
    ts_at = _frame_time_index(job_dir)
    out = []
    for r in read_stage(job_dir / "contact_reads").to_pylist():
        if r.get("number") is None:
            continue
        t_ms = ts_at(r["track_id"], None)
        if t_ms is None:
            continue
        out.append(
            _token(game_key, t_ms, CH_JERSEY_READ,
                   text=r["number"], conf=r.get("confidence"),
                   payload_json=json.dumps({
                       "track_id": r["track_id"],
                       "stage": "contact_sheet",
                       "n_crops": r.get("n_crops"),
                   }))
        )
    return out


def _score_tokens(game_key: str, job_dir: Path) -> list[dict]:
    return [
        _token(game_key, r["ts_ms"], CH_SCORE_DELTA,
               text=r["side"], value=float(r["delta"]),
               conf=r.get("read_conf"),
               payload_json=json.dumps({
                   "before": r.get("before_val"), "after": r.get("after_val"),
               }))
        for r in read_stage(job_dir / "score_events").to_pylist()
    ]


def _name_call_tokens(game_key: str, stage_dir: Path) -> list[dict]:
    return [
        _token(game_key, r["t_ms"], CH_NAME_CALL,
               text=r["surname"], conf=r.get("conf"),
               payload_json=json.dumps({
                   "athlete_ids": list(r.get("athlete_ids") or []),
                   "jerseys": list(r.get("jerseys") or []),
                   "ambiguous": bool(r.get("ambiguous")),
               }))
        for r in read_stage(stage_dir).to_pylist()
    ]


# (stage subdir relative to job dir, channel builder) — perception tier.
_JOB_SOURCES = [
    ("positions", _position_tokens),
    ("ball_controls", _ball_control_tokens),
    ("localized", _rim_tokens),
    ("identity", _jersey_tokens),
    ("contact_reads", _contact_read_tokens),
    ("score_events", _score_tokens),
]


def tokenize_game(game_dir: Path, align_dir: Path | None = None,
                  out_dir: Path | None = None, game_key: str | None = None) -> dict:
    """Build the token stream for one game from whatever stages exist."""
    game_dir = Path(game_dir)
    align_dir = Path(align_dir) if align_dir else None
    out = Path(out_dir) if out_dir else game_dir / "tokens"
    key = game_key or game_dir.name

    if stage_complete(out):
        meta_file = out / META_NAME
        if meta_file.exists():
            return json.loads(meta_file.read_text())
        return {"game_key": key, "skipped": True}

    # An align dir may be the game dir itself (align-only eids) or passed
    # separately (eval jobs keep align under /work/align/<game>).
    effective_align = align_dir or (
        game_dir if stage_complete(game_dir / "pbp_alignment") else None
    )

    tokens: list[dict] = []
    sources: list[str] = []
    if effective_align is not None:
        if stage_complete(effective_align / "clock_reads"):
            tokens += _clock_tokens(key, effective_align)
            sources.append("clock_reads")
        if stage_complete(effective_align / "pbp_alignment"):
            tokens += _pbp_tokens(key, effective_align)
            sources.append("pbp_alignment")

    for subdir, builder in _JOB_SOURCES:
        stage = game_dir / subdir
        if stage_complete(stage):
            # jersey tokens need tracklet spans; skip when absent
            if subdir == "identity" and not stage_complete(game_dir / "tracklets"):
                continue
            tokens += builder(key, game_dir)
            sources.append(subdir)

    if stage_complete(game_dir / "name_calls"):
        tokens += _name_call_tokens(key, game_dir / "name_calls")
        sources.append("name_calls")

    if not tokens:
        raise RuntimeError(
            f"no tokens produced for {game_dir} — refusing to stamp success "
            f"(sources found: {sources or 'none'})"
        )

    tokens.sort(key=sort_key)
    counts: dict[str, int] = {}
    writer = ArtifactWriter(out, TOKENS_SCHEMA)
    for idx, row in enumerate(tokens):
        row["token_idx"] = idx
        counts[row["channel"]] = counts.get(row["channel"], 0) + 1
        writer.add(row)
    writer.close()
    return write_meta(out, key, counts, sources)


def tokenize_all(align_root: Path) -> dict:
    """Every align/eid_* dir, existence-skip; per-game persist."""
    done = skipped = failed = 0
    for game_dir in sorted(Path(align_root).glob("eid_*")):
        if stage_complete(game_dir / "tokens"):
            skipped += 1
            continue
        try:
            meta = tokenize_game(game_dir)
            done += 1
            print(f"TOKENIZE: {game_dir.name} total={meta['total']}", flush=True)
        except (RuntimeError, FileNotFoundError) as exc:
            failed += 1
            print(f"TOKENIZE: {game_dir.name} FAILED: {exc}", flush=True)
    summary = {"tokenized": done, "skipped": skipped, "failed": failed}
    print(f"TOKENIZE_ALL_DONE {json.dumps(summary)}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game-dir", type=Path,
                    help="one game dir (align-only eid dir or full job dir)")
    ap.add_argument("--align-dir", type=Path, default=None,
                    help="alignment dir when it lives outside --game-dir")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--game-key", default=None)
    ap.add_argument("--align-root", type=Path, default=None,
                    help="batch mode: tokenize every eid_* under this root")
    args = ap.parse_args()
    if bool(args.game_dir) == bool(args.align_root):
        ap.error("pass exactly one of --game-dir or --align-root")
    if args.align_root:
        tokenize_all(args.align_root)
    else:
        meta = tokenize_game(args.game_dir, align_dir=args.align_dir,
                             out_dir=args.out, game_key=args.game_key)
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
