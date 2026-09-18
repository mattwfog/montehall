"""Mega-mine driver (Ask 2): pair -> align -> mine every non-holdout game.

One long-running pass over the film corpus, resumable at every grain:

  pairing   pair_videos_local (existing pbp.json = skip)
  align     per-game dir align/eid_<event_id>/ with _SUCCESS (CPU pool)
  mine      per-game shard datasets/broadcast-eid_<event_id>/ with _SUCCESS
            (GPU, strictly one at a time — cvbench allocator rule)

Keys are ESPN event_ids, never glob order (the game06/gameNN numbering
trap). Sealed holdouts (montehall_cv.eval.holdouts — the registry contract)
are never aligned, never mined. Legacy gameNN shards are left untouched and
UNUSED: they carry pre-filter ball labels; every game re-mines fresh under
its eid key with the latent-state filter (mining is 3.5 min/game — clean
labels are worth re-buying).

Drain mode: after an idle pass the driver re-runs pairing (the film grab
appends) and sleeps; it exits when a pass finds no work AND the grab log
contains its ALL_DONE marker (or immediately with --once).

CLI (spark, inside cvbench):
    python -m montehall_cv.training.mine_all \
        --videos /work/pairing/videos --corpus /work/pbp-corpus \
        --pbp /work/pairing/pbp --align-root /work/align \
        --shard-root /work/datasets --weights <detector.pth> \
        --grab-log /host_logs/film-grab-r2.log
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from montehall_cv.eval.espn_pbp import pair_videos_local
from montehall_cv.eval.holdouts import is_sealed

ALIGN_POOL = 4
MINE_RETRIES = 3
IDLE_SLEEP_S = 600


def event_id_of(pbp_file: Path) -> str | None:
    try:
        comp = ((json.loads(pbp_file.read_text()).get("header") or {})
                .get("competitions") or [{}])[0]
        return str(comp.get("id")) if comp.get("id") else None
    except (OSError, json.JSONDecodeError):
        return None


def work_items(videos_dir: Path, pbp_dir: Path, align_root: Path,
               shard_root: Path) -> list[dict]:
    """Every paired, non-holdout game with its per-stage done-ness."""
    items: list[dict] = []
    for pbp_file in sorted(pbp_dir.glob("*.pbp.json")):
        stem = pbp_file.name[: -len(".pbp.json")]
        video = videos_dir / f"{stem}.mp4"
        if not video.exists():
            continue
        eid = event_id_of(pbp_file)
        if eid is None or is_sealed(eid):
            continue
        align_dir = align_root / f"eid_{eid}"
        shard_dir = shard_root / f"broadcast-eid_{eid}"
        items.append({
            "eid": eid, "video": video, "pbp": pbp_file,
            "align_dir": align_dir, "shard_dir": shard_dir,
            "aligned": (align_dir / "pbp_alignment" / "_SUCCESS").exists(),
            "mined": (shard_dir / "_SUCCESS").exists(),
        })
    return items


def _run(cmd: list[str], log_prefix: str) -> bool:
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        tail = (out.stdout + out.stderr)[-400:].replace("\n", " | ")
        print(f"{log_prefix} FAILED rc={out.returncode}: {tail}", flush=True)
        return False
    return True


def align_one(item: dict) -> bool:
    item["align_dir"].mkdir(parents=True, exist_ok=True)
    (item["align_dir"] / "GAME_NAME").write_text(item["video"].stem)
    ok = _run(
        [sys.executable, "-m", "montehall_cv.eval.pbp_align",
         "--video", str(item["video"]), "--pbp", str(item["pbp"]),
         "--out", str(item["align_dir"])],
        f"ALIGN eid_{item['eid']}",
    )
    if ok:
        print(f"MINEALL: aligned eid_{item['eid']}", flush=True)
    return ok


def mine_one(item: dict, weights: Path, max_frames: int) -> bool:
    for attempt in range(1, MINE_RETRIES + 1):
        if _run(
            [sys.executable, "-m", "montehall_cv.training.mine_broadcast",
             "--video", str(item["video"]), "--align-dir", str(item["align_dir"]),
             "--out", str(item["shard_dir"]), "--weights", str(weights),
             "--game-tag", f"eid{item['eid']}", "--max-frames", str(max_frames)],
            f"MINE eid_{item['eid']} (attempt {attempt})",
        ):
            print(f"MINEALL: mined eid_{item['eid']}", flush=True)
            return True
        time.sleep(60)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--pbp", type=Path, required=True)
    ap.add_argument("--align-root", type=Path, required=True)
    ap.add_argument("--shard-root", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--grab-log", type=Path,
                    help="film grab log; ALL_DONE there ends drain mode")
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    passes = 0
    while True:
        passes += 1
        pair_videos_local(args.videos, args.corpus, args.pbp)
        items = work_items(args.videos, args.pbp, args.align_root,
                           args.shard_root)
        to_align = [i for i in items if not i["aligned"]]
        with ThreadPoolExecutor(max_workers=ALIGN_POOL) as pool:
            list(pool.map(align_one, to_align))

        items = work_items(args.videos, args.pbp, args.align_root,
                           args.shard_root)
        to_mine = [i for i in items if i["aligned"] and not i["mined"]]
        mined = failed = 0
        for item in to_mine:  # strictly serial: one GPU job at a time
            if mine_one(item, args.weights, args.max_frames):
                mined += 1
            else:
                failed += 1
        done = sum(1 for i in work_items(args.videos, args.pbp,
                                         args.align_root, args.shard_root)
                   if i["mined"])
        print(f"MINEALL: pass {passes} — {len(items)} paired, {done} mined, "
              f"{mined} new, {failed} failed", flush=True)

        progressed = bool(to_align or mined)
        grab_done = bool(
            args.grab_log and args.grab_log.exists()
            and "ALL_DONE" in args.grab_log.read_text(errors="ignore")
        )
        if args.once or (not progressed and grab_done):
            break
        if not progressed:
            time.sleep(IDLE_SLEEP_S)
    print("MINEALL_DONE", flush=True)


if __name__ == "__main__":
    main()
