"""Announcer name-calls: whisper transcription -> exact roster-surname hits.

The densest identity anchor in broadcast film — the commentary names the
event actor within seconds of nearly every play. Output is a `name_calls/`
stage (NAME_CALLS_SCHEMA + _SUCCESS) inside the game's align dir; the
tokenizer merges it as the `name_call` channel.

Matching is EXACT lowercase surname token match against the ESPN summary
boxscore roster (word boundary; surnames under 3 chars skipped). Shared
surnames emit one row with ambiguous=True and every candidate athlete —
never a guess.

CLI (one game):
    python -m montehall_cv.brain.name_calls \
        --video pairing/videos/G.mp4 --pbp pairing/pbp/G.pbp.json \
        --out align/eid_<id>

Batch (every aligned eid with a resolvable video, existence-skip):
    python -m montehall_cv.brain.name_calls --align-root /work/align \
        --videos /work/pairing/videos --pbp-dir /work/pairing/pbp
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

from montehall_cv.store.artifacts import ArtifactWriter, stage_complete
from montehall_cv.store.schemas import NAME_CALLS_SCHEMA

MIN_SURNAME_LEN = 3
WHISPER_MODEL = "medium.en"
GPU_FREE_MB = 4096  # headroom floor before choosing cuda over cpu
BACKENDS = ("auto", "torch", "faster")


def roster_surnames(summary: dict) -> dict[str, list[dict]]:
    """lowercase surname -> [{athlete_id, jersey, name}] from the boxscore."""
    out: dict[str, list[dict]] = {}
    for team in (summary.get("boxscore") or {}).get("players") or []:
        for grp in team.get("statistics") or []:
            for a in grp.get("athletes") or []:
                athlete = a.get("athlete") or {}
                name = (athlete.get("displayName") or athlete.get("shortName")
                        or "")
                if not athlete.get("id") or not name:
                    continue
                surname = name.split()[-1].lower()
                if len(surname) < MIN_SURNAME_LEN:
                    continue
                entry = {
                    "athlete_id": str(athlete["id"]),
                    "jersey": str(a.get("jersey") or athlete.get("jersey") or ""),
                    "name": name,
                }
                bucket = out.setdefault(surname, [])
                if all(e["athlete_id"] != entry["athlete_id"] for e in bucket):
                    bucket.append(entry)
    return out


def match_segment(text: str, surnames: dict[str, list[dict]]) -> list[tuple[str, list[dict]]]:
    """Exact word-boundary surname hits in one transcript segment."""
    words = set(re.findall(r"[a-z']+", text.lower()))
    return [(s, surnames[s]) for s in sorted(surnames) if s in words]


def _gpu_free_mb() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    return 0


def _clamp_conf(avg_logprob: float) -> float:
    return max(0.0, min(1.0, math.exp(avg_logprob)))


def _segments_torch(video: Path, model_name: str) -> list[tuple[float, str, float]]:
    """openai-whisper on the container's torch CUDA. EAGER: the full
    transcript exists before the first row is returned, so a failure here
    never leaves partial stage output behind."""
    import torch
    import whisper

    model = whisper.load_model(model_name, device="cuda")
    try:
        result = model.transcribe(str(video), language="en", fp16=True,
                                  condition_on_previous_text=False)
    finally:
        del model
        torch.cuda.empty_cache()
    return [
        (float(seg["start"]), seg["text"],
         _clamp_conf(float(seg.get("avg_logprob", -10.0))))
        for seg in result["segments"]
    ]


def _segments_faster(video: Path, model_name: str) -> tuple[list, str]:
    """faster-whisper fallback (CPU on platforms without ctranslate2 CUDA)."""
    from faster_whisper import WhisperModel

    device, compute_type = ("cuda", "int8_float16") \
        if _gpu_free_mb() >= GPU_FREE_MB else ("cpu", "int8")
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception:
        device, compute_type = "cpu", "int8"
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        str(video), vad_filter=True, beam_size=1, language="en",
    )
    rows = [(float(s.start), s.text, _clamp_conf(s.avg_logprob))
            for s in segments]
    return rows, f"faster-{device}"


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _transcribe(video: Path, model_name: str,
                backend: str) -> tuple[list[tuple[float, str, float]], str]:
    # Gate on torch's own cuda probe, never on nvidia-smi memory.free —
    # unified-memory boxes (GB10) report it as N/A. _segments_torch is
    # eager, so a cuda failure falls through with zero partial output.
    if backend in ("auto", "torch"):
        if backend == "torch" or _cuda_available():
            try:
                return _segments_torch(video, model_name), "torch-cuda"
            except Exception:
                if backend == "torch":
                    raise
    rows, label = _segments_faster(video, model_name)
    return rows, label


def extract_name_calls(video: Path, pbp_json: Path, out_dir: Path,
                       game_key: str | None = None,
                       model_name: str = WHISPER_MODEL,
                       backend: str = "auto") -> dict:
    stage = Path(out_dir) / "name_calls"
    report_path = Path(out_dir) / "name_calls_report.json"
    if stage_complete(stage):
        if report_path.exists():
            return json.loads(report_path.read_text())
        return {"skipped": True}

    key = game_key or Path(out_dir).name
    surnames = roster_surnames(json.loads(Path(pbp_json).read_text()))
    rows, device_label = _transcribe(Path(video), model_name, backend)

    writer = ArtifactWriter(stage, NAME_CALLS_SCHEMA)
    n_hits = 0
    for start_s, text, conf in rows:
        for surname, entries in match_segment(text, surnames):
            n_hits += 1
            writer.add({
                "game_key": key,
                "t_ms": round(start_s * 1000),
                "surname": surname,
                "athlete_ids": [e["athlete_id"] for e in entries],
                "jerseys": [e["jersey"] for e in entries],
                "ambiguous": len(entries) > 1,
                "conf": conf,
                "segment_text": text.strip()[:200],
            })
    writer.close()  # empty is a valid, honest result (quiet feed)

    report = {
        "game_key": key, "device": device_label, "model": model_name,
        "segments": len(rows), "name_calls": n_hits,
        "roster_surnames": len(surnames),
    }
    report_path.write_text(json.dumps(report, indent=2))
    return report


def run_all(align_root: Path, videos_dir: Path, pbp_dir: Path,
            model_name: str = WHISPER_MODEL, backend: str = "auto") -> dict:
    """Every aligned eid with a resolvable video; per-game persist + skip."""
    done = skipped = failed = 0
    for game_dir in sorted(Path(align_root).glob("eid_*")):
        if stage_complete(game_dir / "name_calls"):
            skipped += 1
            continue
        name_file = game_dir / "GAME_NAME"
        if not name_file.exists():
            continue
        stem = name_file.read_text().strip()
        video = Path(videos_dir) / f"{stem}.mp4"
        pbp = Path(pbp_dir) / f"{stem}.pbp.json"
        if not video.exists() or not pbp.exists():
            continue
        try:
            report = extract_name_calls(video, pbp, game_dir,
                                        model_name=model_name, backend=backend)
            done += 1
            print(f"NAMECALLS: {game_dir.name} hits={report.get('name_calls')} "
                  f"device={report.get('device')}", flush=True)
        except Exception as exc:  # one bad game never kills the sweep
            failed += 1
            print(f"NAMECALLS: {game_dir.name} FAILED: {str(exc)[:200]}",
                  flush=True)
    summary = {"extracted": done, "skipped": skipped, "failed": failed}
    print(f"NAMECALLS_ALL_DONE {json.dumps(summary)}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--pbp", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--align-root", type=Path)
    ap.add_argument("--videos", type=Path)
    ap.add_argument("--pbp-dir", type=Path)
    ap.add_argument("--model", default=WHISPER_MODEL)
    ap.add_argument("--backend", default="auto", choices=BACKENDS)
    args = ap.parse_args()
    if args.align_root:
        if not (args.videos and args.pbp_dir):
            ap.error("--align-root needs --videos and --pbp-dir")
        run_all(args.align_root, args.videos, args.pbp_dir, args.model,
                backend=args.backend)
    elif args.video and args.pbp and args.out:
        print(json.dumps(
            extract_name_calls(args.video, args.pbp, args.out,
                               model_name=args.model, backend=args.backend),
            indent=2))
    else:
        ap.error("pass --video/--pbp/--out or --align-root/--videos/--pbp-dir")


if __name__ == "__main__":
    main()
