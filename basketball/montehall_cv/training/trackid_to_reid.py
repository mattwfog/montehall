"""TrackID3x3 ground truth -> ReID identity-crop dataset.

Walks ground_truth/<Subset>/MOT/**.txt (Outdoor: comma-separated per-video
files; Indoor/Drone: space- or comma-separated per-segment `*_gt.txt`),
resolves each file's video by stem under --videos-root, and crops players
into <out>/<identity>/ folders. Identity = (subset, gt-file stem, track_id):
per-file purity — segments of the same game are NOT merged, because a false
identity merge poisons metric learning worse than fragmentation dilutes it.

CPU-only (decode+crop+resize) — safe to run while the GPU trains.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

VIDEO_EXTS = (".MOV", ".mov", ".mp4", ".MP4", ".avi")
CROP_W, CROP_H = 128, 256
MIN_BOX_PX = 40
SAMPLE_EVERY_FRAMES = 15
MAX_CROPS_PER_IDENTITY = 40


def build(gt_root: Path, videos_root: Path, out_dir: Path) -> dict:
    from PIL import Image

    import av

    started = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, float] = {"gt_files": 0, "videos_missing": 0, "identities": 0, "crops": 0}
    video_index = _index_videos(videos_root)

    for gt_file in sorted(gt_root.glob("*/MOT/**/*.txt")):
        stats["gt_files"] += 1
        stem = gt_file.stem.removesuffix("_gt")
        game_dir = gt_file.parent.name if gt_file.parent.name != "MOT" else None
        video = _resolve_video(video_index, stem, game_dir)
        if video is None:
            stats["videos_missing"] += 1
            continue
        subset = gt_file.relative_to(gt_root).parts[0]
        tracks = _load_mot(gt_file)
        plan = _sample_plan(tracks)
        if not plan:
            continue

        identity_counts: dict[int, int] = defaultdict(int)
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            for frame_idx, frame in enumerate(container.decode(stream), start=1):
                boxes = plan.get(frame_idx)
                if boxes is None:
                    continue
                image = frame.to_ndarray(format="rgb24")
                h, w = image.shape[:2]
                for track_id, (x, y, bw, bh) in boxes:
                    x1, y1 = max(int(x), 0), max(int(y), 0)
                    x2, y2 = min(int(x + bw), w), min(int(y + bh), h)
                    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
                        continue
                    identity = f"{subset}_{stem}_t{track_id}"
                    ident_dir = out_dir / identity
                    if identity_counts[track_id] == 0:
                        ident_dir.mkdir(parents=True, exist_ok=True)
                    crop = Image.fromarray(image[y1:y2, x1:x2]).resize(
                        (CROP_W, CROP_H), Image.BILINEAR
                    )
                    crop.save(ident_dir / f"f{frame_idx:06d}.jpg", quality=88)
                    identity_counts[track_id] += 1
                    stats["crops"] += 1
                plan.pop(frame_idx)
                if not plan:
                    break
        stats["identities"] += sum(1 for c in identity_counts.values() if c > 0)

    stats["wall_min"] = round((time.monotonic() - started) / 60, 1)
    return stats


def _index_videos(videos_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in videos_root.rglob("*"):
        if path.suffix in VIDEO_EXTS and path.stem not in index:
            index[path.stem] = path
    return index


def _resolve_video(index: dict[str, Path], stem: str, game_dir: str | None) -> Path | None:
    """Exact stem, else unique suffix match (Drone splits carry prefixes,
    e.g. GT `40_1215` -> video `1_1_6000_40_1215.mp4`); a game-dir hint
    disambiguates suffix collisions across games."""
    exact = index.get(stem)
    if exact is not None:
        return exact
    candidates = [p for s, p in index.items() if s.endswith(f"_{stem}")]
    if game_dir and len(candidates) > 1:
        candidates = [p for p in candidates if game_dir in p.parts]
    return candidates[0] if len(candidates) == 1 else None


def _load_mot(gt_file: Path) -> dict[int, list[tuple[int, tuple]]]:
    """frame -> [(track_id, (x, y, w, h))]; tolerates comma or whitespace rows."""
    tracks: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
    for line in gt_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",") if "," in line else line.split()
        if len(parts) < 6:
            continue
        try:
            frame, track = int(float(parts[0])), int(float(parts[1]))
            x, y, w, h = (float(v) for v in parts[2:6])
        except ValueError:
            continue
        tracks[frame].append((track, (x, y, w, h)))
    return dict(tracks)


def _sample_plan(
    tracks: dict[int, list[tuple[int, tuple]]],
) -> dict[int, list[tuple[int, tuple]]]:
    """Subsample frames per track: every Nth appearance, capped per identity."""
    per_track_frames: dict[int, list[int]] = defaultdict(list)
    for frame in sorted(tracks):
        for track_id, _ in tracks[frame]:
            per_track_frames[track_id].append(frame)
    wanted: dict[int, set[int]] = defaultdict(set)
    for track_id, frames in per_track_frames.items():
        picks = frames[::SAMPLE_EVERY_FRAMES][:MAX_CROPS_PER_IDENTITY]
        for frame in picks:
            wanted[frame].add(track_id)
    plan: dict[int, list[tuple[int, tuple]]] = {}
    for frame, track_ids in wanted.items():
        rows = [(t, box) for t, box in tracks[frame] if t in track_ids]
        if rows:
            plan[frame] = rows
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True,
                        help="TrackID3x3 repo ground_truth/ dir")
    parser.add_argument("--videos-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.gt_root, args.videos_root, args.out), indent=2))


if __name__ == "__main__":
    main()
