"""Lossless SkillCorner V3 public-data ingestion and provider-event summaries.

Tracking and Dynamic Events are provider/model-derived, never human ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

import pandas as pd

from soccerviz.core.assets import sha256
from soccerviz.core.data import write_json

COMMIT = "c1e17a0cc3e07e1774b52d929c1a0b85115143fc"
REPOSITORY = "https://github.com/SkillCorner/opendata"
RAW = f"https://raw.githubusercontent.com/SkillCorner/opendata/{COMMIT}/"
MEDIA = f"https://media.githubusercontent.com/media/SkillCorner/opendata/{COMMIT}/"
PROVENANCE = "provider_model_derived"


def _fetch(url: str, path: Path, *, max_bytes: int, max_lines: int | None = None) -> dict:
    """Bound both downloads and memory; commit-pinned source bytes remain auditable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_suffix(path.suffix + ".source.json")
    if path.exists() and sidecar.exists():
        record = json.loads(sidecar.read_text())
        if record["url"] != url or record["sha256"] != sha256(path):
            raise ValueError("Cached SkillCorner source provenance mismatch")
        return record
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.partial")
    size = lines = 0
    reached_eof = False
    try:
        with urllib.request.urlopen(url, timeout=45) as response, tmp.open("wb") as out:
            while max_lines is None or lines < max_lines:
                block = response.readline(max_bytes + 1) if max_lines else response.read(65536)
                if not block:
                    reached_eof = True
                    break
                size += len(block)
                if size > max_bytes:
                    raise ValueError(f"Source exceeds download bound of {max_bytes} bytes")
                if max_lines and not block.endswith(b"\n"):
                    # A final JSON line need not end with a newline. JSON validation follows.
                    json.loads(block)
                out.write(block)
                lines += bool(max_lines)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    record = {
        "url": url,
        "upstream_commit": COMMIT,
        "sha256": sha256(path),
        "bytes_downloaded": size,
        "complete_source": reached_eof,
        "complete_json_lines": lines if max_lines else None,
    }
    write_json(sidecar, record)
    return record


def fetch_match(match_id: int, cache: Path, *, max_frames: int = 600) -> dict:
    if not isinstance(max_frames, int) or not 1 <= max_frames <= 100000:
        raise ValueError("max_frames must be an integer in [1,100000]")
    cache = Path(cache) / COMMIT
    sources = {}
    matches = cache / "matches.json"
    sources["matches"] = _fetch(RAW + "data/matches.json", matches, max_bytes=1_000_000)
    catalog = json.loads(matches.read_text())
    if match_id not in {m["id"] for m in catalog}:
        raise ValueError("Requested match is absent from pinned SkillCorner catalog")
    prefix = f"data/matches/{match_id}/{match_id}"
    directory = cache / str(match_id)
    paths = {"matches": str(matches)}
    for key, suffix in [
        ("metadata", "_match.json"),
        ("events", "_dynamic_events.csv"),
        ("phases", "_phases_of_play.csv"),
    ]:
        path = directory / (str(match_id) + suffix)
        sources[key] = _fetch(RAW + prefix + suffix, path, max_bytes=12_000_000)
        paths[key] = str(path)
    pointer = directory / "tracking-lfs-pointer.txt"
    sources["tracking_pointer"] = _fetch(
        RAW + prefix + "_tracking_extrapolated.jsonl", pointer, max_bytes=1024
    )
    match = re.fullmatch(
        r"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n?",
        pointer.read_text(),
    )
    if not match:
        raise ValueError("Unexpected tracking Git LFS pointer")
    path = directory / f"tracking-first-{max_frames}.jsonl"
    sources["tracking"] = _fetch(
        MEDIA + prefix + "_tracking_extrapolated.jsonl",
        path,
        max_bytes=min(250_000_000, max_frames * 20000),
        max_lines=max_frames,
    )
    sources["tracking"].update(
        {"full_source_lfs_sha256": match[1], "full_source_bytes": int(match[2])}
    )
    if sources["tracking"]["complete_source"] and sha256(path) != match[1]:
        raise ValueError("Complete tracking download disagrees with Git LFS hash")
    paths["tracking"] = str(path)
    manifest = directory / f"source-manifest-first-{max_frames}.json"
    write_json(
        manifest,
        {
            "repository": REPOSITORY,
            "upstream_commit": COMMIT,
            "max_frames": max_frames,
            "sources": sources,
        },
    )
    paths["source_manifest"] = str(manifest)
    return paths


def split_manifest(matches: list[dict]) -> dict:
    """Chronological 60/20/20 match-disjoint development split, not an official split."""
    ordered = sorted(matches, key=lambda x: (x["date_time"], x["id"]))
    if len({m["id"] for m in ordered}) != len(ordered) or len(ordered) < 3:
        raise ValueError("Catalog must contain at least three unique matches")
    train_end = max(1, int(len(ordered) * 0.6))
    val_end = max(train_end + 1, int(len(ordered) * 0.8))
    return {
        "schema": "skillcorner-game-split/v1",
        "official_split": False,
        "policy": "chronological matches, earliest 60% train, next 20% validation, latest 20% test",
        "limitations": [
            "Teams and players can recur across splits",
            "Ten-game development corpus; no generalization claim",
        ],
        "matches": [
            {**m, "split": "train" if i < train_end else "validation" if i < val_end else "test"}
            for i, m in enumerate(ordered)
        ],
    }


def _seconds(value):
    if value is None:
        return None
    parts = str(value).split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Unexpected source timestamp: {value}")
    return sum(float(x) * 60**i for i, x in enumerate(reversed(parts)))


def _finite(value):
    return value is not None and math.isfinite(float(value))


def _active(player: dict, frame_id: int) -> bool:
    periods = (player.get("playing_time") or {}).get("by_period", [])
    return any(p["start_frame"] <= frame_id < p["end_frame"] for p in periods)


def load_tracking(metadata: dict, tracking: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    players = {p["id"]: p for p in metadata["players"]}
    if len(players) != len(metadata["players"]):
        raise ValueError("Duplicate registered player IDs")
    observations, frames = [], []
    previous = -1
    with Path(tracking).open() as source:
        for line in source:
            if not line.strip():
                continue
            frame = json.loads(line)
            fid = frame["frame"]
            if not isinstance(fid, int) or fid <= previous:
                raise ValueError("Tracking frame IDs must be strictly increasing integers")
            previous = fid
            common = {
                "match_id": metadata["id"],
                "frame_id": fid,
                "period": frame["period"],
                "source_timestamp": frame["timestamp"],
                "timestamp_s": _seconds(frame["timestamp"]),
                "source_frame_time_s": fid / 10,
                "source": "skillcorner",
                "provenance": PROVENANCE,
                "coordinate_system": "skillcorner_metres_center_origin_source_orientation",
            }
            possession = frame.get("possession") or {}
            frames.append(
                {
                    **common,
                    "possession_player_id": possession.get("player_id"),
                    "possession_group": possession.get("group"),
                    "image_corners_projection_json": json.dumps(
                        frame.get("image_corners_projection")
                    ),
                }
            )
            by_id = {p["player_id"]: p for p in frame["player_data"]}
            if len(by_id) != len(frame["player_data"]):
                raise ValueError("Duplicate player in source tracking frame")
            if set(by_id) - set(players):
                raise ValueError("Tracking references an unregistered player")
            expected = {pid for pid, player in players.items() if _active(player, fid)}
            entities = [("player", pid, by_id.get(pid)) for pid in sorted(expected | set(by_id))]
            entities.append(("ball", "ball", frame.get("ball_data")))
            for entity, pid, data in entities:
                item = data or {}
                flag = item.get("is_detected")
                if flag is not None and not isinstance(flag, bool):
                    raise ValueError("is_detected must be boolean or null")
                available = _finite(item.get("x")) and _finite(item.get("y"))
                status = (
                    "unavailable"
                    if not available
                    else "detected"
                    if flag is True
                    else "extrapolated"
                    if flag is False
                    else "unknown_detection_status"
                )
                player = players.get(pid, {})
                observations.append(
                    {
                        **common,
                        "provider_entity_id": str(pid),
                        "entity": entity,
                        "team_id": player.get("team_id"),
                        "trackable_object": player.get("trackable_object")
                        if entity == "player"
                        else (metadata.get("ball") or {}).get("trackable_object"),
                        "x_m": item.get("x"),
                        "y_m": item.get("y"),
                        "z_m": item.get("z"),
                        "is_detected": flag,
                        "status": status,
                        "present_in_source": data is not None,
                        "expected_on_pitch": pid in expected if entity == "player" else None,
                    }
                )
    if not frames:
        raise ValueError("No SkillCorner tracking frames")
    return pd.DataFrame(observations).convert_dtypes(), pd.DataFrame(frames).convert_dtypes()


def _events(path: Path, match_id: int, *, phases: bool = False) -> pd.DataFrame:
    # StringDtype retains source identifiers and empty cells without float coercion.
    table = pd.read_csv(path, dtype="string")
    required = {
        "match_id",
        "frame_start",
        "frame_end",
        "period",
        "duration",
        "index" if phases else "event_id",
    }
    if not required <= set(table.columns):
        raise ValueError(f"Missing event columns: {sorted(required - set(table.columns))}")
    if set(table.match_id.dropna()) != {str(match_id)} or table.match_id.isna().any():
        raise ValueError("Event match IDs disagree with metadata")
    key = "index" if phases else "event_id"
    if table[key].isna().any() or table[key].duplicated().any():
        raise ValueError("Event IDs must be unique within a match")
    for column in ("frame_start", "frame_end", "period"):
        table[column] = pd.to_numeric(table[column], errors="raise").astype("Int64")
    table["duration"] = pd.to_numeric(table.duration, errors="raise").astype("Float64")
    if table[["frame_start", "frame_end", "period", "duration"]].isna().any().any():
        raise ValueError("Event intervals and periods cannot be missing")
    if (table.frame_end < table.frame_start).any() or (table.duration < 0).any():
        raise ValueError("Invalid event interval")
    table["event_uid"] = str(match_id) + (":phase:" if phases else ":event:") + table[key]
    table["provenance"] = PROVENANCE
    table["coordinate_system"] = "skillcorner_metres_center_origin_possession_team_left_to_right"
    return table


def import_match(
    metadata: Path,
    tracking: Path,
    events: Path,
    phases: Path,
    matches: Path,
    out: Path,
    source_manifest: Path | None = None,
) -> dict:
    out = Path(out)
    if out.exists():
        raise ValueError(f"Output already exists: {out}")
    meta = json.loads(Path(metadata).read_text())
    if not all(_finite(meta.get(k)) and meta[k] > 0 for k in ("pitch_length", "pitch_width")):
        raise ValueError("Actual pitch dimensions are required")
    splits = split_manifest(json.loads(Path(matches).read_text()))
    selected = [m for m in splits["matches"] if m["id"] == meta["id"]]
    if len(selected) != 1:
        raise ValueError("Match metadata is absent from split catalog")
    paths = {
        "metadata": Path(metadata),
        "tracking": Path(tracking),
        "events": Path(events),
        "phases": Path(phases),
        "matches": Path(matches),
    }
    hashes = {k: sha256(v) for k, v in paths.items()}
    upstream = json.loads(Path(source_manifest).read_text()) if source_manifest else None
    if upstream:
        for key, digest in hashes.items():
            if upstream["sources"][key]["sha256"] != digest:
                raise ValueError(f"Source hash mismatch: {key}")
    obs, frames = load_tracking(meta, Path(tracking))
    event_table = _events(Path(events), meta["id"])
    phase_table = _events(Path(phases), meta["id"], phases=True)
    covered = set(frames.frame_id)
    for table in (event_table, phase_table):
        table["start_tracking_available"] = table.frame_start.isin(covered)
        table["end_tracking_available"] = table.frame_end.isin(covered)
        table["tracking_interval_fully_available"] = [
            all(fid in covered for fid in range(int(a), int(b) + 1))
            for a, b in zip(table.frame_start, table.frame_end, strict=True)
        ]
    event_counts = (
        {str(k): int(v) for k, v in event_table.event_type.value_counts().items()}
        if "event_type" in event_table
        else {}
    )
    offball = (
        event_table[event_table.event_type == "off_ball_run"]
        if "event_type" in event_table
        else event_table.iloc[:0]
    )
    subtype_counts = (
        {str(k): int(v) for k, v in offball.event_subtype.value_counts().items()}
        if "event_subtype" in offball
        else {}
    )
    phase_seconds = {}
    if "team_in_possession_phase_type" in phase_table:
        phase_seconds = {
            str(k): round(float(v), 3)
            for k, v in phase_table.groupby("team_in_possession_phase_type").duration.sum().items()
        }
    report = {
        "schema": "skillcorner-ingestion/v1",
        "match_id": meta["id"],
        "source": REPOSITORY,
        "provenance": PROVENANCE,
        "human_ground_truth": False,
        "loader": "native SkillCorner V3 JSONL and complete CSV columns",
        "source_sha256": hashes,
        "upstream": upstream,
        "game_metadata": selected[0],
        "split": selected[0]["split"],
        "pitch_length_m": meta["pitch_length"],
        "pitch_width_m": meta["pitch_width"],
        "home_team_side_by_period": meta.get("home_team_side"),
        "coordinate_policy": "Tracking retains centered source axes in actual metres. Dynamic events assume the possession team attacks left-to-right, except source attacking_side fields; no cross-table position join or 105x68 rescaling.",
        "clock_policy": "Exact source timestamp and period retained, including nulls. frame/10 is a separate source frame clock, never substituted for match time. Event frame references are preserved.",
        "frames": len(frames),
        "frame_range": [int(frames.frame_id.min()), int(frames.frame_id.max())],
        "observation_rows": len(obs),
        "registered_players": len(meta["players"]),
        "observation_status_counts": dict(Counter(obs.status)),
        "null_timestamp_frames": int(frames.timestamp_s.isna().sum()),
        "frames_with_player_positions": int(
            obs[(obs.entity == "player") & (obs.status != "unavailable")].frame_id.nunique()
        ),
        "events": len(event_table),
        "phases": len(phase_table),
        "events_with_complete_tracking_interval": int(
            event_table.tracking_interval_fully_available.sum()
        ),
        "descriptive_summaries": {
            "event_type_counts": event_counts,
            "off_ball_run_subtype_counts": subtype_counts,
            "provider_in_possession_phase_seconds": phase_seconds,
        },
        "limitations": [
            "Provider tracking identities and coordinates contain errors; this is not perception ground truth.",
            "Detected means the provider reported an on-screen detection; extrapolated coordinates remain explicitly separate.",
            "Dynamic Events and phase labels are upstream derived outputs, not independent tactical validation.",
            "Events may cover the full match while tracking is a bounded prefix; coverage flags prevent silently assuming overlap.",
            "No interpolation, speed estimation, invented phase labels, or coaching-performance claims.",
            "Native parser preserves V3 flags and empty frames; no lossy Kloppy conversion is forced.",
        ],
        "outputs": {
            name: str(out / (name + ".parquet"))
            for name in ("observations", "frames", "events", "phases")
        },
    }
    out.mkdir(parents=True)
    for name, table in [
        ("observations", obs),
        ("frames", frames),
        ("events", event_table),
        ("phases", phase_table),
    ]:
        table.to_parquet(out / f"{name}.parquet", index=False)
    write_json(out / "split-manifest.json", splits)
    write_json(out / "match-metadata.json", meta)
    report["output_sha256"] = {name: sha256(Path(path)) for name, path in report["outputs"].items()}
    report["output_sha256"]["split_manifest"] = sha256(out / "split-manifest.json")
    report["source_verification"] = (
        "pinned_source_manifest_hashes_verified"
        if upstream
        else "local_files_hashed_without_upstream_manifest"
    )
    report["event_missing_values"] = {
        c: int(event_table[c].isna().sum())
        for c in ("x_start", "y_start", "x_end", "y_end", "is_player_possession_end_matched")
        if c in event_table
    }
    write_json(out / "report.json", report)
    return report


def run(request: dict) -> dict:
    if request.get("schema_version", 1) != 1:
        raise ValueError("Unsupported request schema_version")
    operation = request.get("operation", "import")
    if operation == "fetch-import":
        paths = fetch_match(
            int(request["match_id"]),
            Path(request["cache"]),
            max_frames=request.get("max_frames", 600),
        )
    elif operation == "import":
        paths = {
            key: request[key] for key in ("metadata", "tracking", "events", "phases", "matches")
        }
        if request.get("source_manifest"):
            paths["source_manifest"] = request["source_manifest"]
    else:
        raise ValueError(f"Unknown SkillCorner operation: {operation}")
    return import_match(**{k: Path(v) for k, v in paths.items()}, out=Path(request["out"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.response, run(json.loads(args.request.read_text())))


if __name__ == "__main__":
    main()
