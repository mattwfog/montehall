"""Freeze existing provider imports into the same revisable match-state contract."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from soccerviz.core.assets import sha256
from soccerviz.harness.engine import digest


def import_provider(folder: Path, start_s: float, end_s: float) -> dict:
    import json

    if not all(math.isfinite(t) for t in (start_s, end_s)) or start_s < 0 or end_s <= start_s:
        raise ValueError("Use a finite increasing source-clock interval")
    folder = Path(folder)
    paths = {
        k: folder / v
        for k, v in {"observations": "observations.parquet", "report": "report.json"}.items()
    }
    hashes = {k: sha256(p) for k, p in paths.items()}
    report = json.loads(paths["report"].read_text())
    expected = report.get("output_sha256", {}).get("observations")
    if expected is not None and expected != hashes["observations"]:
        raise ValueError("Provider observations disagree with the ingestion report checksum")
    table = pd.read_parquet(paths["observations"])
    skillcorner = report.get("schema") == "skillcorner-ingestion/v1"
    if skillcorner:
        clock = "source_frame_time_s"
        length, width = report["pitch_length_m"], report["pitch_width_m"]
        coordinate = "skillcorner_metres_center_origin_source_orientation"
        offset = [length / 2, width / 2]
    elif report.get("provider") == "Metrica CSV" and report.get("loader") == "kloppy":
        clock, length, width = "timestamp_s", 105, 68
        coordinate, offset = "metres_top_left_105x68", [0, 0]
    else:
        raise ValueError("Expected an existing Metrica/Kloppy or SkillCorner import")
    if not all(math.isfinite(v) and v > 0 for v in (length, width)):
        raise ValueError("Invalid provider pitch dimensions")
    if set(table.coordinate_system.dropna()) != {coordinate}:
        raise ValueError("Provider coordinate declaration disagrees with observations")
    if not np.isfinite(table[clock]).all():
        raise ValueError("Provider source clock is unavailable")
    teams = sorted({str(v) for v in table.team_id.dropna()})
    if len(teams) != 2:
        raise ValueError("Expected exactly two provider teams")
    team_map = {team: i for i, team in enumerate(teams)}
    pairs = sorted(
        {
            (str(r.team_id), str(r.provider_entity_id))
            for r in table.itertuples()
            if r.entity == "player"
        }
    )
    tracks = {pair: i + 1 for i, pair in enumerate(pairs)}
    selected = table[(table[clock] >= start_s) & (table[clock] < end_s)].copy()
    if selected.empty:
        raise ValueError("No provider frames in interval")
    if selected.duplicated(["frame_id", "team_id", "entity", "provider_entity_id"]).any():
        raise ValueError("Duplicate provider entity in one frame")
    frames, rows = [], []
    last_time = None
    source_id = digest(
        {"provider": report.get("provider", "SkillCorner"), "sources": report["source_sha256"]}
    )
    for fid, group in selected.groupby("frame_id", sort=True):
        times = group[clock].unique()
        periods = group.period.fillna(-1).unique()
        if len(times) != 1 or len(periods) != 1:
            raise ValueError("Provider rows disagree on frame clock or period")
        time = float(times[0])
        if last_time is not None and time <= last_time:
            raise ValueError("Provider source timestamps must increase")
        last_time = time
        period = None if periods[0] == -1 else int(periods[0])
        frames.append(
            {"frame_id": int(fid), "timestamp_s": time, "shot_id": period or 0, "period": period}
        )
        for raw in json.loads(group.to_json(orient="records", double_precision=15)):
            kind, status = raw["entity"], raw["status"]
            if kind not in {"player", "ball"}:
                raise ValueError("Unsupported provider entity class")
            if status not in {
                "observed",
                "detected",
                "extrapolated",
                "unavailable",
                "unknown_detection_status",
            }:
                raise ValueError("Unsupported provider observation status")
            xy = [raw["x_m"], raw["y_m"]]
            if any(v is None for v in xy):
                if xy != [None, None] or status != "unavailable":
                    raise ValueError("Provider missingness and coordinates disagree")
                point = None
            elif not all(math.isfinite(v) for v in xy) or status == "unavailable":
                raise ValueError("Invalid provider coordinates")
            else:
                point = [xy[0] + offset[0], xy[1] + offset[1]]
            team_id = str(raw["team_id"]) if raw["team_id"] is not None else None
            track = tracks.get((team_id, str(raw["provider_entity_id"])))
            # JSON floats can stringify integer team IDs with .0; pandas/source IDs
            # are normalized through the original values above, never guessed.
            if kind == "player" and (team_id not in team_map or track is None):
                raise ValueError("Provider team/entity mapping is inconsistent")
            evidence_id = f"provider:{int(fid)}:{kind}:{track or 'ball'}"
            rows.append(
                {
                    "frame_id": int(fid),
                    "timestamp_s": time,
                    "evidence_id": evidence_id,
                    "entity": kind,
                    "provider_entity_id": str(raw["provider_entity_id"]),
                    "tracklet_id": track,
                    "team": team_map.get(team_id),
                    "xy_m": point,
                    "source_xy_m": xy,
                    "provider_status": status,
                    "position_status": "unavailable"
                    if point is None
                    else (
                        "estimated"
                        if status == "extrapolated"
                        else ("unknown" if status == "unknown_detection_status" else "observed")
                    ),
                    "match_timestamp_s": raw.get("timestamp_s"),
                    "provenance": raw.get("provenance", "provider_tracking"),
                }
            )
    if hashes != {k: sha256(p) for k, p in paths.items()}:
        raise ValueError("Provider artifacts changed during import")
    steps = np.diff([f["timestamp_s"] for f in frames])
    return {
        "schema": "provider-input/v1",
        "source_kind": "provider",
        "source_sha256": source_id,
        "artifact_sha256": hashes,
        "source_folder": str(folder.resolve()),
        "source_report": report,
        "window": {"start_s": start_s, "end_s": end_s},
        "sample_period_s": float(np.median(steps)) if len(steps) else 0.2,
        "clock": "provider frame clock (frame/10); match timestamp retained separately"
        if skillcorner
        else "exact provider CSV timestamp seconds",
        "pitch": {
            "length_m": length,
            "width_m": width,
            "coordinate_system": "metres, source positive axes, translated to zero origin",
            "source_coordinate_system": coordinate,
            "translation_m": offset,
            "dimensions_status": "provider_reported" if skillcorner else "adapter_assumption",
        },
        "team_map": team_map,
        "frames": frames,
        "observations": rows,
    }


def provider_evidence(source, parents, reviews):
    if source.get("schema") != "provider-input/v1":
        raise ValueError("Expected provider-input/v1")
    return {
        "schema": "provider-evidence/v1",
        "source_kind": "provider",
        "source_sha256": source["source_sha256"],
        "window": source["window"],
        "pitch": source["pitch"],
        "clock": source["clock"],
        "frames": source["frames"],
        "observations": source["observations"],
        "coverage": {
            s: sum(r["position_status"] == s for r in source["observations"])
            for s in ("observed", "estimated", "unknown", "unavailable")
        },
    }


def provider_state(source, parents, reviews):
    from soccerviz.vision.specialists import alternatives, applicable

    evidence = parents["evidence"]
    frames = []
    for frame in evidence["frames"]:
        fid, time = frame["frame_id"], frame["timestamp_s"]
        players, balls = [], []
        for row in evidence["observations"]:
            if row["frame_id"] != fid:
                continue
            common = {
                "evidence_id": row["evidence_id"],
                "xy_m": row["xy_m"],
                "position_sigma_m": None,
                "position_status": row["position_status"],
                "position_origin": "provider_tracking",
                "provider_status": row["provider_status"],
            }
            if row["entity"] == "ball":
                if row["xy_m"] is not None:
                    balls.append({**common, "detector_confidence": None})
                continue
            track = row["tracklet_id"]
            team = alternatives(
                [
                    {
                        "value": row["team"],
                        "source": "provider_team",
                        "evidence_ids": [row["evidence_id"]],
                    }
                ],
                applicable(reviews, time, "team", tracklet_id=track),
            )
            identity = alternatives(
                [{"value": None, "source": "anonymous_provider_entity"}],
                applicable(reviews, time, "identity", tracklet_id=track),
            )
            players.append(
                {
                    **common,
                    "tracklet_id": track,
                    "entity_id": f"{source['source_sha256']}:provider:{track}",
                    "provider_entity_id": row["provider_entity_id"],
                    "team": team[-1]["value"],
                    "team_hypotheses": team,
                    "identity": identity[-1]["value"],
                    "identity_hypotheses": identity,
                }
            )
        candidates = []
        for ball in balls:
            if ball["position_status"] != "observed":
                continue
            for team in (0, 1):
                visible = [
                    p
                    for p in players
                    if p["team"] == team
                    and p["position_status"] == "observed"
                    and p["xy_m"] is not None
                ]
                if visible:
                    near = min(visible, key=lambda p: math.dist(p["xy_m"], ball["xy_m"]))
                    candidates.append(
                        {
                            "value": team,
                            "source": "nearest_player_proxy",
                            "distance_m": math.dist(near["xy_m"], ball["xy_m"]),
                            "evidence_ids": [ball["evidence_id"], near["evidence_id"]],
                        }
                    )
        ranked = sorted(candidates, key=lambda r: r["distance_m"])
        selected = (
            ranked[0]["value"]
            if (
                len(balls) == 1
                and len(ranked) == 2
                and ranked[0]["distance_m"] < 2.5
                and ranked[1]["distance_m"] - ranked[0]["distance_m"] > 0.75
            )
            else None
        )
        updates = applicable(reviews, time, "possession")
        if updates:
            selected = updates[-1]["review"]["value"]
        orientation = {}
        for team in (0, 1):
            changes = applicable(reviews, time, "orientation", team_cluster=team)
            orientation[str(team)] = {
                "selected": changes[-1]["review"]["value"] if changes else None,
                "hypotheses": alternatives(
                    [{"value": d, "source": "unresolved"} for d in (-1, 1)], changes
                ),
            }
        frames.append(
            {
                **frame,
                "source_kind": "provider",
                "position_origin": "provider_tracking",
                "calibration_usable": True,
                "metric_accuracy": "provider_unvalidated",
                "players": players,
                "ball_hypotheses": balls,
                "orientation": orientation,
                "possession": {
                    "team": selected,
                    "status": "reviewed"
                    if updates
                    else ("hypothesis" if selected is not None else "unknown"),
                    "hypotheses": alternatives(candidates, updates),
                },
            }
        )
    return {
        "schema": "match-state/v1",
        "source_kind": "provider",
        "source_sha256": source["source_sha256"],
        "window": source["window"],
        "pitch": source["pitch"],
        "frames": frames,
        "channels": evidence["coverage"],
        "review_revisions": [r["revision"] for r in reviews],
        "uncertainty": "Provider coordinates are not human ground truth; estimates retain their status",
    }


def validate_provider_review(review, source):
    """Use the same contextual review rules with genuine provider observation IDs."""
    from soccerviz.vision.specialists import validate_review

    detections = [
        {
            "detection_id": r["evidence_id"],
            "timestamp_s": r["timestamp_s"],
            "tracklet_id": r["tracklet_id"],
        }
        for r in source["observations"]
        if r["entity"] == "player"
    ]
    validate_review(
        review,
        {"window": source["window"], "tables": {"detections": detections, "ball_candidates": []}},
    )
