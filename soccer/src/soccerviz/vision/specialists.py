"""Video evidence → revisable state → bounded tactical findings → analyst brief."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from soccerviz.core.assets import sha256
from soccerviz.core.geometry import project as project_dependency
from soccerviz.harness.engine import Specialist
from soccerviz.vision import observation_channels
from soccerviz.vision.observation_channels import (
    ball_stage,
    calibration_stage,
    camera_stage,
    compose_state,
    context_stage,
    players_stage,
    video_channels,
)

TABLES = ("frames", "detections", "state", "calibration", "ball_candidates")

# Public review helpers retained for provider adapters and older callers.
applicable = observation_channels.applicable
alternatives = observation_channels.alternatives


def import_video(folder: Path, start_s: float, end_s: float):
    """Freeze a bounded adapter input; never trust a mutable path as a cache identity."""
    if not all(math.isfinite(t) for t in (start_s, end_s)) or start_s < 0 or end_s <= start_s:
        raise ValueError("Use a finite, increasing source-PTS interval")
    paths = {name: folder / f"{name}.parquet" for name in TABLES}
    paths["report"] = folder / "report.json"
    hashes = {name: sha256(path) for name, path in paths.items()}
    report = json.loads(paths["report"].read_text())
    frames = pd.read_parquet(paths["frames"])
    selected = frames[(frames.timestamp_s >= start_s) & (frames.timestamp_s < end_s)]
    if selected.empty:
        raise ValueError("No frames in requested interval")
    ids = set(selected.frame_id)
    tables = {"frames": json.loads(selected.to_json(orient="records", double_precision=15))}
    for name in TABLES[1:]:
        table = pd.read_parquet(paths[name])
        table = table[table.frame_id.isin(ids)]
        tables[name] = json.loads(table.to_json(orient="records", double_precision=15))
    if hashes != {name: sha256(path) for name, path in paths.items()}:
        raise ValueError("Video artifacts changed during import; retry once the run is stable")
    result = {
        "schema": "video-input/v1",
        "window": {"start_s": start_s, "end_s": end_s},
        "source_sha256": report["source_sha256"],
        "model_sha256": report["model_sha256"],
        "artifact_sha256": hashes,
        "source_folder": str(folder.resolve()),
        "tables": tables,
        "accuracy": report.get("accuracy", {}),
        "sample_period_s": 1 / report["sample_hz"],
    }
    validate_input(result)
    return result


def validate_input(value):
    if value.get("schema") != "video-input/v1":
        raise ValueError("Unsupported video input schema")
    frames = value["tables"]["frames"]
    ids = [f["frame_id"] for f in frames]
    times = [f["timestamp_s"] for f in frames]
    start, end = value["window"]["start_s"], value["window"]["end_s"]
    period = value.get("sample_period_s", 0.2)
    if not math.isfinite(period) or period <= 0:
        raise ValueError("Sampling period must be finite and positive")
    if not frames or len(set(ids)) != len(ids) or any(b <= a for a, b in itertools.pairwise(times)):
        raise ValueError("Frames need unique IDs and strictly increasing source PTS")
    if not all(start <= t < end for t in times):
        raise ValueError("Frame outside source-time window")
    clock = dict(zip(ids, times, strict=True))
    for name in TABLES[1:]:
        for row in value["tables"][name]:
            if row["frame_id"] not in clock or row["timestamp_s"] != clock[row["frame_id"]]:
                raise ValueError("Orphan or mismatched observation timestamp")
    detection_ids = [r["detection_id"] for r in value["tables"]["detections"]]
    if len(set(detection_ids)) != len(detection_ids):
        raise ValueError("Duplicate detection evidence ID")
    detection_frames = {r["detection_id"]: r["frame_id"] for r in value["tables"]["detections"]}
    for row in value["tables"]["state"]:
        if (
            row["entity"] != "ball_candidate"
            and detection_frames.get(row["evidence_id"]) != row["frame_id"]
        ):
            raise ValueError("Projected player lacks detection evidence from the same frame")
    for name, key in (
        ("calibration", lambda r: r["frame_id"]),
        ("ball_candidates", lambda r: (r["frame_id"], r["rank"])),
    ):
        keys = [key(row) for row in value["tables"][name]]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate {name} evidence ID")


def validate_review(review, evidence):
    if evidence.get("schema") == "provider-input/v1":
        from soccerviz.harness.provider_colony import validate_provider_review

        return validate_provider_review(review, evidence)
    required = {"field", "value", "start_s", "end_s", "reviewer", "reason", "evidence_ids"}
    allowed = required | {"tracklet_id", "team_cluster"}
    if not required <= review.keys() or review.keys() - allowed:
        raise ValueError("Review fields do not match the correction contract")
    if not all(isinstance(review[k], str) and review[k].strip() for k in ("reviewer", "reason")):
        raise ValueError("A reviewer and reason are required")
    start, end = review["start_s"], review["end_s"]
    window = evidence["window"]
    if not (
        math.isfinite(start)
        and math.isfinite(end)
        and window["start_s"] <= start < end <= window["end_s"]
    ):
        raise ValueError("Correction interval must be inside the imported window")
    detections = evidence["tables"]["detections"]
    known = {r["detection_id"] for r in detections if start <= r["timestamp_s"] < end}
    known |= {
        f"f{r['frame_id']}-ball{r['rank']}"
        for r in evidence["tables"]["ball_candidates"]
        if start <= r["timestamp_s"] < end
    }
    if (
        not isinstance(review["evidence_ids"], list)
        or not review["evidence_ids"]
        or not set(review["evidence_ids"]) <= known
    ):
        raise ValueError("Review must reference evidence in its time interval")
    field, value = review["field"], review["value"]
    if field in {"team", "identity"}:
        tracklet = review.get("tracklet_id")
        if type(tracklet) is not int or not any(
            r["tracklet_id"] == tracklet and start <= r["timestamp_s"] < end for r in detections
        ):
            raise ValueError("Unknown tracklet in correction interval")
        if not any(
            r["tracklet_id"] == tracklet and r["detection_id"] in review["evidence_ids"]
            for r in detections
        ):
            raise ValueError("Tracklet correction needs evidence of the target tracklet")
    if field in {"team", "possession"}:
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ValueError("Team must be anonymous cluster 0, 1, or null")
    elif field == "orientation":
        if (
            type(review.get("team_cluster")) is not int
            or review["team_cluster"] not in (0, 1)
            or (value is not None and (type(value) is not int or value not in (-1, 1)))
        ):
            raise ValueError("Orientation requires a team cluster and direction -1, 1, or null")
    elif field == "identity":
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("Identity must be a nonempty label or null")
    else:
        raise ValueError("Unsupported correction field")


def evidence_stage(source, parents, reviews):
    validate_input(source)
    primitive_columns = (
        "detection_id",
        "frame_id",
        "timestamp_s",
        "role_hypothesis",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "confidence",
        "source",
    )
    return {
        "schema": "evidence/v1",
        "source_sha256": source["source_sha256"],
        "model_sha256": source["model_sha256"],
        "artifact_sha256": source.get("artifact_sha256", {}),
        "window": source["window"],
        "frames": source["tables"]["frames"],
        "detections": [
            {**{k: row[k] for k in primitive_columns}, "tracklet_id": row.get("tracklet_id")}
            for row in source["tables"]["detections"]
        ],
        "ball_candidates": source["tables"]["ball_candidates"],
        "perception_hypotheses": source["tables"]["state"],
        "calibration_hypotheses": source["tables"]["calibration"],
        "metric_accuracy": "unvalidated",
        "clock": "source PTS seconds",
    }


def state_stage(source, parents, reviews):
    """Compose explicit observation organs; retain the legacy evidence-only call."""
    if "camera" not in parents:
        parents = video_channels(source, parents["evidence"], reviews)
    return compose_state(source, parents, reviews)


def tactics_stage(source, parents, reviews):
    state = parents["state"]
    findings, segments = [], []
    current = None
    for frame in state["frames"]:
        team = frame["possession"]["team"]
        time, fid = frame["timestamp_s"], frame["frame_id"]
        if team is None:
            current = None
            continue
        # A missing/ambiguous frame or cut breaks a candidate possession; never bridge it silently.
        if (
            current is None
            or current["team"] != team
            or current["shot_id"] != frame["shot_id"]
            or time - current["last_observed_s"] > 1.5 * source.get("sample_period_s", 0.2)
        ):
            current = {
                "team": team,
                "shot_id": frame["shot_id"],
                "start_s": time,
                "last_observed_s": time,
                "frame_ids": [],
                "status": "hypothesis",
            }
            segments.append(current)
        current["last_observed_s"] = time
        current["frame_ids"].append(fid)
        players = [p for p in frame["players"] if p["team"] == team and p["xy_m"] is not None]
        requests = []
        if not frame["calibration_usable"]:
            requests.append("Review pitch calibration")
        if len(players) < 7:
            requests.append("Insufficient visible teammates: inspect another frame or camera")
        if frame["orientation"][str(team)]["selected"] is None:
            requests.append("Review attacking direction before directional analysis")
        geometric = None
        if frame["calibration_usable"] and len(players) >= 7:
            points = np.array([p["xy_m"] for p in players])
            # Same robust width/depth convention as the existing analysis feature implementation.
            span = np.percentile(points, 90, axis=0) - np.percentile(points, 10, axis=0)
            geometric = {
                "visible_team_width_m": float(span[1]),
                "visible_team_depth_m": float(span[0]),
            }
        findings.append(
            {
                "frame_id": fid,
                "timestamp_s": time,
                "team": team,
                "possession_status": frame["possession"]["status"],
                "geometry": geometric,
                "observed_players": len(players),
                "evidence_ids": [p["evidence_id"] for p in players],
                "requests": requests,
                "tactical_recommendation": {
                    "status": "abstain",
                    "reason": "Video metric accuracy and tactical transfer are unvalidated",
                },
            }
        )
    selected = max(segments, key=lambda s: len(s["frame_ids"])) if segments else None
    return {
        "schema": "tactical-findings/v1",
        "candidate_possessions": segments,
        "selected_candidate": selected,
        "findings": findings,
        "total_frames": len(state["frames"]),
        "unknown_possession_frames": sum(f["possession"]["team"] is None for f in state["frames"]),
        "validated_recommendations": 0,
    }


def brief_stage(source, parents, reviews):
    tactics = parents["tactics"]
    selected = tactics["selected_candidate"]
    lines = [
        "# Video possession review",
        "",
        "Source video SHA-256: `" + source["source_sha256"] + "`.",
        "",
        (
            f"Inspected source PTS {source['window']['start_s']:.2f}–{source['window']['end_s']:.2f}s "
            f"(end exclusive), {tactics['total_frames']} sampled frames."
        ),
        "",
    ]
    if selected:
        lines += [
            (
                f"Candidate possession: anonymous team {selected['team']}, "
                f"{selected['start_s']:.2f}–{selected['last_observed_s']:.2f}s, "
                f"{len(selected['frame_ids'])} observed frames."
            ),
            "",
            "These are first and last supporting observations, not confirmed possession boundaries.",
            "",
        ]
        rows = [f for f in tactics["findings"] if f["frame_id"] in selected["frame_ids"]]
        usable = [r for r in rows if r["geometry"]]
        if usable:
            widths = [r["geometry"]["visible_team_width_m"] for r in usable]
            depths = [r["geometry"]["visible_team_depth_m"] for r in usable]
            lines += [
                (
                    f"Visible-team width spans {min(widths):.1f}–{max(widths):.1f}m; "
                    f"depth spans {min(depths):.1f}–{max(depths):.1f}m across {len(usable)} usable frames."
                ),
                "These are descriptive projected measurements; physical accuracy is unvalidated.",
                "",
            ]
        requests = sorted({request for row in rows for request in row["requests"]})
        lines += ["Review requests:", "", *[f"- {request}." for request in requests], ""]
    else:
        lines += [
            "Possession abstention: no frame resolves the team under the current evidence gates.",
            "Review ball candidates and team association before tactical analysis.",
            "",
        ]
    lines += [
        f"Unknown possession in {tactics['unknown_possession_frames']} of {tactics['total_frames']} frames.",
        "",
        (
            "No coaching recommendation is issued. Ball projection assumes the ground plane; "
            "off-screen players, identity, and metric uncertainty remain unresolved where evidence is absent."
        ),
        "",
        "Original model hypotheses and reviewed alternatives remain available in the state artifact.",
    ]
    return {
        "schema": "analyst-brief/v1",
        "markdown": "\n".join(lines),
        "status": "review_required",
        "selected_candidate": selected,
    }


def specialists():
    return [
        Specialist(
            "evidence",
            "2",
            (),
            "evidence/v1",
            "cpu",
            "Source integrity and clock checks",
            evidence_stage,
        ),
        Specialist(
            "camera",
            "1",
            ("evidence",),
            "observation-channel/v1",
            "cpu",
            "Retain source PTS, shot namespaces, cuts and gaps",
            camera_stage,
        ),
        Specialist(
            "calibration",
            "1",
            ("evidence",),
            "observation-channel/v1",
            "cpu",
            "Keep all fit hypotheses; reject unusable matrices without claiming accuracy",
            calibration_stage,
            libraries=("numpy",),
        ),
        Specialist(
            "players",
            "1",
            ("evidence", "calibration", "camera"),
            "observation-channel/v1",
            "cpu",
            "Retain every person detection; scope anonymous IDs to source and camera shot",
            players_stage,
        ),
        Specialist(
            "ball",
            "1",
            ("evidence", "calibration"),
            "observation-channel/v1",
            "cpu",
            "Preserve every ball hypothesis; ground-plane projection is not a 3D ball state",
            ball_stage,
            code_dependencies=(project_dependency,),
            libraries=("numpy",),
        ),
        Specialist(
            "context",
            "1",
            ("evidence", "camera", "players"),
            "observation-channel/v1",
            "cpu",
            "Append human alternatives with source-time, knowledge-time and shot-scoped evidence",
            context_stage,
            uses_reviews=True,
        ),
        Specialist(
            "state",
            "2",
            ("camera", "calibration", "players", "ball", "context"),
            "match-state/v1",
            "cpu",
            "Compose declared organs; infer possession separately; preserve uncertainty",
            state_stage,
            code_dependencies=(compose_state,),
        ),
        Specialist(
            "tactics",
            "1",
            ("state",),
            "tactical-findings/v1",
            "cpu",
            "No unsupported recommendation; no bridging unknown possession",
            tactics_stage,
            libraries=("numpy",),
        ),
        Specialist(
            "brief",
            "1",
            ("state", "tactics"),
            "analyst-brief/v1",
            "cpu",
            "Every measurement references the versioned state and findings",
            brief_stage,
        ),
    ]
