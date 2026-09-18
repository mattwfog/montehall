"""Independent observation organs and an explicit shared match-state composition.

Channels carry observations, model hypotheses, or human amendments separately.
No detector score is promoted to a calibrated state probability. The frame clock
and camera-shot namespace survive composition, including unavailable channels.
"""

from __future__ import annotations

import math
from typing import Literal, TypedDict

import numpy as np

from soccerviz.core.geometry import project


class ChannelArtifact(TypedDict):
    schema: Literal["observation-channel/v1"]
    channel: str
    source_sha256: str
    window: dict
    frames: list[dict]
    census: dict
    provenance: dict
    uncertainty: str


def channel_artifact(
    source,
    name,
    frames,
    *,
    observations,
    observed_frames,
    available=True,
    consumed=(),
    kind="model_hypotheses",
) -> ChannelArtifact:
    expected = [frame["frame_id"] for frame in source["frames"]]
    observed = set(observed_frames)
    return {
        "schema": "observation-channel/v1",
        "channel": name,
        "source_sha256": source["source_sha256"],
        "window": source["window"],
        "frames": frames,
        "census": {
            "status": "missing" if not available else ("present" if observations else "empty"),
            "expected_frames": len(expected),
            "frames_with_observations": len(observed),
            "observation_count": observations,
            "frames_without_observations": [fid for fid in expected if fid not in observed],
        },
        "provenance": {
            "kind": kind,
            "clock": "source PTS seconds",
            "consumed_channels": list(consumed),
            "model_sha256": source.get("model_sha256", {}),
            "input_artifact_sha256": source.get("artifact_sha256", {}),
        },
        "uncertainty": "Scores remain source scores; calibrated probability and metric sigma are unknown",
    }


def applicable(reviews, time, field, **target):
    return [
        item
        for item in reviews
        if item["review"]["field"] == field
        and item["review"]["start_s"] <= time < item["review"]["end_s"]
        and all(item["review"].get(key) == value for key, value in target.items())
    ]


def alternatives(initial, items):
    return [
        *initial,
        *[
            {
                "value": item["review"]["value"],
                "source": "review",
                "revision": item["revision"],
                "recorded_at": item["created_at"],
                "evidence_ids": item["review"]["evidence_ids"],
                "reviewer": item["review"]["reviewer"],
                "reason": item["review"]["reason"],
            }
            for item in items
        ],
    ]


def _group(rows, key="frame_id"):
    result = {}
    for row in rows:
        result.setdefault(row[key], []).append(row)
    return result


def camera_stage(source, parents, reviews):
    evidence = parents["evidence"]
    frames = []
    previous = None
    for frame in evidence["frames"]:
        current = dict(frame)
        current["cut_before"] = previous is not None and frame.get("shot_id") != previous.get(
            "shot_id"
        )
        current["source_gap_s"] = (
            frame["timestamp_s"] - previous["timestamp_s"] if previous is not None else None
        )
        current["shot_status"] = (
            "source_hypothesis" if frame.get("shot_id") is not None else "unknown"
        )
        current["evidence_ids"] = [f"frame:{frame['frame_id']}"]
        frames.append(current)
        previous = frame
    return channel_artifact(
        evidence,
        "camera",
        frames,
        observations=len(frames),
        observed_frames=[f["frame_id"] for f in frames],
        consumed=("evidence",),
        kind="source_clock_and_camera_hypotheses",
    )


def calibration_stage(source, parents, reviews):
    evidence = parents["evidence"]
    grouped = _group(evidence.get("calibration_hypotheses", []))
    frames = []
    for frame in evidence["frames"]:
        candidates = grouped.get(frame["frame_id"], [])
        fit = candidates[0] if len(candidates) == 1 else None
        matrix = None
        reason = "missing" if not candidates else "ambiguous_calibration"
        if fit is not None:
            reason = "source_rejected"
            if fit.get("accepted"):
                try:
                    candidate = np.asarray(fit.get("homography"), dtype=float).reshape(3, 3)
                    if np.isfinite(candidate).all() and np.linalg.matrix_rank(candidate) == 3:
                        matrix = candidate.ravel().tolist()
                        reason = "accepted_hypothesis"
                    else:
                        reason = "invalid_homography"
                except (ValueError, TypeError):
                    reason = "invalid_homography"
        frames.append(
            {
                "frame_id": frame["frame_id"],
                "timestamp_s": frame["timestamp_s"],
                "usable": matrix is not None,
                "homography": matrix,
                "status": reason,
                "hypotheses": candidates,
                "metric_accuracy": "unvalidated",
                "coordinate_system": "assumed_105x68_top_left_metres",
                "evidence_ids": [f"calibration:{frame['frame_id']}"] if candidates else [],
            }
        )
    return channel_artifact(
        evidence,
        "calibration",
        frames,
        observations=sum(map(len, grouped.values())),
        observed_frames=grouped,
        available="calibration_hypotheses" in evidence,
        consumed=("evidence",),
    )


def players_stage(source, parents, reviews):
    evidence = parents["evidence"]
    calibrations = {r["frame_id"]: r for r in parents["calibration"]["frames"]}
    cameras = {r["frame_id"]: r for r in parents["camera"]["frames"]}
    detections = _group(evidence.get("detections", []))
    projections = _group(evidence.get("perception_hypotheses", []), "evidence_id")
    frames = []
    for frame in evidence["frames"]:
        fid = frame["frame_id"]
        entities = []
        for detection in detections.get(fid, []):
            candidates = projections.get(detection["detection_id"], [])
            row = candidates[0] if len(candidates) == 1 else None
            track = detection.get("tracklet_id")
            if track is None and row is not None:
                track = row.get("tracklet_id")
            shot = cameras[fid].get("shot_id")
            # A track ID is local to a camera shot, not a permanent person identity.
            anonymous = (
                f"{evidence['source_sha256']}:shot:{shot}:track:{track}"
                if track is not None and shot is not None
                else f"{evidence['source_sha256']}:detection:{detection['detection_id']}"
            )
            xy = [row.get("x_m"), row.get("y_m")] if row else [None, None]
            valid = bool(
                row
                and calibrations[fid]["usable"]
                and row.get("calibration_accepted")
                and all(v is not None and math.isfinite(v) for v in xy)
            )
            team = row.get("team_cluster") if row else None
            entities.append(
                {
                    "entity_id": anonymous,
                    "tracklet_id": track,
                    "shot_id": shot,
                    "evidence_id": detection["detection_id"],
                    "role": detection.get("role_hypothesis", "unknown"),
                    "bbox_xyxy": [
                        detection.get(key) for key in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")
                    ],
                    "detector_confidence": detection.get("confidence"),
                    "xy_m": xy if valid else None,
                    "position_sigma_m": None,
                    "position_status": "projected_hypothesis" if valid else "unknown",
                    "position_hypotheses": candidates,
                    "team": team,
                    "team_hypotheses": [
                        {
                            "value": team,
                            "source": "color_association" if row else "unknown",
                            "evidence_ids": [detection["detection_id"]],
                        }
                    ],
                    "identity": None,
                    "identity_hypotheses": [{"value": None, "source": "anonymous_tracklet"}],
                    "jersey_hypothesis": detection.get("jersey_hypothesis"),
                    "jersey_status": detection.get("jersey_status", "unknown"),
                }
            )
        frames.append({"frame_id": fid, "timestamp_s": frame["timestamp_s"], "entities": entities})
    return channel_artifact(
        evidence,
        "players",
        frames,
        observations=sum(map(len, detections.values())),
        observed_frames=detections,
        available="detections" in evidence,
        consumed=("evidence", "calibration", "camera"),
    )


def ball_stage(source, parents, reviews):
    evidence = parents["evidence"]
    fits = {r["frame_id"]: r for r in parents["calibration"]["frames"]}
    grouped = _group(evidence.get("ball_candidates", []))
    frames = []
    for frame in evidence["frames"]:
        fid = frame["frame_id"]
        fit = fits[fid]
        hypotheses = []
        for row in grouped.get(fid, []):
            xy = None
            eligible = row.get("eligible_for_state", True)
            if fit["usable"] and eligible:
                center = [[(row["x0"] + row["x1"]) / 2, (row["y0"] + row["y1"]) / 2]]
                point = project(np.array(center), np.array(fit["homography"]).reshape(3, 3))[0]
                if np.isfinite(point).all() and -3 <= point[0] <= 108 and -3 <= point[1] <= 71:
                    xy = point.tolist()
            hypotheses.append(
                {
                    "evidence_id": f"f{fid}-ball{row['rank']}",
                    "xy_m": xy,
                    "bbox_xyxy": [row[key] for key in ("x0", "y0", "x1", "y1")],
                    "detector_confidence": row["confidence"],
                    "position_sigma_m": None,
                    "position_status": "ground_plane_hypothesis" if xy is not None else "unknown",
                    "height_m": None,
                    "mode": "unknown",
                    "source": "ball_detection",
                    "eligible_for_state": eligible,
                }
            )
        frames.append(
            {"frame_id": fid, "timestamp_s": frame["timestamp_s"], "hypotheses": hypotheses}
        )
    return channel_artifact(
        evidence,
        "ball",
        frames,
        observations=sum(map(len, grouped.values())),
        observed_frames=grouped,
        available="ball_candidates" in evidence,
        consumed=("evidence", "calibration"),
    )


def context_stage(source, parents, reviews):
    evidence = parents["evidence"]
    player_frames = {r["frame_id"]: r for r in parents["players"]["frames"]}
    cameras = {r["frame_id"]: r for r in parents["camera"]["frames"]}
    evidence_shots = {
        entity["evidence_id"]: cameras[frame["frame_id"]].get("shot_id")
        for frame in parents["players"]["frames"]
        for entity in frame["entities"]
    }
    frames, touched, count = [], set(), 0
    for frame in evidence["frames"]:
        fid, time = frame["frame_id"], frame["timestamp_s"]
        amendments = {}
        for entity in player_frames[fid]["entities"]:
            fields = {}
            for field in ("team", "identity"):
                items = applicable(reviews, time, field, tracklet_id=entity["tracklet_id"])
                # Numeric tracker IDs may restart after a cut. Human evidence scopes
                # a person amendment to the shots containing its referenced person.
                items = [
                    item
                    for item in items
                    if cameras[fid].get("shot_id")
                    in {
                        evidence_shots[eid]
                        for eid in item["review"]["evidence_ids"]
                        if eid in evidence_shots
                    }
                ]
                fields[field] = alternatives([], items)
                count += len(items)
            if any(fields.values()):
                amendments[entity["entity_id"]] = fields
        possession = alternatives([], applicable(reviews, time, "possession"))
        orientation = {
            str(team): {
                "selected": None,
                "hypotheses": [
                    {"value": direction, "source": "unresolved"} for direction in (-1, 1)
                ],
            }
            for team in (0, 1)
        }
        for team in (0, 1):
            changes = applicable(reviews, time, "orientation", team_cluster=team)
            orientation[str(team)]["hypotheses"] = alternatives(
                orientation[str(team)]["hypotheses"], changes
            )
            if changes:
                orientation[str(team)]["selected"] = changes[-1]["review"]["value"]
            count += len(changes)
        count += len(possession)
        if (
            amendments
            or possession
            or any(o["hypotheses"][-1]["source"] == "review" for o in orientation.values())
        ):
            touched.add(fid)
        frames.append(
            {
                "frame_id": fid,
                "timestamp_s": time,
                "entity_amendments": amendments,
                "possession_amendments": possession,
                "orientation": orientation,
            }
        )
    result = channel_artifact(
        evidence,
        "context",
        frames,
        observations=count,
        observed_frames=touched,
        consumed=("evidence", "camera", "players"),
        kind="human_review_amendments",
    )
    result["review_history"] = reviews
    result["review_revisions"] = [item["revision"] for item in reviews]
    return result


def _channel_frames(parents, name, camera):
    channel = parents[name]
    if channel["schema"] != "observation-channel/v1" or channel["channel"] != name:
        raise ValueError(f"Invalid {name} channel schema")
    if channel["source_sha256"] != camera["source_sha256"] or channel["window"] != camera["window"]:
        raise ValueError(f"{name} belongs to another source or time window")
    frames = channel["frames"]
    ids = [frame["frame_id"] for frame in frames]
    expected = {frame["frame_id"]: frame["timestamp_s"] for frame in camera["frames"]}
    if len(ids) != len(set(ids)) or any(
        expected.get(frame["frame_id"]) != frame["timestamp_s"] for frame in frames
    ):
        raise ValueError(f"{name} frame IDs or source clock differ")
    if set(ids) != set(expected):
        raise ValueError(f"{name} must retain every frame, including empty observations")
    return {frame["frame_id"]: frame for frame in frames}


def compose_state(source, parents, reviews=()):
    """Compose organs into revisable match-state; infer possession explicitly."""
    camera = parents["camera"]
    by_channel = {
        name: _channel_frames(parents, name, camera)
        for name in ("camera", "calibration", "players", "ball", "context")
    }
    frames = []
    for frame in camera["frames"]:
        fid = frame["frame_id"]
        fit = by_channel["calibration"][fid]
        context = by_channel["context"][fid]
        entities = []
        for original in by_channel["players"][fid]["entities"]:
            entity = dict(original)
            amendments = context["entity_amendments"].get(entity["entity_id"], {})
            for field in ("team", "identity"):
                hypotheses = [*original[f"{field}_hypotheses"], *amendments.get(field, [])]
                entity[f"{field}_hypotheses"] = hypotheses
                entity[field] = hypotheses[-1]["value"]
                entity[f"{field}_status"] = (
                    "reviewed"
                    if amendments.get(field)
                    else ("hypothesis" if entity[field] is not None else "unknown")
                )
            entities.append(entity)
        # Keep the legacy player role selection explicit. Goalkeepers/referees
        # remain in entities, so their absence from tactical geometry is visible.
        players = [entity for entity in entities if entity["role"] == "player"]
        balls = by_channel["ball"][fid]["hypotheses"]
        eligible_balls = [ball for ball in balls if ball.get("eligible_for_state", True)]
        candidates = []
        for ball in eligible_balls:
            if ball["xy_m"] is None:
                continue
            for team in (0, 1):
                teammates = [p for p in players if p["team"] == team and p["xy_m"] is not None]
                if teammates:
                    near = min(
                        teammates, key=lambda player: math.dist(player["xy_m"], ball["xy_m"])
                    )
                    candidates.append(
                        {
                            "value": team,
                            "source": "nearest_player_proxy",
                            "distance_m": math.dist(near["xy_m"], ball["xy_m"]),
                            "evidence_ids": [ball["evidence_id"], near["evidence_id"]],
                        }
                    )
        ordered = sorted(candidates, key=lambda candidate: candidate["distance_m"])
        selected = None
        if (
            len(eligible_balls) == 1
            and len(ordered) == 2
            and ordered[0]["distance_m"] < 2.5
            and ordered[1]["distance_m"] - ordered[0]["distance_m"] > 0.75
        ):
            selected = ordered[0]["value"]
        changes = context["possession_amendments"]
        if changes:
            selected = changes[-1]["value"]
        frames.append(
            {
                "frame_id": fid,
                "timestamp_s": frame["timestamp_s"],
                "shot_id": frame.get("shot_id"),
                "camera": frame,
                "calibration_usable": fit["usable"],
                "calibration_status": fit["status"],
                "metric_accuracy": "unvalidated",
                "players": players,
                "entities": entities,
                "ball_hypotheses": balls,
                "orientation": context["orientation"],
                "possession": {
                    "team": selected,
                    "status": "reviewed"
                    if changes
                    else ("hypothesis" if selected is not None else "unknown"),
                    "hypotheses": [*candidates, *changes],
                    "probability": None,
                },
            }
        )
    return {
        "schema": "match-state/v1",
        "source_sha256": camera["source_sha256"],
        "window": camera["window"],
        "frames": frames,
        "review_revisions": parents["context"].get("review_revisions", []),
        "channels": {
            name: {
                "schema": parents[name]["schema"],
                "census": parents[name]["census"],
                "provenance": parents[name]["provenance"],
            }
            for name in by_channel
        },
        "uncertainty": "Detector confidence is not positional or possession probability",
        "coordinate_system": "assumed_105x68_top_left_metres",
        "inference_policy": "Single projected ball; nearest player for both anonymous teams; proximity and margin gates; human amendments remain separate alternatives",
    }


def video_channels(source, evidence, reviews=()):
    """Compatibility composition for callers predating the explicit organ graph."""
    parents = {"evidence": evidence}
    for name, stage in (
        ("camera", camera_stage),
        ("calibration", calibration_stage),
        ("players", players_stage),
        ("ball", ball_stage),
        ("context", context_stage),
    ):
        parents[name] = stage(source, parents, reviews if name == "context" else ())
    return parents
