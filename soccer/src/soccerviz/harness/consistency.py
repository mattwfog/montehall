"""Independent consistency checks for revisable state; never a perception accuracy score."""

from __future__ import annotations

import math
from collections import Counter

RULE_OWNERS = {
    "frame_clock": "camera",
    "source_alignment": "camera",
    "simultaneous_tracks": "players",
    "simultaneous_identity": "context",
    "geometry": "players",
    "calibration_gate": "calibration",
    "pitch_bounds": "calibration",
    "temporal_continuity": "camera",
    "ball_possession": "context",
    "orientation": "context",
    "motion_diagnostic": "players",
}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _point(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(_finite(x) for x in value)


def check_state(state: dict, evidence: dict | None = None, config: dict | None = None) -> dict:
    """Return failures AND skipped coverage without mutating either input.

    Explicit pitch_bounds_m are {x_min,x_max,y_min,y_max} in the same axes as xy_m.
    max_speed_m_s is an optional diagnostic threshold, not a validated acceptance floor.
    max_gap_s is an optional continuity policy, not a physical football constant.
    """
    config = config or {}
    if state.get("schema") != "match-state/v1":
        raise ValueError("Expected match-state/v1")
    for key in ("max_gap_s", "max_speed_m_s"):
        if key in config and (not _finite(config[key]) or config[key] <= 0):
            raise ValueError(f"{key} must be finite and positive")
    frames = state.get("frames", [])
    rules = {
        name: {
            "owner": owner,
            "checked": 0,
            "skipped": 0,
            "skip_reasons": Counter(),
            "violations": 0,
        }
        for name, owner in RULE_OWNERS.items()
    }
    violations, boundaries = [], []
    blocked, break_before = set(), set()
    artifact = {"schema": state["schema"], "source_sha256": (evidence or {}).get("source_sha256")}

    def checked(rule):
        rules[rule]["checked"] += 1

    def skipped(rule, reason):
        rules[rule]["skipped"] += 1
        rules[rule]["skip_reasons"][reason] += 1

    def fail(rule, frame, code, *, refs=(), details=None, key=None, blocking=True):
        rules[rule]["violations"] += 1
        fid = frame.get("frame_id")
        if blocking and fid is not None:
            blocked.add(fid)
        violations.append(
            {
                "rule": rule,
                "code": code,
                "owner": RULE_OWNERS[rule],
                "artifact": artifact,
                "frame_id": fid,
                "timestamp_s": frame.get("timestamp_s")
                if _finite(frame.get("timestamp_s"))
                else None,
                "shot_id": frame.get("shot_id"),
                "key": key if key is not None else f"frame:{fid}",
                "evidence_ids": [r for r in refs if r is not None],
                "details": details or {},
                "blocking": blocking,
            }
        )

    def boundary(frame, reason, previous, dt=None):
        fid = frame.get("frame_id")
        if fid is not None:
            break_before.add(fid)
        boundaries.append(
            {
                "frame_id": fid,
                "timestamp_s": frame.get("timestamp_s")
                if _finite(frame.get("timestamp_s"))
                else None,
                "previous_frame_id": previous.get("frame_id"),
                "reason": reason,
                "delta_s": dt,
            }
        )

    evidence_frames = {(r.get("frame_id")): r for r in (evidence or {}).get("frames", [])}
    calibration = {}
    for fit in (evidence or {}).get("calibration_hypotheses", []):
        calibration.setdefault(fit.get("frame_id"), []).append(fit)
    bounds = config.get("pitch_bounds_m", state.get("pitch_bounds_m"))
    if bounds is not None and (
        not isinstance(bounds, dict)
        or not all(_finite(bounds.get(k)) for k in ("x_min", "x_max", "y_min", "y_max"))
        or bounds["x_min"] >= bounds["x_max"]
        or bounds["y_min"] >= bounds["y_max"]
    ):
        raise ValueError("pitch_bounds_m requires finite ordered limits in state coordinate axes")
    seen_frames = set()
    previous = None
    previous_entities = {}
    for frame in frames:
        fid, timestamp, shot = frame.get("frame_id"), frame.get("timestamp_s"), frame.get("shot_id")
        checked("frame_clock")
        if fid is None or fid in seen_frames or not _finite(timestamp):
            fail(
                "frame_clock",
                frame,
                "invalid_or_duplicate_frame_clock",
                details={
                    "duplicate_frame": fid in seen_frames,
                    "finite_timestamp": _finite(timestamp),
                },
            )
        seen_frames.add(fid)
        if evidence is None or "frames" not in evidence:
            skipped("source_alignment", "source_frame_channel_absent")
        else:
            checked("source_alignment")
            original = evidence_frames.get(fid)
            if original is None:
                fail("source_alignment", frame, "frame_absent_from_source")
            elif timestamp != original.get("timestamp_s") or (
                "shot_id" in original and shot != original["shot_id"]
            ):
                fail(
                    "source_alignment",
                    frame,
                    "source_clock_or_camera_mismatch",
                    details={
                        "source_timestamp_s": original.get("timestamp_s"),
                        "source_shot_id": original.get("shot_id"),
                    },
                )
        entities = frame.get("entities", frame.get("players"))
        if entities is None:
            for rule in ("simultaneous_tracks", "simultaneous_identity"):
                skipped(rule, "player_channel_absent")
            entities = []
        else:
            tracks, identities = {}, {}
            for entity in entities:
                refs = [entity.get("evidence_id")]
                track, identity = entity.get("tracklet_id"), entity.get("identity")
                if shot is None or track is None:
                    skipped("simultaneous_tracks", "shot_or_tracklet_unknown")
                else:
                    checked("simultaneous_tracks")
                    key = (shot, track)
                    if key in tracks:
                        fail(
                            "simultaneous_tracks",
                            frame,
                            "duplicate_simultaneous_tracklet",
                            refs=[*refs, tracks[key]],
                            key=f"shot:{shot}:tracklet:{track}",
                        )
                    tracks[key] = entity.get("evidence_id")
                if shot is None or identity is None:
                    skipped("simultaneous_identity", "shot_or_identity_unknown")
                else:
                    checked("simultaneous_identity")
                    key = (shot, identity)
                    if key in identities:
                        fail(
                            "simultaneous_identity",
                            frame,
                            "duplicate_simultaneous_identity",
                            refs=[*refs, identities[key]],
                            key=f"shot:{shot}:identity:{identity}",
                        )
                    identities[key] = entity.get("evidence_id")
        balls = frame.get("ball_hypotheses")
        geometry_entities = [*entities, *(balls or [])]
        if not geometry_entities:
            for rule in ("geometry", "calibration_gate", "pitch_bounds"):
                skipped(rule, "no_entity_geometry_channel")
        for entity in geometry_entities:
            xy, ref = entity.get("xy_m"), entity.get("evidence_id")
            if xy is None:
                for rule in ("geometry", "calibration_gate", "pitch_bounds"):
                    skipped(rule, "position_unknown")
                continue
            checked("geometry")
            if entity.get("position_status") == "unavailable":
                fail("geometry", frame, "unavailable_position_has_coordinates", refs=[ref])
            if not _point(xy):
                fail(
                    "geometry",
                    frame,
                    "invalid_position",
                    refs=[ref],
                    details={
                        "coordinate_count": len(xy) if isinstance(xy, (list, tuple)) else None
                    },
                )
            provider_position = (
                frame.get("position_origin", state.get("position_origin")) == "provider_tracking"
            )
            if provider_position:
                skipped(
                    "calibration_gate",
                    "provider_metric_positions_do_not_require_camera_calibration",
                )
            elif "calibration_usable" not in frame:
                skipped("calibration_gate", "calibration_channel_absent")
            else:
                checked("calibration_gate")
                if frame["calibration_usable"] is not True:
                    fail(
                        "calibration_gate", frame, "position_with_rejected_calibration", refs=[ref]
                    )
                elif evidence is not None and "calibration_hypotheses" in evidence:
                    fits = calibration.get(fid, [])
                    if len(fits) != 1 or fits[0].get("accepted") is not True:
                        fail(
                            "calibration_gate",
                            frame,
                            "position_without_unique_accepted_source_calibration",
                            refs=[ref],
                            details={"source_fits": len(fits)},
                        )
            if bounds is None:
                skipped("pitch_bounds", "explicit_pitch_bounds_absent")
            elif not _point(xy):
                skipped("pitch_bounds", "position_invalid")
            else:
                checked("pitch_bounds")
                if not (
                    bounds["x_min"] <= xy[0] <= bounds["x_max"]
                    and bounds["y_min"] <= xy[1] <= bounds["y_max"]
                ):
                    fail(
                        "pitch_bounds",
                        frame,
                        "outside_explicit_pitch_bounds",
                        refs=[ref],
                        details={
                            "xy_m": list(xy),
                            "bounds_m": dict(bounds),
                            "interpretation": "outside playing surface may be legitimate; diagnostic only",
                        },
                        blocking=False,
                    )
        possession = frame.get("possession")
        if possession is None or balls is None:
            skipped("ball_possession", "ball_or_possession_channel_absent")
        elif possession.get("team") is None:
            checked("ball_possession")  # Abstention is consistent, not evidence of good tracking.
        else:
            checked("ball_possession")
            team = possession["team"]
            support = [h for h in possession.get("hypotheses", []) if h.get("value") == team]
            review = [h for h in support if h.get("source") == "review" and h.get("evidence_ids")]
            refs = [r for h in support for r in h.get("evidence_ids", [])]
            provider_support = [
                h
                for h in support
                if h.get("source") in ("provider_tracking", "provider") and h.get("evidence_ids")
            ]
            if state.get("source_kind") == "provider" and provider_support:
                pass  # A provider claim remains model-derived; it is not our visual ball proxy.
            elif possession.get("status") == "reviewed":
                if not review:
                    fail(
                        "ball_possession",
                        frame,
                        "reviewed_possession_without_review_evidence",
                        refs=refs,
                    )
            else:
                localized = [
                    b
                    for b in balls
                    if _point(b.get("xy_m"))
                    and b.get("position_status")
                    not in ("estimated", "extrapolated", "unknown", "unavailable")
                ]
                player_ids = {
                    p.get("evidence_id")
                    for p in entities
                    if p.get("team") == team
                    and _point(p.get("xy_m"))
                    and p.get("position_status")
                    not in ("estimated", "extrapolated", "unknown", "unavailable")
                }
                ball_ids = {b.get("evidence_id") for b in localized}
                supported = any(
                    set(h.get("evidence_ids", [])) & player_ids
                    and set(h.get("evidence_ids", [])) & ball_ids
                    for h in support
                )
                if len(balls) != 1 or len(localized) != 1 or not supported:
                    fail(
                        "ball_possession",
                        frame,
                        "unsupported_or_ambiguous_possession",
                        refs=refs,
                        details={
                            "ball_candidates": len(balls),
                            "localized_balls": len(localized),
                            "matching_support": bool(supported),
                        },
                    )
        orientation = frame.get("orientation")
        supported_directions = {}
        if not orientation:
            skipped("orientation", "orientation_channel_absent")
        else:
            for team, item in orientation.items():
                selected = item.get("selected")
                if selected is None:
                    skipped("orientation", "direction_unknown")
                    continue
                if type(selected) is not int or selected not in (-1, 1):
                    checked("orientation")
                    fail("orientation", frame, "invalid_orientation", key=str(team))
                    continue
                support = [
                    h
                    for h in item.get("hypotheses", [])
                    if h.get("value") == selected
                    and h.get("source") not in (None, "unresolved")
                    and h.get("evidence_ids")
                ]
                if not support:
                    skipped("orientation", "direction_evidence_unavailable")
                    continue
                checked("orientation")
                supported_directions[team] = (selected, support)
            if len(supported_directions) == 2:
                values = list(supported_directions.values())
                if values[0][0] == values[1][0]:
                    fail(
                        "orientation",
                        frame,
                        "opponents_share_attacking_direction",
                        refs=[r for _, hs in values for h in hs for r in h["evidence_ids"]],
                        details={
                            "directions": {str(k): v[0] for k, v in supported_directions.items()}
                        },
                    )
        if previous is None:
            skipped("temporal_continuity", "no_previous_frame")
            skipped("motion_diagnostic", "no_previous_frame")
        else:
            previous_time = previous.get("timestamp_s")
            dt = (
                timestamp - previous_time if _finite(timestamp) and _finite(previous_time) else None
            )
            if dt is None or shot is None or previous.get("shot_id") is None:
                skipped("temporal_continuity", "clock_or_camera_unknown")
                boundary(frame, "clock_or_camera_unknown", previous, dt)
            else:
                checked("temporal_continuity")
                if dt <= 0:
                    fail(
                        "temporal_continuity",
                        frame,
                        "nonincreasing_source_time",
                        details={"delta_s": dt},
                    )
                    boundary(frame, "nonincreasing_source_time", previous, dt)
                elif shot != previous["shot_id"]:
                    boundary(frame, "camera_cut", previous, dt)
                elif "max_gap_s" in config and dt > config["max_gap_s"]:
                    boundary(frame, "sampling_gap", previous, dt)
            if "max_speed_m_s" not in config:
                skipped("motion_diagnostic", "diagnostic_threshold_not_configured")
            elif fid in break_before or dt is None or dt <= 0:
                skipped("motion_diagnostic", "discontinuous_frame_pair")
            else:
                matched = 0
                for entity in entities:
                    key = (shot, entity.get("tracklet_id"))
                    old = previous_entities.get(key)
                    if (
                        key[1] is None
                        or old is None
                        or not _point(old.get("xy_m"))
                        or not _point(entity.get("xy_m"))
                    ):
                        continue
                    checked("motion_diagnostic")
                    matched += 1
                    speed = math.dist(old["xy_m"], entity["xy_m"]) / dt
                    if speed > config["max_speed_m_s"]:
                        fail(
                            "motion_diagnostic",
                            frame,
                            "configured_speed_diagnostic_exceeded",
                            refs=[old.get("evidence_id"), entity.get("evidence_id")],
                            blocking=False,
                            key=f"shot:{shot}:tracklet:{key[1]}",
                            details={
                                "speed_m_s": speed,
                                "threshold_m_s": config["max_speed_m_s"],
                                "delta_s": dt,
                                "interpretation": "diagnostic only; not an accuracy acceptance floor",
                            },
                        )
                if not matched:
                    skipped("motion_diagnostic", "no_corresponding_localized_tracks")
        previous = frame
        previous_entities = {(shot, p.get("tracklet_id")): p for p in entities}
    if not frames:
        for rule in rules:
            skipped(rule, "state_frames_absent")
    for rule, entry in rules.items():
        if not entry["checked"] and not entry["skipped"]:
            skipped(rule, "no_applicable_observations")
        entry["skip_reasons"] = dict(entry["skip_reasons"])
        entry["status"] = (
            "violations"
            if entry["violations"]
            else "partial"
            if entry["skipped"] and entry["checked"]
            else "skipped"
            if entry["skipped"]
            else "checked"
        )
    status = (
        "violations"
        if violations
        else "partial"
        if any(r["skipped"] for r in rules.values())
        else "checked"
    )
    return {
        "schema": "consistency-report/v1",
        "status": status,
        "frames": len(frames),
        "rules": rules,
        "violations": violations,
        "blocking_frame_ids": sorted(blocked),
        "break_before_frame_ids": sorted(break_before),
        "boundaries": boundaries,
        "config": dict(config),
        "interpretation": "Consistency is not tracking or tactical accuracy. Skipped checks remain unknown.",
    }


def run_consistency(source, parents, reviews):
    """Harness specialist wrapper; reviewed alternatives are already present in state."""
    config = dict(source.get("consistency_config", {}))
    if "max_gap_s" not in config and _finite(source.get("sample_period_s")):
        config["max_gap_s"] = 1.5 * source["sample_period_s"]
    return check_state(parents["state"], parents.get("evidence"), config)
