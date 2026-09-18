"""Causal, label-free camera and ball checks. Heuristics are not calibrated uncertainty."""

from dataclasses import asdict, dataclass

import numpy as np

from soccerviz.core.geometry import project


@dataclass(frozen=True)
class SafetyConfig:
    minimum_people: int = 4
    minimum_in_pitch_fraction: float = 0.7
    pitch_margin_m: float = 10.0
    maximum_scale_m_per_pixel: float = 1.5
    camera_jump_m: float = 12.0
    camera_reacquire_frames: int = 3
    ball_confirmations: int = 2
    ball_expiry_s: float = 0.4
    ball_gate_diagonal_fraction: float = 0.025
    ball_speed_diagonals_per_s: float = 0.35


class CameraGuard:
    def __init__(self, config=None):
        self.config = config or SafetyConfig()
        self.reset()

    def reset(self):
        self.previous = None
        self.pending = None
        self.pending_count = 0

    def check(self, raw, feet, timestamp):
        """Input/output homographies use centered pitch meters; never hold a stale camera."""
        if not np.isfinite(timestamp) or (
            self.previous is not None and timestamp <= self.previous[1]
        ):
            raise ValueError("Camera timestamps must be finite and increase")
        diagnostics = {"policy": asdict(self.config), "raw_accepted": bool(raw.get("accepted"))}
        result = {
            "accepted": False,
            "reason": raw.get("reason", "missing_camera"),
            "homography": None,
            "diagnostics": diagnostics,
        }
        if not raw.get("accepted") or raw.get("homography") is None:
            self.pending, self.pending_count = None, 0
            return result
        matrix = np.asarray(raw["homography"], float)
        feet = np.asarray(feet, float).reshape(-1, 2)
        reason = None
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            reason = "invalid_matrix"
        elif np.linalg.matrix_rank(matrix) != 3 or np.linalg.cond(matrix) > 1e12:
            reason = "singular_camera"
        elif len(feet) < self.config.minimum_people:
            reason = "insufficient_person_support"
        else:
            xy = project(feet, matrix)
            finite = np.isfinite(xy).all(axis=1)
            inside = finite & (np.abs(xy) <= np.array([52.5, 34]) + self.config.pitch_margin_m).all(
                axis=1
            )
            fraction = float(inside.mean())
            local_scales = np.concatenate(
                [
                    np.linalg.norm(project(feet + offset, matrix) - xy, axis=1) / 2
                    for offset in ([2, 0], [0, 2])
                ]
            )
            scale = float(np.quantile(local_scales, 0.9))
            diagnostics.update(
                in_pitch_fraction=fraction,
                local_scale_p90_m_per_pixel=scale if np.isfinite(scale) else None,
            )
            if not finite.all() or fraction < self.config.minimum_in_pitch_fraction:
                reason = "implausible_person_projection"
            elif not np.isfinite(scale) or scale > self.config.maximum_scale_m_per_pixel:
                reason = "implausible_metric_scale"
            elif self.previous is not None:
                previous_matrix, previous_time = self.previous
                drift = float(
                    np.median(np.linalg.norm(xy - project(feet, previous_matrix), axis=1))
                )
                allowance = self.config.camera_jump_m + 20 * min(timestamp - previous_time, 1)
                diagnostics["camera_drift_m"] = drift if np.isfinite(drift) else None
                if not np.isfinite(drift) or drift > allowance:
                    stable_pending = (
                        self.pending is not None
                        and float(
                            np.median(np.linalg.norm(xy - project(feet, self.pending), axis=1))
                        )
                        < self.config.camera_jump_m
                    )
                    self.pending_count = self.pending_count + 1 if stable_pending else 1
                    self.pending = matrix.copy()
                    if self.pending_count < self.config.camera_reacquire_frames:
                        reason = "camera_jump_pending_reacquisition"
        if reason:
            if reason != "camera_jump_pending_reacquisition":
                self.pending, self.pending_count = None, 0
            result["reason"] = reason
            return result
        self.previous = (matrix.copy(), timestamp)
        self.pending, self.pending_count = None, 0
        result.update(
            accepted=True, reason="passed_physical_and_temporal_checks", homography=matrix.tolist()
        )
        return result


class BallTrack:
    def __init__(self, config=None):
        self.config = config or SafetyConfig()
        self.reset()

    def reset(self):
        self.center = None
        self.velocity = np.zeros(2)
        self.last_observed = None
        self.last_frame = None
        self.hits = 0
        self.age = 0

    def update(self, candidates, timestamp, shape, camera_shift=(0, 0), cut=False):
        if cut:
            self.reset()
        if self.last_frame is not None and timestamp <= self.last_frame:
            raise ValueError("Ball tracker timestamps must increase")
        diagonal = float(np.hypot(*shape[:2]))
        dt = 0 if self.last_frame is None else timestamp - self.last_frame
        self.last_frame = timestamp
        shift = np.asarray(camera_shift, float)
        if not np.isfinite(shift).all():
            raise ValueError("Camera shift must be finite")
        if (
            self.last_observed is not None
            and timestamp - self.last_observed > self.config.ball_expiry_s
        ):
            self.center, self.hits = None, 0
            self.velocity[:] = 0
        predicted = None if self.center is None else self.center + shift + self.velocity * dt
        gate = diagonal * (
            self.config.ball_gate_diagonal_fraction
            + self.config.ball_speed_diagonals_per_s * min(dt, 0.4)
        )
        proposals = []
        for index, candidate in enumerate(candidates):
            box = np.asarray(candidate["bbox_xyxy"], float)
            score = float(candidate["score"])
            if box.shape != (4,) or not np.isfinite(box).all() or not 0 <= score <= 1:
                raise ValueError("Invalid ball candidate")
            center = (box[:2] + box[2:]) / 2
            distance = 0 if predicted is None else float(np.linalg.norm(center - predicted))
            if predicted is None or distance <= gate:
                cost = -score if predicted is None else distance / max(gate, 1) - 0.35 * score
                proposals.append((cost, index, center))
        if proposals:
            _, index, center = min(proposals, key=lambda row: (row[0], row[1]))
            if self.center is not None and dt > 0:
                velocity = (center - self.center - shift) / dt
                limit = diagonal * self.config.ball_speed_diagonals_per_s
                velocity *= min(1, limit / max(np.linalg.norm(velocity), 1))
                self.velocity = 0.5 * self.velocity + 0.5 * velocity
            self.center, self.last_observed = center, timestamp
            self.hits += 1
            status = "detected" if self.hits >= self.config.ball_confirmations else "tentative"
            return {
                "status": status,
                "center_xy": center.tolist(),
                "selected_index": index,
                "eligible_for_state": status == "detected",
                "observed": True,
                "age_s": 0.0,
                "reason": "temporal_confirmation",
            }
        age = None if self.last_observed is None else timestamp - self.last_observed
        if predicted is not None and self.hits >= self.config.ball_confirmations:
            self.center = predicted
            return {
                "status": "estimated",
                "center_xy": predicted.tolist(),
                "selected_index": None,
                "eligible_for_state": False,
                "observed": False,
                "age_s": age,
                "reason": "short_gap_extrapolation",
            }
        self.hits = 0
        return {
            "status": "unknown",
            "center_xy": None,
            "selected_index": None,
            "eligible_for_state": False,
            "observed": False,
            "age_s": age,
            "reason": "missing_or_motion_rejected",
        }


def jersey_consensus(rows, minimum_frames=2):
    """Return causal per-track hypotheses only after independent accepted frames agree."""
    evidence = {}
    outputs = []
    for row in sorted(rows, key=lambda r: (r["timestamp_s"], r["detection_id"])):
        key = (row["shot_id"], row["tracklet_id"])
        votes = evidence.setdefault(key, {})
        number = row.get("number")
        if row.get("accepted") and number is not None:
            votes.setdefault(int(number), set()).add(row["frame_id"])
        supported = [number for number, frames in votes.items() if len(frames) >= minimum_frames]
        conflict = len(votes) > 1
        output = supported[0] if len(supported) == 1 and not conflict else None
        outputs.append(
            {
                **row,
                "jersey_hypothesis": output,
                "jersey_status": "conflicting"
                if conflict
                else "supported_hypothesis"
                if output is not None
                else "insufficient_evidence",
            }
        )
    return outputs
