"""Quark layer: per-body seekers claiming raw person detections.

Architecture (ratified 2026-08-05): individual seekers are the
smallest continuity primitive — the quarks — layered UNDER the existing
lock/relink/team/brain stack, not replacing it. Each seeker keeps lock on
A body in pixel space; it never knows who. Identity stays late-bound in
the layers above (records.py's design rulings; the lock engine in
people_relink_smoke.py is the immediate consumer).

The three amendments the literature demands of naive per-target seekers
(researched 2026-08-05: MOTRv2 anchored track queries; OC-SORT
observation-centric re-update; Kropfreiter et al. coalescence analysis;
SoccerNet GSR split-then-stitch):

1. RE-ANCHOR ON DETECTIONS EVERY FRAME. Seekers claim detector output
   via one joint assignment (Hungarian) per frame — never free-running
   templates, never greedy per-seeker picks (independent nearest-neighbor
   provably swaps targets in clutter).
2. HONEST EXISTENCE. A seeker that loses its detections COASTS on
   constant velocity for a bounded horizon, then its segment ENDS at the
   last observed instant — coast tails past death are discarded, never
   emitted. On re-anchor within the horizon the coasted gap is rewritten
   as interpolation between the two OBSERVED endpoints (OC-SORT's
   re-update: kill accumulated prediction error instead of trusting it).
3. NO SILENT SWAPS. When two seekers physically coalesce (claimed boxes
   overlap past COALESCE_IOU) identity through the pile is unknowable
   from motion alone, so both segments are SEVERED at coalescence start
   — continuity resumes under fresh segment ids, and the cut is the
   report. Near-ambiguous assignments that stay physically separate keep
   their segment but carry contested=True rows. A forward and a backward
   pass reconcile per-detection claims; rows where the two passes
   disagree carry disputed=True.

Output rows are binding-shaped (ts_ms, track_id, x1..y2) so the existing
relink -> lock -> render chain composes on top unchanged; the extra
contested/coasted/disputed columns are the ambiguity channel for the
layers above. scripts/quark_probe.py runs this over a game dir's
detections stage and measures shot-anchor supply against the observed
8-16% track-supply floor (eval-real candidate_supply, 2026-08-03).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

GATE_BASE = 0.6      # OUTER claim gate in seeker-heights at zero coast
GATE_GROW = 1.2      # ...growing per coasted second (mirrors the lock
                     # engine's KF_GATE_BASE/KF_GATE_GROW convention)
LOG_SIZE_MAX = 0.7   # |log(det_h / seeker_h)| ceiling for a claim
SIZE_COST_W = 0.5    # size term weight in the assignment cost
MARGIN_H = 0.15      # assignment margin in body-heights below which a
                     # claim is contested. Margins are DE-normalized
                     # (cost delta × the seeker's allowance) so the
                     # threshold means the same thing whether physics
                     # is tight or disengaged — normalized margins
                     # compress exactly when allowances inflate, which
                     # marked everything contested during cuts

# PHYSICS LAYER (design decision 08-05: an incorrect identity switch is
# physically impossible motion, so physics can veto it). Bounds are
# CALIBRATED, not invented — physics_envelope.py over 405 clean
# segments / 206k samples, both holdouts (physics_envelope.json):
# ego-compensated speed p99.9 = 2.87 h/s, accel p99.9 = 20.3 h/s².
# Caps sit above the measured tail because clean segments under-sample
# sprints. All physics is in EGO coordinates: the camera-motion
# estimate (median matched displacement, previous frame) is subtracted
# first — a pan makes every body "teleport" in raw pixels.
SPEED_CAP_H_S = 4.0    # ego speed past this is a switch, not a sprint
ACCEL_CAP_H_S2 = 25.0  # implied acceleration ceiling
JITTER_H = 0.12        # detector box noise floor (fraction of height)
CAM_MIN_PAIRS = 4      # matched pairs needed to trust a camera estimate
UNC_INIT_PX = 20.0     # ego-motion uncertainty before any fit
UNC_MAX_PX = 60.0      # cap ≈ 0.6h at typical box height — physics
                       # fully disengaged = the v1 outer-gate behavior
CAM_UNC_W = 3.0        # allowance inflation per px of fit-residual MAD

# EXTREMITY LAYER (design decision 08-05: lock on the extremities as well
# as the body center). When a pose row (pose_stage.py) matches a candidate
# detection, the claim cost adds the residual of the seeker's last
# limb configuration against the detection's — bodies whose centers
# tie still differ at head/wrists/ankles, exactly in piles.
LIMB_W = 0.8           # limb-residual weight in the claim cost
LIMB_KP_MIN = 0.5      # keypoint confidence floor (render convention)
LIMB_IOU_MIN = 0.4     # pose row <-> detection box binding floor
LIMB_MAX_AGE_MS = 500  # stale limb state stops informing claims
LIMB_IDX = (0, 9, 10, 15, 16)  # nose, wrists, ankles — the extremities
COALESCE_IOU = 0.55  # claimed-box IoU that counts as deep overlap...
COALESCE_EXIT_IOU = 0.3  # ...and the band that keeps an interval warm
COALESCE_MIN_MS = 120.0  # deep overlap must persist this long to ARM
                         # (a drive-by brush is contested, not a pile)
COALESCE_COOL_MS = 300.0  # verified separation this long before the
                          # sever fires (alternating detector dropouts
                          # inside a pile must not churn open/sever —
                          # the 08-05 smoke hit 2.5 severs/second)
COAST_MAX_MS = 2000.0  # coast horizon (matches the lock engine)
CONFIRM_N = 3        # claims before a tentative seeker earns a segment
VEL_ALPHA = 0.5      # EMA on velocity from consecutive observed claims
DIM_ALPHA = 0.3      # EMA on box dimensions


@dataclass
class _Row:
    frame_idx: int
    ts_ms: int
    det_idx: int  # -1 for coasted interpolation rows
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    contested: bool
    coasted: bool


@dataclass
class _Seeker:
    sid: int
    cx: float
    cy: float
    w: float
    h: float
    vx: float = 0.0  # px/ms
    vy: float = 0.0
    t_last_obs: float = 0.0
    last_obs_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    n_claims: int = 0
    rows: list[_Row] = field(default_factory=list)
    coast_rows: list[_Row] = field(default_factory=list)
    coalesce_start_ms: float | None = None  # first deep overlap
    coalesce_last_ms: float | None = None   # last overlap of any depth
    coalesce_armed: bool = False
    kp_last: np.ndarray | None = None  # (17, 3) last bound pose
    kp_ts: float = 0.0
    last_margin_h: float = float("inf")  # assignment confidence, h units

    @property
    def tentative(self) -> bool:
        return self.n_claims < CONFIRM_N

    def predict(self, ts_ms: float) -> tuple[float, float]:
        """vx/vy are EGO velocity in the camera-warped frame; the
        engine warps all seeker state per step, so prediction is pure
        body motion."""
        dt = ts_ms - self.t_last_obs
        return self.cx + self.vx * dt, self.cy + self.vy * dt


def _iou(a: tuple, b: tuple) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


class QuarkEngine:
    """Sequential per-frame seeker engine; feed frames in time order.

    Frames arrive as (frame_idx, ts_ms, dets[n, 5]) with det columns
    x1,y1,x2,y2,conf — person class only, caller-filtered. Closed
    segments accumulate in .segments as (sid, [_Row...]); call finish()
    to flush live seekers.
    """

    def __init__(self, pose_lookup=None) -> None:
        """pose_lookup: optional callable ts_ms -> (boxes[N,4],
        kps[N,17,3]) or None — the extremities stage (pose_stage.py).
        Limbs never gate a claim; they reorder near-ties (the
        extremity decision: bodies whose centers tie still differ at
        nose/wrists/ankles, exactly in piles)."""
        self._pose_lookup = pose_lookup
        self._det_kp: dict[int, np.ndarray] = {}  # frame-local j -> (17,3)
        self._next_sid = 1
        self._seekers: list[_Seeker] = []
        self.segments: list[tuple[int, list[_Row]]] = []
        self.n_coalescences = 0
        self.n_occlusion_severs = 0
        self.n_discarded_coast_rows = 0
        # (ts_ms, id(seeker), box) for recent CLAIMED rows — the
        # occlusion-sever check needs other bodies' observed positions
        # across a coast window
        self._recent_claims: list[tuple[float, int, tuple]] = []
        self._prev_dets: tuple[float, np.ndarray] | None = None
        self._cam_unc_px = UNC_INIT_PX  # ego-motion fit uncertainty

    def _update_camera(self, ts_ms: float, dets: np.ndarray) -> None:
        """Ego-motion as a SIMILARITY transform (translation + scale)
        fit robustly on the raw det clouds of consecutive frames, then
        applied to every seeker's state. Deliberately NOT learned from
        claimed pairs (a fast pan makes first claims physics-infeasible
        — bootstrap deadlock, test-pinned) and deliberately not
        translation-only (a broadcast ZOOM moves boxes radially and
        rescales heights; translation-only physics coasted half of Cal,
        08-05 v2 first run). The fit's own residual MAD becomes the
        physics allowance inflation: physics is tight when ego-motion
        is certain and disengages toward the outer gate during cuts."""
        centers = np.stack([(dets[:, 0] + dets[:, 2]) / 2,
                            (dets[:, 1] + dets[:, 3]) / 2], axis=1) \
            if len(dets) else np.zeros((0, 2))
        prev = self._prev_dets
        self._prev_dets = (ts_ms, centers)
        if prev is None or len(centers) < CAM_MIN_PAIRS \
                or len(prev[1]) < CAM_MIN_PAIRS \
                or not 0 < ts_ms - prev[0] <= 100:
            self._cam_unc_px = min(self._cam_unc_px * 1.5, UNC_MAX_PX)
            return
        d2 = ((centers[:, None, :] - prev[1][None, :, :]) ** 2).sum(axis=2)
        nearest = prev[1][np.argmin(d2, axis=1)]
        cen_p = nearest.mean(axis=0)
        cen_c = centers.mean(axis=0)
        r_p = np.linalg.norm(nearest - cen_p, axis=1)
        r_c = np.linalg.norm(centers - cen_c, axis=1)
        ok = r_p > 1.0
        scale = float(np.median(r_c[ok] / r_p[ok])) if ok.sum() >= 3 else 1.0
        scale = min(max(scale, 0.8), 1.25)  # per-frame zoom sanity
        warped = scale * (nearest - cen_p) + cen_c
        resid = np.linalg.norm(centers - warped, axis=1)
        self._cam_unc_px = float(np.median(resid))
        for s in self._seekers:
            s.cx, s.cy = (scale * (s.cx - cen_p[0]) + cen_c[0],
                          scale * (s.cy - cen_p[1]) + cen_c[1])
            x1, y1, x2, y2 = s.last_obs_box
            s.last_obs_box = (scale * (x1 - cen_p[0]) + cen_c[0],
                              scale * (y1 - cen_p[1]) + cen_c[1],
                              scale * (x2 - cen_p[0]) + cen_c[0],
                              scale * (y2 - cen_p[1]) + cen_c[1])
            s.w *= scale
            s.h *= scale
            s.vx *= scale
            s.vy *= scale
            if s.kp_last is not None:
                s.kp_last[:, 0] = scale * (s.kp_last[:, 0] - cen_p[0]) \
                    + cen_c[0]
                s.kp_last[:, 1] = scale * (s.kp_last[:, 1] - cen_p[1]) \
                    + cen_c[1]

    # ---- assignment -------------------------------------------------

    def _cost_matrix(
            self, ts_ms: float, dets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Physics-normalized cost: the residual against the ego-motion
        prediction, in units of what bounded human acceleration allows
        over the elapsed gap. A claim needing an impossible move is
        infeasible; a swapped assignment prices as (near-)impossible in
        the joint optimum — that is the physics layer's veto."""
        n_s, n_d = len(self._seekers), len(dets)
        cost = np.full((n_s, n_d), np.inf)
        allowed_vec = np.ones(n_s)
        if n_d == 0 or n_s == 0:
            return cost, allowed_vec
        dcx = (dets[:, 0] + dets[:, 2]) / 2
        dcy = (dets[:, 1] + dets[:, 3]) / 2
        dh = dets[:, 3] - dets[:, 1]
        for i, s in enumerate(self._seekers):
            px, py = s.predict(ts_ms)
            dt_s = max(ts_ms - s.t_last_obs, 1.0) / 1000.0
            allowed = (JITTER_H
                       + min(0.5 * ACCEL_CAP_H_S2 * dt_s * dt_s,
                             2.0 * SPEED_CAP_H_S * dt_s)
                       + CAM_UNC_W * self._cam_unc_px / max(s.h, 1e-6))
            gate = GATE_BASE + GATE_GROW * dt_s  # outer sanity bound
            resid = np.hypot(dcx - px, dcy - py) / max(s.h, 1e-6)
            size = np.abs(np.log(np.maximum(dh, 1e-6) / max(s.h, 1e-6)))
            feasible = (resid <= allowed) & (resid <= gate) \
                & (size <= LOG_SIZE_MAX)
            row_cost = resid / allowed + SIZE_COST_W * size
            if s.kp_last is not None \
                    and ts_ms - s.kp_ts <= LIMB_MAX_AGE_MS:
                for j, kp in self._det_kp.items():
                    limb = self._limb_residual(s, kp, dcx[j], dcy[j])
                    if limb is not None:
                        row_cost[j] += LIMB_W * limb
            cost[i, feasible] = row_cost[feasible]
            allowed_vec[i] = allowed
        return cost, allowed_vec

    @staticmethod
    def _limb_residual(s: "_Seeker", kp: np.ndarray,
                       dcx: float, dcy: float) -> float | None:
        """Extremity-shape residual in h units: the seeker's last limb
        configuration, translated by the center displacement, against
        the candidate detection's — confident extremities only."""
        last = s.kp_last
        assert last is not None
        lx = (s.last_obs_box[0] + s.last_obs_box[2]) / 2
        ly = (s.last_obs_box[1] + s.last_obs_box[3]) / 2
        deltas = []
        for k in LIMB_IDX:
            if last[k][2] >= LIMB_KP_MIN and kp[k][2] >= LIMB_KP_MIN:
                deltas.append(float(np.hypot(
                    kp[k][0] - (last[k][0] + dcx - lx),
                    kp[k][1] - (last[k][1] + dcy - ly))))
        if len(deltas) < 2:
            return None
        return float(np.mean(deltas)) / max(s.h, 1e-6)

    @staticmethod
    def _assign(cost: np.ndarray,
                allowed_vec: np.ndarray) -> list[tuple[int, int, bool]]:
        """[(seeker_i, det_j, contested)] — Hungarian over feasible
        pairs; contested when either side had a near-equal alternative.
        Margins are DE-normalized back to h units by the seeker's
        allowance: normalized margins compress exactly when allowances
        inflate (cuts), which marked everything contested."""
        from scipy.optimize import linear_sum_assignment

        if not cost.size or not np.isfinite(cost).any():
            return []
        # linear_sum_assignment rejects inf; large-but-finite sentinel,
        # then drop sentinel pairs from the result
        big = 1e6
        rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, big))
        out = []
        for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
            if not np.isfinite(cost[i, j]):
                continue
            alt_i = np.min(np.delete(cost[i, :], j)) if cost.shape[1] > 1 else np.inf
            alt_j = np.min(np.delete(cost[:, j], i)) if cost.shape[0] > 1 else np.inf
            margin_h = float(
                (min(alt_i, alt_j) - cost[i, j]) * allowed_vec[i])
            out.append((i, j, margin_h))
        return out

    # ---- lifecycle --------------------------------------------------

    def _close(self, s: _Seeker) -> None:
        """Segment ends at the last OBSERVED instant: coast tail is
        discarded (honest existence), tentative seekers vanish."""
        self.n_discarded_coast_rows += len(s.coast_rows)
        if not s.tentative and s.rows:
            self.segments.append((s.sid, s.rows))

    def _sever(self, s: _Seeker, at_ms: float) -> None:
        """Coalescence cut: rows before at_ms stay with the old sid;
        the seeker continues physically under a fresh sid. The severed
        boundary IS the swap report — nothing continues silently."""
        keep = [r for r in s.rows if r.ts_ms < at_ms]
        rest = [r for r in s.rows if r.ts_ms >= at_ms]
        if keep and s.n_claims >= CONFIRM_N:
            self.segments.append((s.sid, keep))
        s.sid = self._next_sid
        self._next_sid += 1
        s.rows = rest
        # n_claims is preserved: the physical thread was confirmed, only
        # its identity was cut — discarding observed post-pile rows as
        # "tentative" would trade supply for nothing (the sever already
        # reported the ambiguity)
        s.coalesce_start_ms = None
        s.coalesce_last_ms = None
        s.coalesce_armed = False

    @staticmethod
    def _det_idx(det: np.ndarray, j: int) -> int:
        # column 5, when present, is the ORIGINAL stage det_idx so quark
        # rows join back to the detections stage; else the frame-local j
        return int(det[5]) if det.shape[0] > 5 else j

    def _claim(self, s: _Seeker, frame_idx: int, ts_ms: float,
               det: np.ndarray, det_idx: int, contested: bool) -> None:
        box = (float(det[0]), float(det[1]), float(det[2]), float(det[3]))
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if s.coast_rows:
            coast_start = s.coast_rows[0].ts_ms
            self._rewrite_coast(s, ts_ms, box)
            if self._coast_crossed_body(s, coast_start, ts_ms):
                # the coast path passed THROUGH another body: identity
                # across the gap is unknowable — the claimed-box sever
                # machinery never sees coasting seekers, and this hole
                # was the v2.x swap channel (70% of number-conflicts sat
                # on a coast boundary vs 49% base, 08-05 diagnostic)
                self.n_occlusion_severs += 1
                self._sever(s, float(coast_start))
        dt = ts_ms - s.t_last_obs
        if s.n_claims >= 1 and dt > 0:
            # last_obs_box is already camera-warped to this frame, so
            # the displacement IS ego motion
            nvx = (cx - (s.last_obs_box[0] + s.last_obs_box[2]) / 2) / dt
            nvy = (cy - (s.last_obs_box[1] + s.last_obs_box[3]) / 2) / dt
            s.vx = VEL_ALPHA * nvx + (1 - VEL_ALPHA) * s.vx
            s.vy = VEL_ALPHA * nvy + (1 - VEL_ALPHA) * s.vy
        s.cx, s.cy = cx, cy
        s.w = DIM_ALPHA * (box[2] - box[0]) + (1 - DIM_ALPHA) * s.w
        s.h = DIM_ALPHA * (box[3] - box[1]) + (1 - DIM_ALPHA) * s.h
        s.t_last_obs = ts_ms
        s.last_obs_box = box
        s.n_claims += 1
        s.rows.append(_Row(frame_idx, int(ts_ms), det_idx, *box,
                           float(det[4]), contested, False))

    def _coast_crossed_body(self, s: _Seeker, t0: float,
                            t1: float) -> bool:
        """True when the seeker's (rewritten) coast path deeply
        overlapped the SAME other seeker's OBSERVED boxes for at least
        COALESCE_MIN_MS — the identical time-hysteresis the claimed-box
        sever uses. A single-frame brush past a defender is a drive,
        not an occlusion (the un-hysteresed version severed 45k
        segments on Cal, 6x ByteTrack's fragmentation)."""
        me = id(s)
        window = [(ts, sid, box) for ts, sid, box in self._recent_claims
                  if t0 - 25 <= ts <= t1 + 25 and sid != me]
        if not window:
            return False
        overlap_ts: dict[int, list[float]] = defaultdict(list)
        for r in s.rows:
            if not r.coasted or not t0 <= r.ts_ms <= t1:
                continue
            rb = (r.x1, r.y1, r.x2, r.y2)
            for ts, sid, other in window:
                if abs(ts - r.ts_ms) <= 25 \
                        and _iou(rb, other) >= COALESCE_IOU:
                    overlap_ts[sid].append(float(r.ts_ms))
        return any(max(v) - min(v) >= COALESCE_MIN_MS
                   for v in overlap_ts.values() if len(v) >= 2)

    def _rewrite_coast(self, s: _Seeker, ts_ms: float, box: tuple) -> None:
        """OC-SORT re-update: the coasted gap becomes interpolation
        between the two observed endpoints, not the prediction that
        accumulated error while blind."""
        a, b = s.last_obs_box, box
        t0, t1 = s.t_last_obs, ts_ms
        for r in s.coast_rows:
            f = (r.ts_ms - t0) / max(t1 - t0, 1e-9)
            r.x1, r.y1 = a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
            r.x2, r.y2 = a[2] + f * (b[2] - a[2]), a[3] + f * (b[3] - a[3])
        s.rows.extend(s.coast_rows)
        s.coast_rows = []

    # ---- coalescence ------------------------------------------------

    def _update_coalescence(self, claimed: list[tuple["_Seeker", tuple]],
                            ts_ms: float) -> None:
        """claimed: (seeker, claimed box) this frame, confirmed only.
        Deep overlap sustained COALESCE_MIN_MS ARMS a coalescence; the
        sever fires only after COALESCE_COOL_MS of verified separation
        (space hysteresis alone churned 2.5 severs/s on real footage —
        detector dropouts inside a pile alternate which member is
        claimed on a given frame)."""
        for a in range(len(claimed)):
            for b in range(a + 1, len(claimed)):
                s_a, box_a = claimed[a]
                s_b, box_b = claimed[b]
                iou = _iou(box_a, box_b)
                if iou >= COALESCE_IOU:
                    # boxes merged — but if BOTH members' assignment
                    # margins stay high (limb evidence separates them,
                    # the extremity decision), identity was never
                    # actually ambiguous and the interval must not arm
                    ambiguous = (s_a.last_margin_h < MARGIN_H
                                 or s_b.last_margin_h < MARGIN_H)
                    for s in (s_a, s_b):
                        if s.coalesce_start_ms is None:
                            s.coalesce_start_ms = ts_ms
                        s.coalesce_last_ms = ts_ms
                        if ambiguous and ts_ms - s.coalesce_start_ms \
                                >= COALESCE_MIN_MS:
                            s.coalesce_armed = True
                elif iou >= COALESCE_EXIT_IOU:
                    for s in (s_a, s_b):
                        if s.coalesce_start_ms is not None:
                            s.coalesce_last_ms = ts_ms
        for s, _ in claimed:
            if s.coalesce_start_ms is None:
                continue
            # an open coalescence marks its rows contested — the
            # interval is reported even when the cut resolves it later
            if s.rows:
                s.rows[-1].contested = True
            if ts_ms - (s.coalesce_last_ms or ts_ms) > COALESCE_COOL_MS:
                if s.coalesce_armed:
                    self.n_coalescences += 1
                    self._sever(s, s.coalesce_start_ms)
                else:  # a brush that never armed: no cut, just reset
                    s.coalesce_start_ms = None
                    s.coalesce_last_ms = None

    # ---- main step --------------------------------------------------

    def _bind_pose(self, ts_ms: float, dets: np.ndarray) -> None:
        self._det_kp = {}
        if self._pose_lookup is None or not len(dets):
            return
        hit = self._pose_lookup(ts_ms)
        if hit is None:
            return
        pboxes, pkps = hit
        for j in range(len(dets)):
            best, bi = 0.0, -1
            for p in range(len(pboxes)):
                iou = _iou(tuple(dets[j][:4]), tuple(pboxes[p]))
                if iou > best:
                    best, bi = iou, p
            if best >= LIMB_IOU_MIN:
                self._det_kp[j] = np.array(pkps[bi], dtype=np.float64)

    def step(self, frame_idx: int, ts_ms: float, dets: np.ndarray) -> None:
        self._update_camera(ts_ms, dets)
        self._bind_pose(ts_ms, dets)
        cost, allowed_vec = self._cost_matrix(ts_ms, dets)
        pairs = self._assign(cost, allowed_vec)
        claimed_seekers = {i for i, _, _ in pairs}
        claimed_dets = {j for _, j, _ in pairs}
        claimed: list[tuple[_Seeker, tuple]] = []
        for i, j, margin_h in pairs:
            s = self._seekers[i]
            self._claim(s, frame_idx, ts_ms, dets[j],
                        self._det_idx(dets[j], j), margin_h < MARGIN_H)
            s.last_margin_h = margin_h
            if j in self._det_kp:
                s.kp_last = self._det_kp[j].copy()
                s.kp_ts = ts_ms
            self._recent_claims.append((ts_ms, id(s), s.last_obs_box))
            if not s.tentative:
                claimed.append((s, s.last_obs_box))
        cutoff = ts_ms - COAST_MAX_MS - 100
        while self._recent_claims and self._recent_claims[0][0] < cutoff:
            self._recent_claims.pop(0)

        survivors: list[_Seeker] = []
        for i, s in enumerate(self._seekers):
            if i in claimed_seekers:
                survivors.append(s)
                continue
            if s.tentative:  # a tentative that misses a frame was a FP
                continue
            coast_ms = ts_ms - s.t_last_obs
            if coast_ms > COAST_MAX_MS:
                self._close(s)
                continue
            px, py = s.predict(ts_ms)
            s.coast_rows.append(_Row(
                frame_idx, int(ts_ms), -1,
                px - s.w / 2, py - s.h / 2, px + s.w / 2, py + s.h / 2,
                0.0, False, True))
            survivors.append(s)
        self._seekers = survivors

        self._update_coalescence(claimed, ts_ms)

        for j in range(len(dets)):
            if j in claimed_dets:
                continue
            box = dets[j]
            s = _Seeker(
                sid=self._next_sid,
                cx=float(box[0] + box[2]) / 2, cy=float(box[1] + box[3]) / 2,
                w=float(box[2] - box[0]), h=float(box[3] - box[1]),
                t_last_obs=ts_ms,
                last_obs_box=(float(box[0]), float(box[1]),
                              float(box[2]), float(box[3])),
            )
            self._next_sid += 1
            s.n_claims = 1
            s.rows.append(_Row(frame_idx, int(ts_ms),
                               self._det_idx(box, j), *s.last_obs_box,
                               float(box[4]), False, False))
            self._seekers.append(s)

    def finish(self) -> None:
        for s in self._seekers:
            self._close(s)
        self._seekers = []


# ---- bidirectional run + reconciliation -----------------------------


def run_pass(frames: list[tuple[int, float, np.ndarray]],
             pose_lookup=None) -> "QuarkEngine":
    """One directional pass. frames = [(frame_idx, ts_ms, dets[n,5])] in
    the order to be consumed; caller negates ts for the backward pass."""
    eng = QuarkEngine(pose_lookup=pose_lookup)
    for frame_idx, ts_ms, dets in frames:
        eng.step(frame_idx, ts_ms, dets)
    eng.finish()
    return eng


def run_bidirectional(
    frames: list[tuple[int, float, np.ndarray]],
    pose_lookup=None,
) -> tuple[list[dict], dict]:
    """Forward + backward pass, reconciled on per-detection claims.

    The claimed detection keys (frame_idx, det_idx) are the common
    currency between the passes: a forward segment and a backward
    segment that claim mostly the same detections are the same physical
    thread seen from both directions. Rows whose forward and backward
    owners are NOT mutual best partners carry disputed=True — the
    smoothing residue the layers above must treat as ambiguous.

    Returns (rows, stats): rows are output dicts keyed for the
    binding-shaped quark_binding stage, forward segment ids.
    """
    fwd = run_pass(frames, pose_lookup)
    bwd_lookup = None if pose_lookup is None \
        else (lambda t: pose_lookup(-t))  # bwd pass runs in negated time
    bwd = run_pass([(fi, -ts, dets) for fi, ts, dets in reversed(frames)],
                   bwd_lookup)

    owner_f: dict[tuple[int, int], int] = {}
    for sid, rows in fwd.segments:
        for r in rows:
            if r.det_idx >= 0:
                owner_f[(r.frame_idx, r.det_idx)] = sid
    owner_b: dict[tuple[int, int], int] = {}
    for sid, rows in bwd.segments:
        for r in rows:
            if r.det_idx >= 0:
                owner_b[(r.frame_idx, r.det_idx)] = sid

    from collections import Counter

    co: dict[int, Counter] = {}
    co_b: dict[int, Counter] = {}
    for key, sf in owner_f.items():
        sb = owner_b.get(key)
        if sb is None:
            continue
        co.setdefault(sf, Counter())[sb] += 1
        co_b.setdefault(sb, Counter())[sf] += 1
    partner_f = {sf: c.most_common(1)[0][0] for sf, c in co.items()}
    partner_b = {sb: c.most_common(1)[0][0] for sb, c in co_b.items()}

    rows_out: list[dict] = []
    n_disputed = 0
    for sid, rows in fwd.segments:
        for r in rows:
            disputed = False
            if r.det_idx >= 0:
                sb = owner_b.get((r.frame_idx, r.det_idx))
                disputed = (
                    sb is None
                    or partner_f.get(sid) != sb
                    or partner_b.get(sb) != sid
                )
            n_disputed += int(disputed)
            rows_out.append({
                "frame_idx": r.frame_idx, "ts_ms": r.ts_ms,
                "det_idx": r.det_idx, "track_id": sid,
                "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
                "conf": r.conf, "contested": r.contested,
                "coasted": r.coasted, "disputed": disputed,
            })
    rows_out.sort(key=lambda r: (r["ts_ms"], r["track_id"]))
    stats = {
        "segments_fwd": len(fwd.segments),
        "segments_bwd": len(bwd.segments),
        "coalescences_fwd": fwd.n_coalescences,
        "coalescences_bwd": bwd.n_coalescences,
        "occlusion_severs_fwd": fwd.n_occlusion_severs,
        "occlusion_severs_bwd": bwd.n_occlusion_severs,
        "rows": len(rows_out),
        "rows_observed": sum(1 for r in rows_out if r["det_idx"] >= 0),
        "rows_coasted": sum(1 for r in rows_out if r["coasted"]),
        "rows_contested": sum(1 for r in rows_out if r["contested"]),
        "rows_disputed": n_disputed,
        "discarded_coast_rows_fwd": fwd.n_discarded_coast_rows,
    }
    return rows_out, stats
