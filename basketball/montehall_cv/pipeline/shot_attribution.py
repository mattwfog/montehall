"""Shot attribution stage: detected shot windows -> ranked shooter candidates.

The composed chain, promoted from the 08-17/08-24 probe scripts
(shot_sheet_probe -> flight_fit_probe -> teamfilter_rank) into ONE
first-class stage so the shipped path and the measured path are the same
path. Per shot_events attempt window:

  1. candidate bodies from the binding stage (quark_binding when present —
     shot-window supply .97-.99 vs ByteTrack's .08-.16; falls back to
     track_binding),
  2. one contact sheet per candidate (pooled tallest crops across the
     +-2s window), one cached VLM read each, roster-vocabulary optional,
  3. team layer: torso-luminance split + shooting-team-by-possession
     (majority color of the release-window ball-proximate bodies),
  4. rank: launch-point distance when a WASB flight fit exists, else
     release-window ball proximity; a learned ranker (logistic over the
     FEATURES vector, trained on the PBP-labeled corpus) overrides the
     heuristic order when weights are supplied.

Output: <out>/<job_id>/shot_attribution/ — one row per (event, candidate)
with the feature vector, plus picked=True on the top candidate carrying a
read. run_boxscore consumes the picks for shooter naming, superseding the
event_identity read path (2026-08-25 ruling: structural fix A).

Measured lineage (HS clip, 8 hand anchors): ball ranker .125 -> release
.25 -> launch .50-of-fitted -> team-filtered .50, recall 1.0 underneath.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from montehall_cv.pipeline.contact_sheet import CROPS_PER_SHEET, build_sheet, _read_sheet
from montehall_cv.pipeline.flight import (
    CLS_RIM,
    DetIndex,
    RIM_TRUE_FT,
    flight_fits_for_anchors,
)
from montehall_cv.pipeline.jersey import MIN_BBOX_HEIGHT_PX
from montehall_cv.pipeline.run_jersey import _collect_crops
from montehall_cv.store.artifacts import (
    ArtifactWriter,
    read_stage,
    stage_complete,
)
from montehall_cv.store.vlm_cache import VlmCache

import pyarrow as pa

CLS_BALL = 1            # DetClass space (⚠ never 2 — see 08-03 retraction)
WINDOW_MS = 2000        # +-2s candidate window around the anchor
SPINE_MATCH_MS = 2000   # pad around the window SPAN within which a
                        # ball-spine shot claims the event (span-keyed
                        # 2026-08-25: anchor-keyed ±2s missed a spine
                        # shot sitting INSIDE a 5.6s window)
RELEASE_MS = (-2500, -800)  # shot_events anchors sit at rim contact;
                            # the release happened this far earlier
LAUNCH_MATCH_MS = 200   # candidate boxes matched to the launch instant
RIM_NEAR_LAUNCH_MS = 600
INHERIT_SHARE_MIN = 0.6   # game-identity fallback bar: the probe's
                          # measured perfect-precision point (2026-08-25:
                          # inherited@0.6 named 2/2 correct; @0.5 took a
                          # wrong '5')
SEGMENT_SLACK_MS = 2000   # person lookup: nearest relink segment this close
CLUSTER_COLOR_MIN_VOTES = 10  # (team_cluster -> luminance color) map needs
                              # this many agreeing event-local pairs
END_TEAM_MIN_VOTES = 3    # events agreeing before a court_end maps to a
                          # shooting-team color (offense-direction prior)
W_SHEET = 1.0             # person evidence-channel weights (probe's)
W_CONTACT = 1.0
W_OCR = 1.0
MIN_PICK_READ_CONF = 0.6  # the repo's standing read-accept bar
                          # (contact_reads accept / relink READ_CONF_MIN):
                          # a sub-.6 read never takes the pick — v3 e2e had
                          # a 0.0-conf garbage read outrank real candidates
BRIDGE_IOU_MIN = 0.6      # quark row <-> localized ByteTrack row match
                          # (extract_rank_features' measured bridge bar)
OCR_ACCEPT_PROB = 0.7     # a jersey-OCR posterior names the spine body
                          # only when this decisive (posterior already
                          # requires >=2 agreeing PARSeq reads upstream)

FEATURES = (
    "release_prox_px",
    "whole_prox_px",
    "launch_dist_ft",
    "team_match",
    "n_obs",
    "max_h_px",
)

SHOT_ATTRIBUTION_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("event_id", pa.int32()),
        pa.field("ts_anchor_ms", pa.int64()),
        pa.field("track_id", pa.int64()),
        pa.field("rank", pa.int32()),
        pa.field("picked", pa.bool_()),
        pa.field("number", pa.string(), nullable=True),
        pa.field("read_confidence", pa.float32(), nullable=True),
        pa.field("team", pa.string(), nullable=True),
        pa.field("shooting_team", pa.string(), nullable=True),
        pa.field("release_prox_px", pa.float32(), nullable=True),
        pa.field("whole_prox_px", pa.float32(), nullable=True),
        pa.field("launch_dist_ft", pa.float32(), nullable=True),
        pa.field("team_match", pa.bool_(), nullable=True),
        pa.field("person_id", pa.int32(), nullable=True),
        pa.field("n_obs", pa.int32()),
        pa.field("max_h_px", pa.float32()),
        pa.field("method", pa.string()),  # launch | release | learned
    ]
)


# ---- binding / ball helpers (canonical home; probe scripts import these) --

def _iou(a: tuple, b: tuple) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def localized_person_index(game_dir: Path) -> dict[int, list[tuple[int, tuple]]]:
    """frame_idx -> [(bytetrack_id, bbox)] for person rows."""
    from montehall_cv.store.records import DetClass

    t = read_stage(game_dir / "localized")
    out: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
    person = int(DetClass.PERSON)
    for row in t.to_pylist():
        if row["cls"] == person and row["track_id"] is not None:
            out[int(row["frame_idx"])].append((
                int(row["track_id"]),
                (row["x1"], row["y1"], row["x2"], row["y2"]),
            ))
    return out


def _identity_posteriors(game_dir: Path) -> dict[int, dict[str, float]]:
    """bytetrack_id -> {jersey: prob} from the PARSeq OCR vote channel."""
    if not stage_complete(game_dir / "identity"):
        return {}
    out: dict[int, dict[str, float]] = defaultdict(dict)
    for row in read_stage(game_dir / "identity").to_pylist():
        if row["bound_at_stage"] == "ocr_vote" and row["candidate"]:
            out[int(row["track_id"])][str(row["candidate"])] = float(
                row["prob"] or 0.0)
    return dict(out)


def _ocr_number_for(rows: list[tuple], loc_idx: dict,
                    posteriors: dict[int, dict[str, float]],
                    roster: set[str]) -> tuple[str, float] | None:
    """2nd identity channel: the spine body's quark rows bridged (IoU) to
    ByteTrack ids, their OCR posteriors pooled weighted by match count.
    Returns (number, prob) only when decisive — this channel exists to
    rescue refuted/absent sheet reads on the RIGHT body, never to guess."""
    match_count: dict[int, int] = defaultdict(int)
    for row in rows:
        for bt_id, bbox in loc_idx.get(row[1], ()):
            if bt_id in posteriors and _iou(row[2], bbox) >= BRIDGE_IOU_MIN:
                match_count[bt_id] += 1
    if not match_count:
        return None
    pooled: dict[str, float] = defaultdict(float)
    total = sum(match_count.values())
    for bt_id, n in match_count.items():
        for number, prob in posteriors[bt_id].items():
            pooled[number] += prob * (n / total)
    if roster:
        pooled = {n: p for n, p in pooled.items() if n in roster}
    if not pooled:
        return None
    number, prob = max(pooled.items(), key=lambda kv: kv[1])
    return (number, prob) if prob >= OCR_ACCEPT_PROB else None


def person_index(segments: list) -> dict[int, list[tuple[int, int, int]]]:
    """quark_tid -> [(t0, t1, person_id)] sorted by t0 (relink json v3/v4
    'segments' rows). Canonical home (2026-08-25) — game_identity_probe
    imports back."""
    by_tid: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for tid, t0, t1, pid in segments:
        by_tid[int(tid)].append((int(t0), int(t1), int(pid)))
    for spans in by_tid.values():
        spans.sort()
    return dict(by_tid)


def person_at(pidx: dict, tid: int, ts_ms: int) -> int | None:
    """The relink person owning quark body `tid` at `ts_ms`, with
    SEGMENT_SLACK_MS tolerance to the nearest segment edge."""
    from bisect import bisect_right

    spans = pidx.get(tid)
    if not spans:
        return None
    starts = [s[0] for s in spans]
    i = bisect_right(starts, ts_ms) - 1
    best: tuple[int, int] | None = None  # (distance, pid)
    for j in (i, i + 1):
        if 0 <= j < len(spans):
            t0, t1, pid = spans[j]
            d = 0 if t0 <= ts_ms <= t1 else min(abs(ts_ms - t0),
                                                abs(ts_ms - t1))
            if d <= SEGMENT_SLACK_MS and (best is None or d < best[0]):
                best = (d, pid)
    return best[1] if best else None


def bt_to_quark_person(job_dir: Path, binding_stage: Path,
                       pidx: dict) -> dict[int, int]:
    """ByteTrack tid -> majority relink person, via per-frame IoU between
    localized person rows and quark binding rows (the standing bridge)."""
    loc_idx = localized_person_index(job_dir)
    b = read_stage(binding_stage)
    cols = {c: b.column(c).to_numpy(zero_copy_only=False)
            for c in ("frame_idx", "ts_ms", "track_id",
                      "x1", "y1", "x2", "y2")}
    if "coasted" in b.column_names:
        keep = ~b.column("coasted").to_numpy(zero_copy_only=False)
        cols = {k: v[keep] for k, v in cols.items()}
    votes: dict[int, Counter] = defaultdict(Counter)
    for i in range(len(cols["frame_idx"])):
        fi = int(cols["frame_idx"][i])
        others = loc_idx.get(fi)
        if not others:
            continue
        qbox = (cols["x1"][i], cols["y1"][i], cols["x2"][i], cols["y2"][i])
        pid = person_at(pidx, int(cols["track_id"][i]),
                        int(cols["ts_ms"][i]))
        if pid is None:
            continue
        for bt_id, bbox in others:
            if _iou(qbox, bbox) >= BRIDGE_IOU_MIN:
                votes[bt_id][pid] += 1
    return {bt: c.most_common(1)[0][0] for bt, c in votes.items()}


def person_pool(sheet_rows: list[tuple], contact_rows: list[tuple],
                ocr_rows: list[tuple],
                roster: set[str]) -> dict[int, tuple[str, float]]:
    """Game-level identity posteriors: pid -> (number, share). Rows are
    pre-bridged (pid, number, weight) triples per channel; off-roster
    numbers are dropped, never remapped (probe contract, 2026-08-25)."""
    pool: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for w, rows in ((W_SHEET, sheet_rows), (W_CONTACT, contact_rows),
                    (W_OCR, ocr_rows)):
        for pid, num, weight in rows:
            if not num:
                continue
            if roster and str(num) not in roster:
                continue  # empty roster = unconstrained (2026-08-25:
                # the roster list is out of the runtime until ruled in)
            pool[pid][str(num)] += w * float(weight)
    out: dict[int, tuple[str, float]] = {}
    for pid, votes in pool.items():
        total = sum(votes.values())
        if total > 0:
            num, mass = max(votes.items(), key=lambda kv: kv[1])
            out[pid] = (num, mass / total)
    return out


def cluster_colors(pairs: list[tuple[int, str]]) -> dict[int, str]:
    """relink team cluster -> luminance color ('white'/'dark') by majority
    vote over event-local (cluster, color) observations; a cluster maps
    only past CLUSTER_COLOR_MIN_VOTES."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for cluster, color in pairs:
        votes[cluster][color] += 1
    out: dict[int, str] = {}
    for cluster, c in votes.items():
        color, n = c.most_common(1)[0]
        if n >= CLUSTER_COLOR_MIN_VOTES and n / sum(c.values()) > 0.5:
            out[cluster] = color
    return out


def _match_spine_shot(spine: dict[int, int], ts_start_ms: int,
                      ts_end_ms: int,
                      feats: dict) -> tuple[int | None, int | None]:
    """Span-keyed spine match (2026-08-25 coverage diagnostic).

    The VLM window is a ball-near-rim SPAN whose ts_start anchor can sit
    seconds before the shot the window resolves on (HS-clip w14: anchor
    124866, spine shot 130400 INSIDE the span [124866, 130500] — the
    anchor-keyed ±2s match never engaged and the spine's correct claim
    fell to the release ranker). Spine shots are matched against the
    span padded by SPINE_MATCH_MS; inside the span distance is 0, so the
    tie-break decides a rim scramble: the LATEST in-span shot — the
    attempt whose outcome ends the ball-near-rim window."""
    best = None
    for ts, tid in spine.items():
        if tid not in feats:
            continue
        d = max(ts_start_ms - ts, 0, ts - ts_end_ms)
        if d > SPINE_MATCH_MS:
            continue
        if best is None or d < best[0] or (d == best[0] and ts > best[1]):
            best = (d, ts, tid)
    return (best[1], best[2]) if best else (None, None)


def _team_alternate(contacts: list[dict], arrival_ts: int,
                    feats: dict, color_of: dict, team: str) -> int | None:
    """Latest shooting-team contact whose flight to the arrival is
    plausible — the body the spine SHOULD have bound (ev-9 class)."""
    from montehall_cv.pipeline.ball_track import (
        MIN_FLIGHT_MS,
        RELEASE_RIM_MAX_MS,
    )

    best: tuple[int, int] | None = None
    for c in contacts:
        end = c["ts_end_ms"] or c["ts_ms"]
        if not (MIN_FLIGHT_MS <= arrival_ts - end <= RELEASE_RIM_MAX_MS):
            continue
        tid = int(c["track_id"])
        if tid not in feats or color_of.get(tid) != team:
            continue
        if best is None or end > best[0]:
            best = (end, tid)
    return best[1] if best else None


def load_binding(stage: Path) -> dict:
    """Observed rows only — a coasted box has no body under it to read."""
    t = read_stage(stage)
    cols = {c: t.column(c).to_numpy(zero_copy_only=False)
            for c in ("frame_idx", "ts_ms", "track_id", "x1", "y1", "x2", "y2")}
    if "coasted" in t.column_names:
        keep = ~t.column("coasted").to_numpy(zero_copy_only=False)
        cols = {k: v[keep] for k, v in cols.items()}
    return cols


def window_candidates(b: dict, lo_ms: int, hi_ms: int) -> dict[int, list[tuple]]:
    """track_id -> [(height, frame_idx, bbox, ts_ms)] readable in window."""
    win = (b["ts_ms"] >= lo_ms) & (b["ts_ms"] < hi_ms)
    h = b["y2"][win] - b["y1"][win]
    tall = h >= MIN_BBOX_HEIGHT_PX
    idx = np.nonzero(win)[0][tall]
    out: dict[int, list[tuple]] = defaultdict(list)
    for i in idx:
        out[int(b["track_id"][i])].append((
            float(b["y2"][i] - b["y1"][i]),
            int(b["frame_idx"][i]),
            (float(b["x1"][i]), float(b["y1"][i]),
             float(b["x2"][i]), float(b["y2"][i])),
            int(b["ts_ms"][i]),
        ))
    return out


def ball_dets(game_dir: Path) -> dict[int, list[tuple[float, float]]]:
    """frame_idx -> [(ball_cx, ball_cy)] in PIXELS. Homography-free: the
    court stage is dead in 84-92% of shot windows, but the ball is
    frame-detected in every one of them (2026-08-03 corrected probe)."""
    t = read_stage(game_dir / "detections")
    cls = t.column("cls").to_numpy(zero_copy_only=False)
    ball = cls == CLS_BALL
    fi = t.column("frame_idx").to_numpy(zero_copy_only=False)[ball]
    cx = (t.column("x1").to_numpy(zero_copy_only=False)[ball]
          + t.column("x2").to_numpy(zero_copy_only=False)[ball]) / 2.0
    cy = (t.column("y1").to_numpy(zero_copy_only=False)[ball]
          + t.column("y2").to_numpy(zero_copy_only=False)[ball]) / 2.0
    out: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for f, x, y in zip(fi, cx, cy):
        out[int(f)].append((float(x), float(y)))
    return out


def box_to_point(bbox: tuple, px: float, py: float) -> float:
    """0 inside the box, else pixel distance to its nearest edge."""
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float((dx * dx + dy * dy) ** 0.5)


def ball_proximity(rows: list[tuple], balls: dict) -> float:
    """Closest this body ever came to a detected ball in these rows."""
    best = float("inf")
    for row in rows:
        for px, py in balls.get(row[1], ()):
            best = min(best, box_to_point(row[2], px, py))
    return best


# ---- feature extraction (shared with the corpus ranker trainer) ----------

def candidate_features(
    cands: dict[int, list[tuple]],
    balls: dict,
    a_ms: int,
    fit: dict | None,
    det: DetIndex | None,
    release_ms: tuple[int, int] = RELEASE_MS,
) -> dict[int, dict]:
    """Per-candidate geometry features for one anchor. Read/team features
    are attached by the caller when they exist — geometry never needs the
    VLM, which is what lets the corpus trainer reuse this exactly."""
    launch = _launch_pixel_frame(fit, det) if fit and det is not None else None
    out: dict[int, dict] = {}
    for tid, rows in cands.items():
        rel_rows = [r for r in rows
                    if a_ms + release_ms[0] <= r[3] < a_ms + release_ms[1]]
        f: dict = {
            "release_prox_px": ball_proximity(rel_rows, balls)
            if rel_rows else float("inf"),
            "whole_prox_px": ball_proximity(rows, balls),
            "launch_dist_ft": None,
            "n_obs": len(rows),
            "max_h_px": max(r[0] for r in rows),
        }
        if launch is not None:
            lt_ms, r_cx, r_cy, fpp, lx, ly = launch
            near = [r for r in rows if abs(r[3] - lt_ms) <= LAUNCH_MATCH_MS]
            if near:
                bx = np.array([(r[2][0] + r[2][2]) / 2 for r in near])
                bb = np.array([r[2][3] for r in near])
                d = np.hypot((bx - r_cx) * fpp - lx, (bb - r_cy) * fpp - ly)
                f["launch_dist_ft"] = float(d.min())
        out[tid] = f
    return out


def _launch_pixel_frame(fit: dict, det: DetIndex):
    """Rim pixel frame at the launch instant -> (lt_ms, rim_cx, rim_cy,
    ft_per_px, launch_x_ft, launch_y_ft), or None without a rim det."""
    lt_ms = fit["t_launch"] * 1000.0
    r_ts, r_cx, r_cy, r_w, _ = det.window(
        lt_ms - RIM_NEAR_LAUNCH_MS, lt_ms + RIM_NEAR_LAUNCH_MS, CLS_RIM)
    if not r_ts.size:
        return None
    k = int(np.argmin(np.abs(r_ts - lt_ms)))
    fpp = RIM_TRUE_FT / max(float(r_w[k]), 1e-6)
    lx, ly = fit["launch_ft"]
    return (lt_ms, float(r_cx[k]), float(r_cy[k]), fpp, float(lx), float(ly))


# ---- team layer -----------------------------------------------------------

def _torso_luminance(crop: np.ndarray) -> float:
    torso = crop[: max(int(crop.shape[0] * 0.45), 1)]
    return float(np.median(torso.astype(np.float32).mean(axis=2)))


def _team_split(lums: dict[tuple, list[float]]) -> dict[tuple, str]:
    if not lums:
        return {}
    med = {k: float(np.median(v)) for k, v in lums.items()}
    split = float(np.median(sorted(med.values())))
    return {k: ("white" if v >= split else "dark") for k, v in med.items()}


def _shooting_team(features: dict[int, dict], color_of: dict) -> str | None:
    """Majority color of the 3 most release-ball-proximate bodies."""
    by_rel = sorted(features.items(), key=lambda kv: kv[1]["release_prox_px"])
    top = [color_of.get(tid) for tid, _f in by_rel[:3]]
    top = [c for c in top if c]
    return max(set(top), key=top.count) if top else None


def end_team_map(votes: list[tuple[str, str]]) -> dict[str, str]:
    """OFFENSE-DIRECTION PRIOR (design decision 2026-08-25: know which team
    attacks which end): within a
    period every shot at one rim belongs to one team, so the noisy
    per-event luminance votes pool globally per court_end. A mapping
    needs END_TEAM_MIN_VOTES agreeing events and a majority; both ends
    resolving to the SAME color contradicts the structure (one team
    attacks each rim) — return {} and let per-event votes stand."""
    by_end: dict[str, Counter] = defaultdict(Counter)
    for end, color in votes:
        by_end[end][color] += 1
    out: dict[str, str] = {}
    for end, c in by_end.items():
        color, n = c.most_common(1)[0]
        if n >= END_TEAM_MIN_VOTES and n / sum(c.values()) > 0.5:
            out[end] = color
    if len(out) == 2 and len(set(out.values())) == 1:
        return {}
    return out


# ---- ranking --------------------------------------------------------------

def _load_ranker(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    model = json.loads(path.read_text())
    return model if "weights" in model else None


def _learned_score(f: dict, model: dict) -> float:
    z = model.get("bias", 0.0)
    for name, w in model["weights"].items():
        v = f.get(name)
        if name == "team_match":
            v = {True: 1.0, False: 0.0, None: 0.5}[v]
        if v is None or v == float("inf"):
            v = model.get("impute", {}).get(name, 0.0)
        mean = model.get("mean", {}).get(name, 0.0)
        std = model.get("std", {}).get(name, 1.0) or 1.0
        z += w * ((float(v) - mean) / std)
    return 1.0 / (1.0 + float(np.exp(-np.clip(z, -30.0, 30.0))))


def rank_candidates(features: dict[int, dict], model: dict | None) -> list[int]:
    """Candidate order, best first. Heuristic (the measured .50 arm):
    team-mismatches last; launch distance when known, else release
    proximity. Learned model replaces the ordering when supplied."""
    if model is not None:
        return sorted(features,
                      key=lambda tid: -_learned_score(features[tid], model))

    def key(tid: int) -> tuple:
        f = features[tid]
        mismatch = f.get("team_match") is False
        ld = f["launch_dist_ft"]
        return (
            mismatch,
            0 if ld is not None else 1,
            ld if ld is not None else f["release_prox_px"],
            # geometry ties (many bodies sit at prox 0 in a scramble):
            # highest read confidence wins — shot_sheet_probe._rank_key's
            # measured composition (rank, then -confidence)
            -(f.get("read_confidence") or 0.0),
        )
    return sorted(features, key=key)


# ---- stage ----------------------------------------------------------------

def run(video: Path, out_root: Path, job_id: str,
        roster_numbers: set[str] | None = None,
        wasb_root: Path | None = None,
        wasb_weights: Path | None = None,
        binding_stage: Path | None = None,
        ranker_weights: Path | None = None,
        max_candidates: int = 0,
        flight_min_inliers: int = 6,
        flight_span_min: float = 0.4,
        relink_map: Path | None = None) -> dict:
    job_dir = out_root / job_id
    if stage_complete(job_dir / "shot_attribution"):
        return {"job_id": job_id, "skipped": True,
                "reason": "stage already complete"}
    if not stage_complete(job_dir / "shot_events"):
        return {"job_id": job_id, "skipped": True,
                "reason": "no shot_events stage"}
    started = time.monotonic()

    stage = binding_stage
    if stage is None:
        for name in ("quark_binding", "track_binding"):
            if stage_complete(job_dir / name):
                stage = job_dir / name
                break
    if stage is None:
        return {"job_id": job_id, "skipped": True,
                "reason": "no binding stage (quark_binding/track_binding)"}

    events = [e for e in read_stage(job_dir / "shot_events").to_pylist()
              if e["verdict_attempt"]]
    binding = load_binding(stage)
    balls = ball_dets(job_dir)
    det = DetIndex(read_stage(job_dir / "detections"))
    model = _load_ranker(ranker_weights)

    # THE BALL SPINE (ball-first ruling, 2026-08-03/25): where the
    # continuous track derived a release->rim shot, the release holder IS
    # the shooter — no ranking. Rankers only cover spine gaps.
    from montehall_cv.pipeline.ball_track import spine_contacts, spine_shooters

    spine = spine_shooters(job_dir)  # rim-arrival ts_ms -> quark track_id
    contacts = spine_contacts(job_dir)  # team-gate alternates (spine v3)
    # 2nd identity channel (spine v3): PARSeq jersey posteriors bridged
    # onto quark bodies — consulted only when the sheet channel fails on
    # a spine-named body
    ocr_posteriors = _identity_posteriors(job_dir)
    loc_idx = localized_person_index(job_dir) if ocr_posteriors else {}

    # Rank anchor = window START (first ball-in-rim-region). Measured on
    # the HS truth set: hand rim-contact labels sit ~250-350ms after
    # ts_start, while ts_end drifts up to +3s into the rebound — the
    # end-anchored arm scored WORSE (fits 2->1, e2e named 1/6->0/6,
    # 2026-08-25 A/B) and is retracted.
    anchor_of = {e["event_id"]: e["ts_start_ms"] for e in events}

    # flight fits once for every attempt anchor (GPU; optional)
    fits: dict[int, dict | None] = {}
    if wasb_weights is not None:
        rows = flight_fits_for_anchors(
            video, sorted(set(anchor_of.values())), det,
            wasb_root or Path("/work/wasb"), wasb_weights,
            min_inliers=flight_min_inliers, span_min=flight_span_min,
            log=lambda m: print(f"SHOT-ATTR {m}", flush=True))
        fits = {r["anchor_ms"]: r.get("fit") for r in rows}

    # one decode pass: sheet crops for every (event, candidate)
    plan: dict[int, list[tuple]] = defaultdict(list)
    cand_rows: dict[tuple[int, int], list[tuple]] = {}
    for e in events:
        a_ms = anchor_of[e["event_id"]]
        cands = window_candidates(binding, e["ts_start_ms"] - WINDOW_MS,
                                  a_ms + WINDOW_MS)
        if max_candidates > 0:
            keep = sorted(cands.items(),
                          key=lambda kv: ball_proximity(kv[1], balls))
            cands = dict(keep[:max_candidates])
        for tid, rows in cands.items():
            rows.sort(key=lambda r: -r[0])  # tallest first
            cand_rows[(e["event_id"], tid)] = rows
            for row in rows[:CROPS_PER_SHEET]:
                plan[row[1]].append(((e["event_id"], tid), row[2]))

    crops, meta = _collect_crops(video, dict(plan))
    by_key: dict[tuple, list] = defaultdict(list)
    for crop, (key, _frame_idx) in zip(crops, meta):
        by_key[key].append(crop)

    import anthropic
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)
    cache = VlmCache(job_dir / "_vlm_cache" / "shot_attribution.jsonl")
    roster = {str(n) for n in roster_numbers} if roster_numbers else set()
    reads: dict[tuple, dict] = {}
    lums: dict[tuple, list[float]] = defaultdict(list)
    for key, crop_list in sorted(by_key.items()):
        reads[key] = _read_sheet(client, build_sheet(crop_list), roster, cache)
        for crop in crop_list[:2]:
            lums[key].append(_torso_luminance(crop))
    color_by_key = _team_split(lums)

    # GAME-LEVEL IDENTITY (contract §1 "shots inherit", wired 2026-08-25
    # after the probe measured inherited@0.6 = 1.0 precision-of-named):
    # every read in the game pools per relink person; a spine body whose
    # sheet + OCR channels fail inherits its person's roster-constrained
    # posterior at the INHERIT_SHARE_MIN bar. Also arbitrates the team
    # gate: a person's team is pooled from its whole lifetime, the
    # event-local luminance split from ~2 crops (ev13: luminance said
    # dark, game identity said white, the gate killed the right claim).
    pidx: dict = {}
    person_team: dict[int, int] = {}
    person_posteriors: dict[int, tuple[str, float]] = {}
    color_of_cluster: dict[int, str] = {}
    if stage_complete(job_dir / "person_registry"):
        # the persistent registry stage is the identity backbone
        # (people-only ruling 2026-08-25); --relink-map stays as the
        # override for ad-hoc runs
        from montehall_cv.pipeline.person_registry import (
            registry_person_index,
            registry_person_teams,
        )

        pidx = registry_person_index(job_dir)
        person_team = registry_person_teams(job_dir)
    if relink_map is not None and relink_map.exists():
        relink = json.loads(relink_map.read_text())
        pidx = person_index(relink["segments"])
        person_team = {int(k): v["team"]
                       for k, v in relink["person_meta"].items()
                       if v.get("team") is not None}
    if pidx:
        bt_person = bt_to_quark_person(job_dir, stage, pidx)
        sheet_rows = []
        color_pairs = []
        for (eid, tid), rd in reads.items():
            a = anchor_of.get(eid)
            if a is None:
                continue
            pid = person_at(pidx, tid, a)
            if pid is None:
                continue
            if rd.get("number"):
                sheet_rows.append((pid, rd["number"],
                                   rd.get("confidence") or 0.0))
            color = color_by_key.get((eid, tid))
            cluster = person_team.get(pid)
            if color is not None and cluster is not None:
                color_pairs.append((cluster, color))
        contact_rows = []
        if stage_complete(job_dir / "contact_reads"):
            for r in read_stage(job_dir / "contact_reads").to_pylist():
                if (r["number"] is None
                        or (r["confidence"] or 0.0) < MIN_PICK_READ_CONF):
                    continue
                pid = bt_person.get(int(r["track_id"]))
                if pid is not None:
                    contact_rows.append((pid, r["number"], r["confidence"]))
        ocr_rows = []
        for bt_id, posterior in ocr_posteriors.items():
            pid = bt_person.get(bt_id)
            if pid is not None:
                for num, prob in posterior.items():
                    ocr_rows.append((pid, num, prob))
        person_posteriors = person_pool(sheet_rows, contact_rows,
                                        ocr_rows, roster)
        color_of_cluster = cluster_colors(color_pairs)

    writer = ArtifactWriter(job_dir / "shot_attribution",
                            SHOT_ATTRIBUTION_SCHEMA)
    n_picked = 0
    n_spine = 0
    n_spine_verified = 0
    n_spine_refuted = 0
    n_spine_ocr_named = 0
    n_spine_inherited = 0
    n_spine_team_repointed = 0
    n_spine_team_dropped = 0
    n_spine_team_kept_by_identity = 0
    # pre-pass: per-event features + local team votes, pooled per
    # court_end into the offense-direction map (one mapping per clip —
    # footage spanning a side switch needs period segmentation first)
    feats_cache: dict[int, dict] = {}
    color_cache: dict[int, dict] = {}
    end_votes: list[tuple[str, str]] = []
    for e in events:
        eid = e["event_id"]
        cands = {tid: rows for (ev, tid), rows in cand_rows.items()
                 if ev == eid}
        if not cands:
            continue
        feats_cache[eid] = candidate_features(
            cands, balls, anchor_of[eid], fits.get(anchor_of[eid]), det)
        color_cache[eid] = {tid: color_by_key.get((eid, tid))
                            for tid in cands}
        local = _shooting_team(feats_cache[eid], color_cache[eid])
        if local is not None and e.get("court_end"):
            end_votes.append((e["court_end"], local))
    end_team = end_team_map(end_votes)
    n_team_from_end = 0
    n_team_from_state = 0

    # GAME STATE first (the third state store, architecture decision
    # 2026-08-25): possession-at-release / period direction
    # answer the shooting team from identified structure; the pooled
    # end map and the ~2-crop local vote are fallbacks only.
    from montehall_cv.pipeline.game_state import (
        load_state,
        shooting_team_cluster,
    )

    game_state = load_state(job_dir) if color_of_cluster else None

    for e in events:
        eid = e["event_id"]
        a_ms = anchor_of[eid]
        if eid not in feats_cache:
            continue
        feats = feats_cache[eid]
        color_of = color_cache[eid]
        team_local = _shooting_team(feats, color_of)
        team = end_team.get(e.get("court_end") or "") or team_local
        if team is not None and team_local is not None \
                and team != team_local:
            n_team_from_end += 1
        if game_state is not None:
            cluster = shooting_team_cluster(
                game_state, None, a_ms, e.get("court_end"))
            state_color = (color_of_cluster.get(cluster)
                           if cluster is not None else None)
            if state_color is not None:
                if team is not None and state_color != team:
                    n_team_from_state += 1
                team = state_color
        for tid in feats:
            feats[tid]["team_match"] = (
                None if team is None or color_of.get(tid) is None
                else color_of[tid] == team
            )
            feats[tid]["read_confidence"] = reads.get(
                (eid, tid), {}).get("confidence")
        order = rank_candidates(feats, model)
        spine_ts, spine_tid = _match_spine_shot(
            spine, e["ts_start_ms"], e["ts_end_ms"], feats)
        # team gate (spine v3, ev-9 defender-binding class): a spine body
        # on the NON-shooting team is the launched ball read off a closeout
        # defender's bbox — re-point to the latest shooting-team contact in
        # the same flight, else drop the spine claim back to ranking
        # (mismatch-last ordering + the .6 read gate, the pre-spine arm).
        if (spine_ts is not None and spine_tid is not None
                and team is not None
                and color_of.get(spine_tid) not in (None, team)):
            # identity arbitration (2026-08-25): the person's team is
            # pooled over its lifetime; the local luminance color over
            # ~2 crops. When game identity says the spine body IS on
            # the shooting team, the local mismatch is the unreliable
            # signal — keep the claim.
            pid = person_at(pidx, spine_tid, a_ms) if pidx else None
            cluster = person_team.get(pid) if pid is not None else None
            id_color = (color_of_cluster.get(cluster)
                        if cluster is not None else None)
            if id_color == team:
                n_spine_team_kept_by_identity += 1
            else:
                alt = _team_alternate(contacts, spine_ts, feats,
                                      color_of, team)
                if alt is not None:
                    spine_tid = alt
                    n_spine_team_repointed += 1
                else:
                    spine_tid = None
                    n_spine_team_dropped += 1
        spine_named_by_ocr = False
        spine_named_by_inherit = False
        if spine_tid is not None:
            # single-body verification (brick 2): the spine names ONE
            # body, so its read gets the scoreboard reader's persist
            # idiom — two disjoint sheets must AGREE. A refuted read
            # ABSTAINS the whole event (0-fabrication): the spine knows
            # WHO shot; if his number can't be read cleanly, an unnamed
            # entity row beats a flood guess from the candidate pool
            # (2026-08-25 grammar-v2 e2e: refuted fall-through handed
            # events 2 and 4 to the '23' flood).
            verdict = _verify_read(client, cache,
                                   by_key.get((eid, spine_tid), []), roster)
            if verdict == "refuted":
                reads[(eid, spine_tid)] = {"number": None, "confidence": 0.0}
                n_spine_refuted += 1
            elif isinstance(verdict, dict):
                reads[(eid, spine_tid)] = verdict
                n_spine_verified += 1
            # 2nd identity channel (spine v3, ev-2/4 class): sheet channel
            # failed on the spine body -> the PARSeq jersey posterior may
            # still name it. Same body, independent sensor — NOT a fall-
            # through to the candidate pool.
            spine_read = reads.get((eid, spine_tid), {})
            if (not spine_read.get("number")
                    or (spine_read.get("confidence") or 0.0)
                    < MIN_PICK_READ_CONF):
                ocr = _ocr_number_for(cand_rows[(eid, spine_tid)], loc_idx,
                                      ocr_posteriors, roster)
                if ocr is not None:
                    reads[(eid, spine_tid)] = {"number": ocr[0],
                                               "confidence": ocr[1]}
                    spine_named_by_ocr = True
                    n_spine_ocr_named += 1
            # 3rd identity channel (game-level inheritance, 2026-08-25):
            # sheet AND OCR failed on the spine body -> its relink
            # person's game-pooled posterior names it, at the measured
            # perfect-precision bar. The ev14 class: both sheets empty
            # on the body in the scramble, but the person carries a
            # converged '23' from the whole clip.
            spine_read = reads.get((eid, spine_tid), {})
            if (person_posteriors
                    and (not spine_read.get("number")
                         or (spine_read.get("confidence") or 0.0)
                         < MIN_PICK_READ_CONF)):
                pid = person_at(pidx, spine_tid, a_ms)
                inh = person_posteriors.get(pid) if pid is not None else None
                if inh is not None and inh[1] >= INHERIT_SHARE_MIN:
                    reads[(eid, spine_tid)] = {"number": inh[0],
                                               "confidence": inh[1]}
                    spine_named_by_inherit = True
                    n_spine_inherited += 1
            order = [spine_tid] + [t for t in order if t != spine_tid]
        # PERSON-ONLY CLAIMS (2026-08-25): the machine claims the BODY/PERSON whenever the
        # spine (post team-gate) names one; a read failing says nothing
        # about identity, so read-abstention no longer withholds the
        # person claim — it only leaves the number HINT empty. Windows
        # with no spine claim abstain from person claims entirely (the
        # rank order is evidence, never a claim — closing the flood's
        # last door: a confident wrong read can no longer take a pick).
        pick_tid = spine_tid
        hint = reads.get((eid, spine_tid), {}) if spine_tid is not None \
            else {}
        has_hint = bool(hint.get("number")) and (
            hint.get("confidence") or 0.0) >= MIN_PICK_READ_CONF
        method = ("ball_spine_inherited" if spine_named_by_inherit
                  else "ball_spine_ocr" if spine_named_by_ocr
                  else "ball_spine" if spine_tid is not None and has_hint
                  else "ball_spine_person" if spine_tid is not None
                  else "learned" if model is not None
                  else "launch" if fits.get(a_ms) else "release")
        n_spine += spine_tid is not None
        for rank, tid in enumerate(order):
            f = feats[tid]
            verdict = reads.get((eid, tid), {})
            picked = tid == pick_tid
            n_picked += picked
            writer.add({
                "job_id": job_id,
                "event_id": eid,
                "ts_anchor_ms": a_ms,
                "track_id": tid,
                "rank": rank,
                "picked": picked,
                "number": verdict.get("number"),
                "read_confidence": verdict.get("confidence"),
                "team": color_of.get(tid),
                "shooting_team": team,
                "release_prox_px": _finite(f["release_prox_px"]),
                "whole_prox_px": _finite(f["whole_prox_px"]),
                "launch_dist_ft": f["launch_dist_ft"],
                "team_match": f["team_match"],
                "person_id": person_at(pidx, tid, a_ms) if pidx else None,
                "n_obs": f["n_obs"],
                "max_h_px": f["max_h_px"],
                "method": method,
            })
    writer.close()

    return {
        "job_id": job_id,
        "binding_stage": stage.name,
        "attempts": len(events),
        "attempts_with_pick": n_picked,
        "attempts_with_spine_shooter": n_spine,
        "spine_reads_verified": n_spine_verified,
        "spine_reads_refuted": n_spine_refuted,
        "spine_named_by_ocr": n_spine_ocr_named,
        "spine_named_by_inheritance": n_spine_inherited,
        "spine_team_repointed": n_spine_team_repointed,
        "spine_team_dropped": n_spine_team_dropped,
        "spine_team_kept_by_identity": n_spine_team_kept_by_identity,
        "identity_persons_pooled": len(person_posteriors),
        "end_team_map": end_team,
        "team_overridden_by_end": n_team_from_end,
        "team_overridden_by_state": n_team_from_state,
        "game_state_loaded": game_state is not None,
        "sheets_read": len(reads),
        "flight_fits": sum(1 for f in fits.values() if f),
        "ranker": "learned" if model is not None else "heuristic",
        "roster_vocabulary": sorted(roster, key=str) if roster else None,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def _finite(v: float | None) -> float | None:
    return None if v is None or not np.isfinite(v) else float(v)


def _verify_read(client, cache, crops: list, roster: set[str]):
    """Two disjoint sheets from one body must AGREE on the number.

    The scoreboard reader's persist idiom applied to jersey reads: a
    flood/hallucination that survives only one view dies here. Returns a
    verified verdict dict on agreement, "refuted" on disagreement, and
    "insufficient" (keep the single-sheet read) under 4 crops."""
    if len(crops) < 4:
        return "insufficient"
    a = _read_sheet(client, build_sheet(crops[0::2]), roster, cache)
    b = _read_sheet(client, build_sheet(crops[1::2]), roster, cache)
    if a.get("number") and a.get("number") == b.get("number"):
        return {"number": a["number"],
                "confidence": max(a.get("confidence") or 0.0,
                                  b.get("confidence") or 0.0)}
    return "refuted"


def attribution_picks(job_dir: Path) -> dict[int, dict]:
    """event_id -> {person_id, team, number_hint, hint_conf} for
    consumers (run_boxscore). PERSON-ONLY (2026-08-25): the pick is a
    person claim; number is a pre-fill HINT, never asserted. Empty when
    the stage hasn't run — callers fall back to the legacy
    event_identity path."""
    if not stage_complete(job_dir / "shot_attribution"):
        return {}
    out: dict[int, dict] = {}
    for row in read_stage(job_dir / "shot_attribution").to_pylist():
        if row["picked"]:
            out[int(row["event_id"])] = {
                "person_id": row["person_id"],
                "team": row["team"],
                "number_hint": str(row["number"]) if row["number"] else None,
                "hint_conf": float(row["read_confidence"] or 0.0),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--attr-roster", type=Path, default=None,
                    help="JSON list of jersey numbers (BOTH teams) used as "
                         "read vocabulary; open-set reads without it")
    ap.add_argument("--wasb-root", type=Path, default=Path("/work/wasb"))
    ap.add_argument("--wasb-weights", type=Path, default=None,
                    help="explicit WASB checkpoint (no auto-discovery); "
                         "omit to rank by release proximity only")
    ap.add_argument("--binding-stage", type=Path, default=None)
    ap.add_argument("--relink-map", type=Path, default=None,
                    help="relink lock json (people_relink_smoke output); "
                         "enables game-identity inheritance + team-gate "
                         "arbitration (same contract as "
                         "people_track_video's flag)")
    ap.add_argument("--ranker-weights", type=Path, default=None,
                    help="learned ranker JSON (train_shot_ranker)")
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="0 = every body in the window (measured recall 1.0)")
    ap.add_argument("--flight-min-inliers", type=int, default=6,
                    help="fit gate: WASB peak inliers required (default = "
                         "the measured probe gate; lower = coverage A/B, "
                         "watch the null-fit rate)")
    ap.add_argument("--flight-span-min", type=float, default=0.4,
                    help="fit gate: minimum flight span seconds")
    args = ap.parse_args()
    roster = (set(json.loads(args.attr_roster.read_text()))
              if args.attr_roster else None)
    print(json.dumps(run(
        args.video, args.out, args.job_id,
        roster_numbers={str(n) for n in roster} if roster else None,
        wasb_root=args.wasb_root, wasb_weights=args.wasb_weights,
        binding_stage=args.binding_stage,
        ranker_weights=args.ranker_weights,
        relink_map=args.relink_map,
        max_candidates=args.max_candidates,
        flight_min_inliers=args.flight_min_inliers,
        flight_span_min=args.flight_span_min,
    ), indent=2))


if __name__ == "__main__":
    main()
