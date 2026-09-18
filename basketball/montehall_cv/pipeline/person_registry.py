"""PERSON REGISTRY: persistent people-only identity across the game.

Design decisions, 2026-08-25:
  * Frigate-style persistent BODY IDs — every person on court is enrolled
    once and maintained for the whole game; events reference persons.
  * People-only identity, no number identity: the identity KEY is the PERSON
    (appearance gallery + team + motion + time). Jersey numbers are
    NEVER identity evidence — no number attaches, no number vetoes, no
    (team,number) exclusivity. A number is a post-hoc ATTRIBUTE pooled
    onto an already-built person, display/scoring only.

The machinery is the relink v4 lock engine (scripts/people_relink_smoke,
2026-08-04 rulings: team-first split layer, gallery best-of matching,
cannot-link intervals, Kalman coast) MOVED to its canonical pipeline home
and stripped of every number-based identity path. The script imports
back and keeps its render-facing JSON contract.

Stages written:
  person_registry/  one row per (track segment -> person) assignment
  person_meta/      one row per person: team, gallery size, lifetime,
                    and the pooled number ATTRIBUTE (nullable)

Consumers: shot_attribution (person_id per candidate row + game-identity
inheritance + team arbitration), people_relink_smoke (render json).
"""

from __future__ import annotations

import argparse
import json
import time
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa

from montehall_cv.store.artifacts import (
    ArtifactWriter,
    read_stage,
    stage_complete,
)

PERSON_SEGMENTS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("track_id", pa.int64()),
        pa.field("t0_ms", pa.int64()),
        pa.field("t1_ms", pa.int64()),
        pa.field("person_id", pa.int32()),
        pa.field("team", pa.int32(), nullable=True),
    ]
)

PERSON_META_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("person_id", pa.int32()),
        pa.field("team", pa.int32(), nullable=True),
        pa.field("segments", pa.int32()),
        pa.field("gallery", pa.int32()),
        pa.field("lifetime_ms", pa.int64()),
        pa.field("number_attr", pa.string(), nullable=True),
        pa.field("number_votes", pa.int32()),
    ]
)

DECODE_STRIDE = 8      # ~7.5fps crop sampling at 60fps source
MAX_CROPS_PER_SEGMENT = 8
MIN_QUALITY_CROPS = 3  # fingerprint birth quorum
MIN_BOX_H_FRAC = 0.08  # of frame height; kills far/tiny boxes
MAX_OCCLUSION = 0.3    # intersection over own area vs concurrent boxes
OVERLAP_MS = 150       # simultaneous-alive threshold => cannot-link
MATCH_MS = 17
READ_CONF_MIN = 0.6    # contact_sheet's accept bar (ATTRIBUTE pooling only)
READ_MIN_VOTES = 10    # bridged frames before the number ATTRIBUTE binds
READ_DOMINANCE = 0.8
LOCK_GALLERY_K = 16    # sharpest crops a person keeps across its lifetime
COAST_MAX_MS = 2000    # Kalman coast horizon
SIM_MIN_MOTION = 0.35  # relaxed appearance floor when the motion gate holds
KF_TAIL_ROWS = 5
KF_GATE_BASE = 0.6
KF_GATE_GROW = 1.2
TEAM_MIN_VOTES = 10
TEAM_DOMINANCE = 0.7
CENTROID_MIN_CROPS = 30
TEAM_MARGIN = 0.015
MIN_RUN = 3            # sustained-flip threshold (clean crops)
BRIDGE_IOU_MIN = 0.5


def load_binding(game_dir: Path, lo_ms: float, hi_ms: float,
                 stage_path: Path | None = None):
    binding = read_stage(stage_path or game_dir / "track_binding")
    bts = binding.column("ts_ms").to_numpy()
    btrack = binding.column("track_id").to_numpy()
    boxes = np.stack([binding.column(c).to_numpy()
                      for c in ("x1", "y1", "x2", "y2")], axis=1)
    win = (bts >= lo_ms) & (bts < hi_ms)
    bts, btrack, boxes = bts[win], btrack[win], boxes[win]
    order = np.argsort(bts, kind="stable")
    return bts[order], btrack[order], boxes[order]


def sharpness(crop: np.ndarray) -> float:
    g = crop.mean(axis=2).astype(np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
           - 4.0 * g[1:-1, 1:-1])
    return float(lap.var())


def occluded(m: int, i: int, j: int, boxes: np.ndarray,
             bts: np.ndarray, btrack: np.ndarray) -> bool:
    # Concurrent = same binding instant, different track (the ±MATCH_MS
    # self-occlusion bug, 08-04).
    x1, y1, x2, y2 = boxes[m]
    own = max(1.0, (x2 - x1) * (y2 - y1))
    for n in range(i, j):
        if btrack[n] == btrack[m] or abs(bts[n] - bts[m]) > 1:
            continue
        ix = min(x2, boxes[n, 2]) - max(x1, boxes[n, 0])
        iy = min(y2, boxes[n, 3]) - max(y1, boxes[n, 1])
        if ix > 0 and iy > 0 and ix * iy / own > MAX_OCCLUSION:
            return True
    return False


def sample_crops(video: Path, start_s: float, duration_s: float,
                 bts, btrack, boxes) -> dict[int, list[dict]]:
    """tid -> [{ts, sharp, occ, crop}] time-ordered. Size gate only:
    occluded crops still team-classify; excluded from centroids and
    fingerprints downstream."""
    import av

    from montehall_cv.pipeline.embed import crop_bbox

    container = av.open(str(video))
    stream = container.streams.video[0]
    tb = stream.time_base
    if start_s > 0:
        container.seek(int(start_s / tb), stream=stream)
    out: dict[int, list[dict]] = defaultdict(list)
    seen = 0
    min_h = None
    for frame in container.decode(stream):
        stamp = frame.pts if frame.pts is not None else frame.dts
        if stamp is None:
            continue
        ts = float(stamp * tb)
        if ts < start_s - 0.02:
            continue
        if duration_s and ts > start_s + duration_s:
            break
        if seen % DECODE_STRIDE:
            seen += 1
            continue
        seen += 1
        if min_h is None:
            min_h = MIN_BOX_H_FRAC * frame.height
        i = bisect_left(bts, ts * 1000 - MATCH_MS)
        j = bisect_left(bts, ts * 1000 + MATCH_MS)
        if i == j:
            continue
        rgb = frame.to_ndarray(format="rgb24")
        for m in range(i, j):
            if boxes[m, 3] - boxes[m, 1] < min_h:
                continue
            crop = crop_bbox(rgb, *boxes[m])
            if crop is None:
                continue
            out[int(btrack[m])].append({
                "ts": int(bts[m]),
                "sharp": sharpness(crop),
                "occ": occluded(m, i, j, boxes, bts, btrack),
                "crop": crop,
            })
    container.close()
    for rows in out.values():
        rows.sort(key=lambda r: r["ts"])
    return out


def bridge_stored_votes(attr_dir: Path, stored_attr: dict, bts, btrack,
                        boxes, lo_ms: float, hi_ms: float) -> list[tuple]:
    """[(replayed_track_id, ts_ms, value)] by ts+IoU best-match against
    the stored run's localized rows (stored ids never join the replay)."""
    loc = read_stage(attr_dir / "localized")
    lts = loc.column("ts_ms").to_numpy()
    ltrack = loc.column("track_id").to_numpy()
    lbox = np.stack([loc.column(c).to_numpy()
                     for c in ("x1", "y1", "x2", "y2")], axis=1)
    keep = ((lts >= lo_ms) & (lts < hi_ms)
            & np.isin(ltrack, list(stored_attr)))
    votes: list[tuple] = []
    for i in np.where(keep)[0]:
        j0 = bisect_left(bts, lts[i] - MATCH_MS)
        j1 = bisect_left(bts, lts[i] + MATCH_MS)
        best, best_j = 0.0, -1
        for j in range(j0, j1):
            ix = min(lbox[i, 2], boxes[j, 2]) - max(lbox[i, 0], boxes[j, 0])
            iy = min(lbox[i, 3], boxes[j, 3]) - max(lbox[i, 1], boxes[j, 1])
            if ix <= 0 or iy <= 0:
                continue
            inter = ix * iy
            a = (lbox[i, 2] - lbox[i, 0]) * (lbox[i, 3] - lbox[i, 1])
            b = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
            iou = inter / (a + b - inter)
            if iou > best:
                best, best_j = iou, j
        if best >= BRIDGE_IOU_MIN:
            votes.append((int(btrack[best_j]), int(lts[i]),
                          stored_attr[int(ltrack[i])]))
    return votes


def aggregate_votes(votes: list[tuple], min_votes: int,
                    dominance: float) -> dict[int, object]:
    by_key: dict[int, Counter] = defaultdict(Counter)
    for key, _ts, value in votes:
        by_key[key][value] += 1
    out: dict[int, object] = {}
    for key, counter in by_key.items():
        value, n = counter.most_common(1)[0]
        if n >= min_votes and n / sum(counter.values()) >= dominance:
            out[key] = value
    return out


def bridged_on_court(game_dir: Path, bts, btrack, boxes,
                     lo_ms: float, hi_ms: float,
                     min_ratio: float = 0.3) -> dict[int, str]:
    """Replayed track_id -> 'on'/'off' via the stored run's spatial
    evidence; kills coach/bench locks without touching refs."""
    stored: dict[int, str] = {}
    for r in read_stage(game_dir / "evidence").to_pylist():
        if r["source_kind"] != "spatial":
            continue
        value = json.loads(r["value_json"])
        if value.get("on_court_ratio") is None:
            continue
        stored[r["track_id"]] = \
            "on" if value["on_court_ratio"] >= min_ratio else "off"
    votes = bridge_stored_votes(game_dir, stored, bts, btrack,
                                boxes, lo_ms, hi_ms)
    return aggregate_votes(votes, TEAM_MIN_VOTES, 0.6)  # type: ignore[return-value]


def bridged_track_teams(game_dir: Path, bts, btrack, boxes,
                        lo_ms: float, hi_ms: float) -> dict[int, int]:
    """Replayed track_id -> stored team cluster, for centroid SEEDING."""
    team_of_stored = {
        r["track_id"]: int(r["team_cluster"])
        for r in read_stage(game_dir / "entities").to_pylist()
        if r["team_cluster"] >= 0
    }
    votes = bridge_stored_votes(game_dir, team_of_stored, bts, btrack,
                                boxes, lo_ms, hi_ms)
    return aggregate_votes(votes, TEAM_MIN_VOTES, TEAM_DOMINANCE)  # type: ignore[return-value]


def team_timelines(crops: dict[int, list[dict]],
                   seed_team: dict[int, int]) -> dict[int, list]:
    """Per-crop SigLIP classification against seeded team centroids."""
    from montehall_cv.pipeline.embed import CropEmbedder

    flat, owner = [], []
    for tid, rows in crops.items():
        for k, r in enumerate(rows):
            flat.append(r["crop"])
            owner.append((tid, k))
    if not flat:
        return {}
    emb = CropEmbedder().embed(flat)

    centroid_rows: dict[int, list[int]] = defaultdict(list)
    for idx, (tid, k) in enumerate(owner):
        team = seed_team.get(tid)
        if team is not None and not crops[tid][k]["occ"]:
            centroid_rows[team].append(idx)
    centroids = {}
    for team, idxs in centroid_rows.items():
        if len(idxs) >= CENTROID_MIN_CROPS:
            c = emb[idxs].mean(axis=0)
            centroids[team] = c / max(np.linalg.norm(c), 1e-8)
    if len(centroids) < 2:
        return {}
    keys = sorted(centroids)
    sims = emb @ np.stack([centroids[t] for t in keys]).T
    top = np.argmax(sims, axis=1)
    sims_sorted = np.sort(sims, axis=1)
    margin = sims_sorted[:, -1] - sims_sorted[:, -2]

    out: dict[int, list] = {tid: [None] * len(rows)
                            for tid, rows in crops.items()}
    for idx, (tid, k) in enumerate(owner):
        if not crops[tid][k]["occ"] and margin[idx] >= TEAM_MARGIN:
            out[tid][k] = int(keys[top[idx]])
    return out


def split_segments(crops: dict[int, list[dict]],
                   timelines: dict[int, list],
                   seed_team: dict[int, int],
                   span: dict[int, list]) -> list[dict]:
    """Track -> single-team segments (clean-run flips only)."""
    segments = []
    for tid, rows in crops.items():
        labels = timelines.get(tid, [])
        clean = [(k, lab) for k, lab in enumerate(labels)
                 if lab is not None]
        runs: list[list] = []
        for k, lab in clean:
            if runs and runs[-1][0] == lab:
                runs[-1][1].append(k)
            else:
                runs.append([lab, [k]])
        strong = [r for r in runs if len(r[1]) >= MIN_RUN]
        merged: list[list] = []
        for team, idxs in strong:
            if merged and merged[-1][0] == team:
                merged[-1][1].extend(idxs)
            else:
                merged.append([team, list(idxs)])
        if len(merged) < 2:
            if merged:
                team = merged[0][0]
            elif clean:
                team = Counter(lab for _, lab in clean).most_common(1)[0][0]
            else:
                team = seed_team.get(tid)
            segments.append({"tid": tid, "t0": span[tid][0],
                             "t1": span[tid][1], "team": team,
                             "rows": list(range(len(rows)))})
            continue
        bounds = [span[tid][0]]
        for si in range(1, len(merged)):
            prev_ts = rows[merged[si - 1][1][-1]]["ts"]
            next_ts = rows[merged[si][1][0]]["ts"]
            bounds.append((prev_ts + next_ts) // 2)
        bounds.append(span[tid][1])
        for si, (team, _idxs) in enumerate(merged):
            t0, t1 = bounds[si], bounds[si + 1]
            segments.append({
                "tid": tid, "t0": t0, "t1": t1, "team": int(team),
                "rows": [k for k in range(len(rows))
                         if t0 <= rows[k]["ts"] <= t1]})
    return segments


def sweep_locks(segments: list[dict], gallery_of: dict[int, list],
                bts, btrack, boxes,
                on_court: dict[int, str],
                sim_min: float) -> tuple[list[dict], dict]:
    """The people-only lock sweep: appearance + motion + team + time.

    STRIPPED vs relink v4 (people-only identity, 2026-08-25): no
    jersey-number veto, no number-agreement score bonus,
    no number-only hard attach, no (team,number) exclusivity. A number
    can never create, join, split, or block an identity."""
    tid_idx: dict[int, list[int]] = defaultdict(list)
    for kk in range(len(bts)):
        tid_idx[int(btrack[kk])].append(kk)
    tid_arr = {tid: (bts[idxs], boxes[idxs])
               for tid, idxs in tid_idx.items()}

    def seg_motion(seg: dict) -> dict:
        ts_a, box_a = tid_arr[seg["tid"]]
        i0 = int(np.searchsorted(ts_a, seg["t0"]))
        i1 = int(np.searchsorted(ts_a, seg["t1"], side="right"))
        first, last = box_a[i0], box_a[i1 - 1]
        tail0 = max(i0, i1 - KF_TAIL_ROWS)
        dt = float(ts_a[i1 - 1] - ts_a[tail0])
        c_last = ((last[0] + last[2]) / 2, (last[1] + last[3]) / 2)
        c_tail = ((box_a[tail0][0] + box_a[tail0][2]) / 2,
                  (box_a[tail0][1] + box_a[tail0][3]) / 2)
        v = ((c_last[0] - c_tail[0]) / dt, (c_last[1] - c_tail[1]) / dt) \
            if dt > 0 else (0.0, 0.0)
        return {"first": first, "t_last": float(ts_a[i1 - 1]),
                "cx": c_last[0], "cy": c_last[1],
                "h": float(last[3] - last[1]), "vx": v[0], "vy": v[1]}

    def interval_clear(lock: dict, t0: float, t1: float) -> bool:
        return all(min(t1, b) - max(t0, a) <= OVERLAP_MS
                   for a, b in lock["intervals"])

    locks: list[dict] = []
    stats = Counter()
    order = sorted(range(len(segments)), key=lambda i: segments[i]["t0"])
    for i in order:
        seg = segments[i]
        if on_court.get(seg["tid"]) == "off":
            stats["off_court_dropped"] += 1
            continue
        gallery = gallery_of.get(i)
        mo = seg_motion(seg)
        cands = []
        for li, lock in enumerate(locks):
            if lock["team"] != seg["team"]:
                continue
            if not interval_clear(lock, seg["t0"], seg["t1"]):
                continue
            kf = lock["kf"]
            gap = seg["t0"] - kf["t_last"]
            motion_ok = False
            if 0 < gap <= COAST_MAX_MS:
                pred = (kf["cx"] + kf["vx"] * gap,
                        kf["cy"] + kf["vy"] * gap)
                c0 = ((mo["first"][0] + mo["first"][2]) / 2,
                      (mo["first"][1] + mo["first"][3]) / 2)
                gate = (KF_GATE_BASE + KF_GATE_GROW * gap / 1000) * kf["h"]
                motion_ok = float(np.hypot(pred[0] - c0[0],
                                           pred[1] - c0[1])) <= gate
            if gallery is not None:
                sim = max(float(ge @ se) for _, ge in lock["gallery"]
                          for _, se in gallery)
                floor = SIM_MIN_MOTION if motion_ok else sim_min
                score = sim + (0.15 if motion_ok else 0.0)
                if sim >= floor:
                    cands.append((score, sim, li,
                                  "motion+app" if motion_ok
                                  else "appearance"))
            elif motion_ok:
                cands.append((0.9, 0.0, li, "motion"))
        if cands:
            motion_only = [c for c in cands if c[3] == "motion"]
            if gallery is None and len(motion_only) != 1:
                stats["ambiguous_unclaimed"] += 1
                continue
            score, sim, li, how = max(cands)
            lock = locks[li]
            gap = seg["t0"] - lock["kf"]["t_last"]
            stats["attach_" + how] += 1
            if how != "motion" and gap > 2000:
                stats["reacquires"] += 1
            lock["segs"].append(i)
            lock["intervals"].append((seg["t0"], seg["t1"]))
            lock["intervals"].sort()
            if gallery is not None:
                lock["gallery"].extend(gallery)
                lock["gallery"].sort(key=lambda e: -e[0])
                del lock["gallery"][LOCK_GALLERY_K:]
        else:
            if gallery is None:
                stats["unclaimed_no_gallery"] += 1
                continue
            locks.append({"team": seg["team"], "segs": [i],
                          "gallery": list(gallery),
                          "intervals": [(seg["t0"], seg["t1"])],
                          "kf": mo})
            lock = locks[-1]
        if mo["t_last"] >= lock["kf"]["t_last"]:
            lock["kf"] = mo
    return locks, dict(stats)


def build(video: Path, out_root: Path, job_id: str,
          reid_weights: Path,
          binding_stage: Path | None = None,
          reads_dir: Path | None = None,
          start_s: float = 0.0, duration_s: float = 0.0,
          sim_min: float = 0.5) -> dict:
    """Build the registry stages. Returns the summary (also usable by the
    relink wrapper to emit its render json)."""
    from montehall_cv.pipeline.embed import ReidEmbedder

    job_dir = out_root / job_id
    if stage_complete(job_dir / "person_registry") and stage_complete(
            job_dir / "person_meta"):
        return {"job_id": job_id, "skipped": True,
                "reason": "stage already complete"}
    started = time.monotonic()

    lo_ms = start_s * 1000
    hi_ms = (start_s + duration_s) * 1000 if duration_s else float("inf")
    bts, btrack, boxes = load_binding(job_dir, lo_ms, hi_ms,
                                      stage_path=binding_stage)
    span: dict[int, list] = {}
    for t, tid in zip(bts.tolist(), btrack.tolist()):
        s = span.setdefault(int(tid), [t, t])
        s[0], s[1] = min(s[0], t), max(s[1], t)

    crops = sample_crops(video, start_s, duration_s, bts, btrack, boxes)
    seed_team = bridged_track_teams(job_dir, bts, btrack, boxes,
                                    lo_ms, hi_ms)
    timelines = team_timelines(crops, seed_team)
    segments = split_segments(crops, timelines, seed_team, span)

    seg_quality: list[list[tuple]] = []
    for seg in segments:
        rows = crops[seg["tid"]]
        good = [(rows[k]["sharp"], rows[k]["crop"]) for k in seg["rows"]
                if not rows[k]["occ"]]
        good.sort(key=lambda e: -e[0])
        seg_quality.append(good[:MAX_CROPS_PER_SEGMENT])
    embeddable = [i for i, q in enumerate(seg_quality)
                  if len(q) >= MIN_QUALITY_CROPS]
    flat = [c for i in embeddable for _, c in seg_quality[i]]
    emb = ReidEmbedder(reid_weights).embed(flat)
    gallery_of: dict[int, list[tuple]] = {}
    k = 0
    for i in embeddable:
        n = len(seg_quality[i])
        gallery_of[i] = [(seg_quality[i][m][0], emb[k + m])
                         for m in range(n)]
        k += n

    on_court = bridged_on_court(job_dir, bts, btrack, boxes, lo_ms, hi_ms)
    locks, stats = sweep_locks(segments, gallery_of, bts, btrack, boxes,
                               on_court, sim_min)

    # number ATTRIBUTE (post-hoc, display/scoring only — never identity)
    votes_by_person: dict[int, Counter] = defaultdict(Counter)
    if reads_dir is not None:
        number_of_stored = {
            r["track_id"]: r["number"]
            for r in read_stage(reads_dir / "contact_reads").to_pylist()
            if r["number"] is not None and r["confidence"] >= READ_CONF_MIN
        }
        votes = bridge_stored_votes(reads_dir, number_of_stored,
                                    bts, btrack, boxes, lo_ms, hi_ms)
        seg_of_track: dict[int, list[int]] = defaultdict(list)
        for i, seg in enumerate(segments):
            seg_of_track[seg["tid"]].append(i)
        seg_person: dict[int, int] = {}
        for p, lock in enumerate(locks, start=1):
            for i in lock["segs"]:
                seg_person[i] = p
        for tid, ts, num in votes:
            for i in seg_of_track.get(tid, []):
                if segments[i]["t0"] <= ts <= segments[i]["t1"]:
                    p = seg_person.get(i)
                    if p is not None:
                        votes_by_person[p][num] += 1
                    break

    def number_attr(p: int) -> tuple[str | None, int]:
        c = votes_by_person.get(p)
        if not c:
            return None, 0
        num, n = c.most_common(1)[0]
        if n >= READ_MIN_VOTES and n / sum(c.values()) >= READ_DOMINANCE:
            return str(num), n
        return None, n

    sw = ArtifactWriter(job_dir / "person_registry", PERSON_SEGMENTS_SCHEMA)
    mw = ArtifactWriter(job_dir / "person_meta", PERSON_META_SCHEMA)
    person_meta: dict[int, dict] = {}
    seg_rows: list[list] = []
    for p, lock in enumerate(locks, start=1):
        for i in lock["segs"]:
            seg = segments[i]
            sw.add({"job_id": job_id, "track_id": int(seg["tid"]),
                    "t0_ms": int(seg["t0"]), "t1_ms": int(seg["t1"]),
                    "person_id": p,
                    "team": None if seg["team"] is None
                    else int(seg["team"])})
            seg_rows.append([int(seg["tid"]), int(seg["t0"]),
                             int(seg["t1"]), p])
        num, nv = number_attr(p)
        meta = {"team": lock["team"], "number": num,
                "tracks": len(lock["segs"]),
                "gallery": len(lock["gallery"]),
                "lifetime_ms": int(lock["intervals"][-1][1]
                                   - lock["intervals"][0][0])}
        person_meta[p] = meta
        mw.add({"job_id": job_id, "person_id": p,
                "team": None if lock["team"] is None
                else int(lock["team"]),
                "segments": len(lock["segs"]),
                "gallery": len(lock["gallery"]),
                "lifetime_ms": meta["lifetime_ms"],
                "number_attr": num, "number_votes": nv})
    sw.close()
    mw.close()

    return {
        "job_id": job_id,
        "tracks_in_window": len(span),
        "segments_total": len(segments),
        "segments_embedded": len(gallery_of),
        "persons": len(locks),
        "team_sizes": dict(Counter(
            s["team"] for s in segments if s["team"] is not None)),
        "numbered_attr_persons": sum(
            1 for p in person_meta if person_meta[p]["number"] is not None),
        "sweep": stats,
        "segments_rows": seg_rows,
        "person_meta": person_meta,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def registry_person_index(job_dir: Path) -> dict:
    """pidx (as shot_attribution.person_index) from the stage, for
    consumers; {} when the stage is absent."""
    if not stage_complete(job_dir / "person_registry"):
        return {}
    rows = read_stage(job_dir / "person_registry").to_pylist()
    from montehall_cv.pipeline.shot_attribution import person_index

    return person_index([[r["track_id"], r["t0_ms"], r["t1_ms"],
                          r["person_id"]] for r in rows])


def registry_person_teams(job_dir: Path) -> dict[int, int]:
    if not stage_complete(job_dir / "person_meta"):
        return {}
    return {int(r["person_id"]): int(r["team"])
            for r in read_stage(job_dir / "person_meta").to_pylist()
            if r["team"] is not None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--reid-weights", type=Path, required=True)
    ap.add_argument("--binding-stage", type=Path, default=None)
    ap.add_argument("--reads-dir", type=Path, default=None,
                    help="keyed extraction carrying contact_reads; the "
                         "number ATTRIBUTE only, never identity")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=0.0)
    ap.add_argument("--sim-min", type=float, default=0.5)
    args = ap.parse_args()
    out = build(args.video, args.out, args.job_id, args.reid_weights,
                binding_stage=args.binding_stage, reads_dir=args.reads_dir,
                start_s=args.start_s, duration_s=args.duration_s,
                sim_min=args.sim_min)
    print("PERSON_REGISTRY_DONE " + json.dumps(
        {k: v for k, v in out.items()
         if k not in ("segments_rows", "person_meta")}))


if __name__ == "__main__":
    main()
