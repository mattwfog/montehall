"""Team clustering + offline tracklet association.

Reads localized detections + on-court evidence, samples crops per on-court
tracklet, embeds them (SigLIP2), then:

1. TEAM: k-means (k=3) over tracklet-mean SigLIP embeddings — two biggest
   clusters are the teams, the remainder bucket catches refs/mixed. Soft
   margins land as Evidence rows.
2. ASSOCIATION: merges tracklet fragments into entities via union-find over
   mutually-best pairs that pass a motion gate (court-space speed) AND an
   appearance gate. With --reid-weights the gate embedder is our from-scratch
   OSNet (metric-learned on player identity, design D9); without it, SigLIP
   fills in, where same-team players look alike, so motion does the gating.
   Conservative by design either way (under-merge > over-merge).

Outputs: <out>/<job_id>/entities/ (track_id -> entity_id) and
<out>/<job_id>/team_evidence/ (per-tracklet team cluster evidence).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa

from montehall_cv.pipeline.embed import CropEmbedder, ReidEmbedder, crop_bbox
from montehall_cv.pipeline.video import decode_frames
from montehall_cv.store.artifacts import ArtifactWriter, read_stage, stage_complete
from montehall_cv.store.schemas import EVIDENCE_SCHEMA

ENTITIES_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("track_id", pa.int32()),
        pa.field("entity_id", pa.int32()),
        pa.field("team_cluster", pa.int8()),  # 0/1 = teams, 2 = other/ref, -1 = unknown
    ]
)

MIN_LOCALIZED = 10
ON_COURT_MIN_RATIO = 0.5
CROPS_PER_TRACKLET = 6
MAX_GAP_FRAMES = 60
MAX_SPEED_FT_PER_FRAME = 1.2  # ~36 ft/s at 30fps, generous sprint bound
SPEED_SLACK_FT = 4.0
MIN_APPEARANCE_COS = 0.80  # SigLIP similarity floor (non-metric distribution)
MIN_REID_COS = 0.55  # OSNet floor; metric-learned cosines spread much wider
LONG_GAP_MAX_FRAMES = 1800  # ~60s at 30fps: bench stints, camera cuts
LONG_GAP_MIN_COS = 0.85  # stricter than short-gap: no motion gate to lean on


def run(video: Path, out_root: Path, job_id: str,
        reid_weights: Path | None = None, reid_arch: str = "osnet_x1_0",
        long_gap: bool = False) -> dict:
    job_dir = out_root / job_id
    if stage_complete(job_dir / "entities") and stage_complete(job_dir / "team_evidence"):
        return {"job_id": job_id, "skipped": True, "reason": "stages already complete"}
    started = time.monotonic()

    tracklets = _eligible_tracklets(job_dir)
    obs = _tracklet_observations(job_dir, set(tracklets))
    crops_plan = _crop_plan(obs)
    crops, owners = _collect_crops(video, crops_plan)

    embeddings = _mean_by_track(CropEmbedder().embed(crops), owners)
    team_of, margins = _cluster_teams(embeddings)

    if reid_weights is not None:
        gate_embeddings = _mean_by_track(
            ReidEmbedder(reid_weights, arch=reid_arch).embed(crops), owners
        )
        gate_floor = MIN_REID_COS
    else:
        gate_embeddings, gate_floor = embeddings, MIN_APPEARANCE_COS
    entity_of, n_merges, gate_cosines = _associate(
        obs, gate_embeddings, team_of, gate_floor
    )

    long_merges, long_cosines = 0, []
    if long_gap and reid_weights is not None:
        entity_of, long_merges, long_cosines = _associate_long_gap(
            obs, gate_embeddings, team_of, entity_of,
            jersey_of=_trusted_jerseys(job_dir),
        )

    _write_outputs(job_dir, job_id, tracklets, team_of, margins, entity_of)
    return {
        "job_id": job_id,
        "eligible_tracklets": len(tracklets),
        "embedded_tracklets": len(embeddings),
        "gate_embedder": "reid" if reid_weights is not None else "siglip",
        "team_sizes": _team_sizes(team_of),
        "merges": n_merges,
        "accepted_gate_cos": _quantiles(gate_cosines),
        "long_gap_merges": long_merges,
        "accepted_long_gap_cos": _quantiles(long_cosines),
        "entities": len(set(entity_of.values())),
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def _eligible_tracklets(job_dir: Path) -> list[int]:
    evidence = read_stage(job_dir / "evidence").to_pylist()
    keep = []
    for row in evidence:
        value = json.loads(row["value_json"])
        if (
            row["source_kind"] == "spatial"
            and value["n_localized"] >= MIN_LOCALIZED
            and value["on_court_ratio"] >= ON_COURT_MIN_RATIO
        ):
            keep.append(row["track_id"])
    return sorted(keep)


def _tracklet_observations(job_dir: Path, wanted: set[int]) -> dict[int, list[dict]]:
    table = read_stage(job_dir / "localized")
    obs: dict[int, list[dict]] = defaultdict(list)
    for row in table.to_pylist():
        tid = row["track_id"]
        if tid in wanted and row["court_x"] is not None:
            obs[tid].append(row)
    for rows in obs.values():
        rows.sort(key=lambda r: r["frame_idx"])
    return dict(obs)


def _crop_plan(obs: dict[int, list[dict]]) -> dict[int, list[tuple[int, tuple]]]:
    """frame_idx -> [(track_id, bbox)] with <=CROPS_PER_TRACKLET spread samples."""
    plan: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
    for tid, rows in obs.items():
        picks = np.linspace(0, len(rows) - 1, min(CROPS_PER_TRACKLET, len(rows))).astype(int)
        for i in sorted(set(picks.tolist())):
            row = rows[i]
            plan[row["frame_idx"]].append(
                (tid, (row["x1"], row["y1"], row["x2"], row["y2"]))
            )
    return dict(plan)


def _collect_crops(
    video: Path, plan: dict[int, list[tuple[int, tuple]]]
) -> tuple[list[np.ndarray], list[int]]:
    """One decode pass shared by every embedder: crops + owning track ids."""
    crops: list[np.ndarray] = []
    owners: list[int] = []
    remaining = set(plan)
    for frame in decode_frames(video):
        if frame.frame_idx in plan:
            for tid, bbox in plan[frame.frame_idx]:
                crop = crop_bbox(frame.image, *bbox)
                if crop is not None:
                    crops.append(crop)
                    owners.append(tid)
            remaining.discard(frame.frame_idx)
            if not remaining:
                break
    return crops, owners


def _mean_by_track(emb: np.ndarray, owners: list[int]) -> dict[int, np.ndarray]:
    if len(emb) == 0:
        return {}
    by_track: dict[int, list[np.ndarray]] = defaultdict(list)
    for tid, e in zip(owners, emb, strict=True):
        by_track[tid].append(e)
    out = {}
    for tid, vectors in by_track.items():
        mean = np.mean(vectors, axis=0)
        out[tid] = mean / max(np.linalg.norm(mean), 1e-9)
    return out


def _cluster_teams(embeddings: dict[int, np.ndarray]) -> tuple[dict[int, int], dict[int, float]]:
    """k=3 k-means; two largest clusters -> teams 0/1, rest -> 2."""
    if len(embeddings) < 6:
        return {tid: -1 for tid in embeddings}, {tid: 0.0 for tid in embeddings}
    from sklearn.cluster import KMeans

    tids = sorted(embeddings)
    matrix = np.stack([embeddings[t] for t in tids])
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(matrix)
    sizes = np.bincount(km.labels_, minlength=3)
    order = np.argsort(-sizes)
    relabel = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}
    dists = km.transform(matrix)
    team_of, margins = {}, {}
    for i, tid in enumerate(tids):
        team_of[tid] = relabel[int(km.labels_[i])]
        sorted_d = np.sort(dists[i])
        margins[tid] = float(sorted_d[1] - sorted_d[0])
    return team_of, margins


def _associate(
    obs: dict[int, list[dict]],
    embeddings: dict[int, np.ndarray],
    team_of: dict[int, int],
    min_cos: float,
) -> tuple[dict[int, int], int, list[float]]:
    """Union-find over mutually-best fragment pairs passing motion+appearance gates."""
    spans = {
        tid: (rows[0]["frame_idx"], rows[-1]["frame_idx"], rows[0], rows[-1])
        for tid, rows in obs.items()
        if tid in embeddings
    }
    candidates: dict[int, tuple[int, float]] = {}
    reverse_best: dict[int, tuple[int, float]] = {}
    for a, (_, a_end, _, a_last) in spans.items():
        best: tuple[int, float] | None = None
        for b, (b_start, _, b_first, _) in spans.items():
            if a == b or team_of.get(a) != team_of.get(b):
                continue
            gap = b_start - a_end
            if gap < 1 or gap > MAX_GAP_FRAMES:
                continue
            dist_ft = float(
                np.hypot(
                    b_first["court_x"] - a_last["court_x"],
                    b_first["court_y"] - a_last["court_y"],
                )
            )
            if dist_ft > MAX_SPEED_FT_PER_FRAME * gap + SPEED_SLACK_FT:
                continue
            cos = float(embeddings[a] @ embeddings[b])
            if cos < min_cos:
                continue
            if best is None or cos > best[1]:
                best = (b, cos)
        if best is not None:
            candidates[a] = best
            prev = reverse_best.get(best[0])
            if prev is None or best[1] > prev[1]:
                reverse_best[best[0]] = (a, best[1])

    parent = {tid: tid for tid in obs}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_merges = 0
    accepted_cos: list[float] = []
    for a, (b, cos) in candidates.items():
        if reverse_best.get(b, (None,))[0] != a:
            continue  # not mutually best
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            n_merges += 1
            accepted_cos.append(cos)

    roots = sorted({find(tid) for tid in obs})
    root_to_entity = {root: i for i, root in enumerate(roots)}
    return {tid: root_to_entity[find(tid)] for tid in obs}, n_merges, accepted_cos


def _trusted_jerseys(job_dir: Path) -> dict[int, set[str]]:
    """track_id -> trusted jersey candidates from run_jersey.
    run_all runs jersey before association precisely so this exists on
    fresh runs; still optional (absent -> no veto, empty mapping) for
    artifact sets produced by the old ordering."""
    identity_dir = job_dir / "identity"
    if not stage_complete(identity_dir):
        return {}
    jerseys: dict[int, set[str]] = defaultdict(set)
    for row in read_stage(identity_dir).to_pylist():
        jerseys[row["track_id"]].add(row["candidate"])
    return dict(jerseys)


def _associate_long_gap(
    obs: dict[int, list[dict]],
    embeddings: dict[int, np.ndarray],
    team_of: dict[int, int],
    entity_of: dict[int, int],
    jersey_of: dict[int, set[str]] | None = None,
) -> tuple[dict[int, int], int, list[float]]:
    """Second pass over MERGED entities: re-identify players across gaps the
    motion gate can't bridge (bench stints, leaving frame). No court-distance
    gate is possible at these gaps, so the ReID cosine floor is stricter,
    mutual-best still applies, and trusted jersey reads VETO any pair whose
    numbers conflict (identical-uniform lookalikes are exactly the failure
    a 0.84 rank-1 embedder cannot see). Under-merge > over-merge is the law."""
    jersey_of = jersey_of or {}
    members: dict[int, list[int]] = defaultdict(list)
    for tid, ent in entity_of.items():
        if tid in embeddings:
            members[ent].append(tid)

    frags = {}
    for ent, tids in members.items():
        rows = sorted(
            (r for tid in tids for r in obs[tid]), key=lambda r: r["frame_idx"]
        )
        teams = {team_of.get(tid) for tid in tids}
        emb = np.mean([embeddings[tid] for tid in tids], axis=0)
        frags[ent] = {
            "start": rows[0]["frame_idx"],
            "end": rows[-1]["frame_idx"],
            "team": teams.pop() if len(teams) == 1 else None,
            "emb": emb / max(np.linalg.norm(emb), 1e-9),
            "jerseys": set().union(*(jersey_of.get(t, set()) for t in tids))
            if tids else set(),
        }

    candidates: dict[int, tuple[int, float]] = {}
    reverse_best: dict[int, tuple[int, float]] = {}
    for a, fa in frags.items():
        best: tuple[int, float] | None = None
        for b, fb in frags.items():
            if a == b or fa["team"] is None or fa["team"] != fb["team"]:
                continue
            gap = fb["start"] - fa["end"]
            if gap <= MAX_GAP_FRAMES or gap > LONG_GAP_MAX_FRAMES:
                continue
            if fa["jerseys"] and fb["jerseys"] and not (fa["jerseys"] & fb["jerseys"]):
                continue  # trusted numbers disagree — hard veto
            cos = float(fa["emb"] @ fb["emb"])
            if cos < LONG_GAP_MIN_COS:
                continue
            if best is None or cos > best[1]:
                best = (b, cos)
        if best is not None:
            candidates[a] = best
            prev = reverse_best.get(best[0])
            if prev is None or best[1] > prev[1]:
                reverse_best[best[0]] = (a, best[1])

    parent = {ent: ent for ent in frags}
    root_jerseys = {ent: set(frags[ent]["jerseys"]) for ent in frags}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_merges = 0
    accepted: list[float] = []
    for a, (b, cos) in candidates.items():
        if reverse_best.get(b, (None,))[0] != a:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            ja, jb = root_jerseys[ra], root_jerseys[rb]
            if ja and jb and not (ja & jb):
                continue  # transitive conflict through a jersey-less chain
            parent[rb] = ra
            root_jerseys[ra] = ja | jb
            n_merges += 1
            accepted.append(cos)

    remap = {ent: find(ent) for ent in frags}
    roots = sorted(set(remap.values()))
    root_to_new = {root: i for i, root in enumerate(roots)}
    next_id = len(roots)
    unembedded: dict[int, int] = {}
    merged = {}
    for tid, ent in entity_of.items():
        if ent in remap:
            merged[tid] = root_to_new[remap[ent]]
        else:  # entity had no embedded members — renumber past the new range
            if ent not in unembedded:
                unembedded[ent] = next_id
                next_id += 1
            merged[tid] = unembedded[ent]
    return merged, n_merges, accepted


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.array(values)
    return {
        "p10": round(float(np.quantile(arr, 0.1)), 4),
        "p50": round(float(np.quantile(arr, 0.5)), 4),
        "p90": round(float(np.quantile(arr, 0.9)), 4),
    }


def _write_outputs(job_dir, job_id, tracklets, team_of, margins, entity_of) -> None:
    entities = ArtifactWriter(job_dir / "entities", ENTITIES_SCHEMA)
    for tid in tracklets:
        entities.add(
            {
                "job_id": job_id,
                "track_id": tid,
                "entity_id": entity_of.get(tid, -1) if entity_of.get(tid) is not None else -1,
                "team_cluster": team_of.get(tid, -1),
            }
        )
    entities.close()

    evidence = ArtifactWriter(job_dir / "team_evidence", EVIDENCE_SCHEMA)
    for tid in tracklets:
        if tid not in team_of:
            continue
        evidence.add(
            {
                "job_id": job_id,
                "track_id": tid,
                "frame_idx": None,
                "source_kind": "reid_match",
                "value_json": json.dumps(
                    {
                        "team_cluster": team_of[tid],
                        "margin": round(margins.get(tid, 0.0), 4),
                        "entity_id": entity_of.get(tid, -1),
                    }
                ),
                "weight": min(margins.get(tid, 0.0), 1.0),
            }
        )
    evidence.close()


def _team_sizes(team_of: dict[int, int]) -> dict[str, int]:
    sizes: dict[str, int] = defaultdict(int)
    for team in team_of.values():
        sizes[str(team)] += 1
    return dict(sizes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--reid-weights", type=Path, default=None,
                        help="from-scratch OSNet checkpoint; gates association "
                             "on metric-learned identity instead of SigLIP")
    parser.add_argument("--reid-arch", default="osnet_x1_0")
    parser.add_argument("--long-gap", action="store_true",
                        help="second ReID-only pass merging entities across "
                             "gaps the motion gate cannot bridge")
    args = parser.parse_args()
    print(json.dumps(run(args.video, args.out, args.job_id,
                         reid_weights=args.reid_weights, reid_arch=args.reid_arch,
                         long_gap=args.long_gap),
                     indent=2))


if __name__ == "__main__":
    main()
