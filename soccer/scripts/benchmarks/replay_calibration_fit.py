"""Replay the primitive fit on dumped calibrator maps and explain each verdict.

For every .npz written by dump_calibration_maps.py: decode, fit exactly as the
benchmark does (fit_from_maps), then report the seed's conditioning, the refined
Jacobian's normalised singular values, and what the fit does under alternatives
(DLT seed, single-source primitives dropped). Read-only over the maps; writes one
JSON report beside them.

usage: python scripts/benchmarks/replay_calibration_fit.py <maps_dir> [--report out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from soccerviz.candidates import pitch_keypoints as pk
from soccerviz.core import primitive_calibration as pc
from soccerviz.core.pitch_template import primitive_names

NAMES = primitive_names()


def primitives_for(maps, width, height, protocol):
    points, confidence = pk.decode_heatmaps(maps[: pk.NUM_KEYPOINTS], width, height)
    valid = confidence >= protocol["landmark_confidence"]
    primitives, pixels = pk.decode_primitives(
        maps[pk.NUM_KEYPOINTS :], width, height, protocol["primitive_threshold"]
    )
    vertices = [
        pc.Primitive("point", int(k), points[k : k + 1], pk.pitch_landmarks()[k])
        for k in np.flatnonzero(valid)
    ]
    return vertices, primitives, pixels


def label(p):
    return f"v{p.index}" if p.kind == "point" else NAMES[p.index]


def singular_values(matrix, primitives, lines_world, threshold_m):
    """Refine from `matrix`, then the normalised singular values of the Jacobian at the
    solution (largest first, relative to the largest); the last one is what `_rank` tests."""
    refined, _ = pc._refine(matrix, primitives, lines_world, threshold_m)
    from scipy.optimize import least_squares

    unpack = pc._gauge(refined)
    result = least_squares(
        lambda x: pc._residuals(unpack(x), primitives, lines_world, weighted=True),
        np.zeros(8),
        loss="soft_l1",
        f_scale=threshold_m,
        x_scale="jac",
        max_nfev=1,
    )
    norms = np.linalg.norm(result.jac, axis=0)
    if not np.all(norms > 0):
        return refined, [0.0] * 8
    s = np.linalg.svd(result.jac / norms, compute_uv=False)
    return refined, [float(v / s[0]) for v in s]


def describe(fit):
    return {
        "accepted": fit.accepted,
        "reason": fit.reason,
        "inliers": fit.inlier_primitives,
        "of": fit.input_primitives,
        "residual_m": fit.median_residual_m,
        "loo_m": fit.median_loo_error_m,
    }


def replay(path):
    data = np.load(path)
    maps = data["maps"].astype(np.float32)
    width, height = int(data["width"]), int(data["height"])
    protocol = dict(pk.DEFAULT_PROTOCOL)
    vertices, primitives, _pixels = primitives_for(maps, width, height, protocol)
    lines_world = pc.line_coefficients()
    threshold_m = 1.5
    everything = vertices + primitives
    baseline = pc.calibrate_primitives(everything)
    seed = pc._seed(everything, lines_world, threshold_m)
    pruned = pc._prune(seed, everything, lines_world) if seed is not None else everything
    if seed is not None:
        _, sv = singular_values(seed, pruned, lines_world, threshold_m)
        library_s7 = sv[7]
    else:
        library_s7 = None
    row = {
        "frame": path.stem,
        "library_min_singular": library_s7,
        "vertices": [label(v) for v in vertices],
        "primitives": {label(p): len(p.image_points) for p in primitives},
        "baseline": describe(baseline),
    }
    if seed is not None:
        _, sv = singular_values(seed, everything, lines_world, threshold_m)
        row["ransac_seed_singular"] = [round(v, 9) for v in sv]
        row["ransac_seed_rank"] = int(sum(v > pc.RANK_TOLERANCE for v in sv))
        # per-primitive errors under the seed: which sources disagree before refinement
        errors = pc._primitive_errors(seed, everything, lines_world)
        row["seed_errors_m"] = {label(p): round(float(e), 2) for p, e in zip(everything, errors)}
    dlt = pc._dlt(everything, lines_world)
    if dlt is not None:
        _, sv = singular_values(dlt, everything, lines_world, threshold_m)
        row["dlt_seed_singular"] = [round(v, 9) for v in sv]
        row["dlt_seed_rank"] = int(sum(v > pc.RANK_TOLERANCE for v in sv))
        errors = pc._primitive_errors(dlt, everything, lines_world)
        row["dlt_errors_m"] = {label(p): round(float(e), 2) for p, e in zip(everything, errors)}
    # alternative: vertices + halfway + centre circle + side_top only (drop everything else)
    core = {"halfway", "centre_circle", "side_top", "side_bottom"}
    trimmed = vertices + [p for p in primitives if label(p) in core]
    row["without_offaxis_lines"] = describe(pc.calibrate_primitives(trimmed))
    # alternative: drop each line primitive in turn
    row["drop_one"] = {
        label(p): describe(pc.calibrate_primitives([q for q in everything if q is not p]))
        for p in primitives
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("maps_dir")
    parser.add_argument("--report")
    args = parser.parse_args()
    rows = [replay(p) for p in sorted(Path(args.maps_dir).glob("*.npz"))]
    report = Path(args.report or Path(args.maps_dir) / "replay-report.json")
    report.write_text(json.dumps(rows, indent=1))
    for r in rows:
        b = r["baseline"]
        print(
            f"{r['frame']} {'ACC' if b['accepted'] else 'REJ'} {b['reason']:<24}"
            f" inl {b['inliers']}/{b['of']} ransac_rank={r.get('ransac_seed_rank')}"
            f" dlt_rank={r.get('dlt_seed_rank')} raw_s7={r.get('ransac_seed_singular', [0] * 8)[7]:.2e}"
            f" library_s7={(r.get('library_min_singular') or 0):.2e}"
            f" no_offaxis={'ACC' if r['without_offaxis_lines']['accepted'] else r['without_offaxis_lines']['reason']}"
        )
    print("report", report)


if __name__ == "__main__":
    main()
