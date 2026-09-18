"""Census of a heatmap calibrator's failures against a reference calibration.

For every dumped map (dump_calibration_maps.py) the network's decoded vertices and
primitives are scored against a reference homography, PnLCalib's on frames where it
lands under REFERENCE_MAX_M of the ground truth. That splits each failure into what
the network delivered (misses, false vertices, mislabelled lines) and what the fitter
did with it (an oracle fit over only the reference-correct evidence says whether the
frame was solvable from what the network gave). Frames without a reference stay
`no_reference`; nothing is inferred there.

One row per frame appended to <maps_dir>/census-rows.jsonl as it is computed (a rerun
skips rows already there); the aggregate census is written to
<maps_dir>/census-report.json and printed.

usage: python scripts/benchmarks/census_calibration_failures.py <maps_dir> <manifest>
           <model_results> <reference_predictions> <reference_results>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from soccerviz.candidates import pitch_keypoints as pk
from soccerviz.core import pitch_template as pt
from soccerviz.core import primitive_calibration as pc
from soccerviz.core.geometry import pitch_landmarks, project

NAMES = pt.primitive_names()
REFERENCE_MAX_M = 1.0  # a reference frame: PnLCalib accepted with median error under this
BAD_MEDIAN_M = 2.0  # an accepted model frame with median error over this is `bad`
CORRECT_M = 1.5  # the fitter's inlier threshold; evidence inside it is `correct`
VISIBLE_SAMPLES = 8  # 0.25 m samples in frame for a primitive to count as visible (2 m)
GRID_STEP = 60
USABLE_VERTEX = ("hit", "hit_offscreen", "offset")  # correctly labelled evidence
USABLE_PRIMITIVE = ("correct", "correct_offscreen", "offset")
SHIFT = np.array([[1.0, 0.0, -pt.LENGTH / 2], [0.0, 1.0, -pt.WIDTH / 2], [0.0, 0.0, 1.0]])


def frame_key(row):
    return f"{row['sequence']}-{row['frame_id']:06d}"


def classify_model_frame(result):
    """rejected / bad / good from the model's own results row."""
    if not result["accepted"]:
        return "rejected"
    error = result.get("error") or {}
    return "bad" if (error.get("median_m") or 0.0) > BAD_MEDIAN_M else "good"


def reference_matrices(prediction, result):
    """(image→world top-left, world top-left→image oriented) or None without a reference."""
    if not (prediction["accepted"] and result["accepted"]):
        return None
    if ((result.get("error") or {}).get("median_m") or np.inf) > REFERENCE_MAX_M:
        return None
    image_to_centred = np.asarray(prediction["homography"], float)
    image_to_world = np.linalg.inv(SHIFT) @ image_to_centred
    world_to_image = np.linalg.inv(image_to_world)
    anchor = project([[prediction["width"] / 2, prediction["height"] - 1]], image_to_world)
    return image_to_world, pt.orient_world_to_image(world_to_image, anchor)


def visible_landmarks(world_to_image, width, height):
    landmarks = pitch_landmarks()
    homogeneous = np.column_stack([landmarks, np.ones(len(landmarks))]) @ world_to_image.T
    xy = project(landmarks, world_to_image)
    return (
        (homogeneous[:, 2] > 1e-8)
        & np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )


def visible_primitives(world_to_image, width, height):
    samples = pt.project_primitives(world_to_image, width, height)
    counts = Counter(int(index) for index in samples[:, 2])
    return {index: counts.get(index, 0) >= VISIBLE_SAMPLES for index in range(pt.NUM_PRIMITIVES)}


def decode(maps, width, height, protocol):
    points, confidence = pk.decode_heatmaps(maps[: pk.NUM_KEYPOINTS], width, height)
    primitives, pixels = pk.decode_primitives(
        maps[pk.NUM_KEYPOINTS :], width, height, protocol["primitive_threshold"]
    )
    valid = confidence >= protocol["landmark_confidence"]
    vertices = [
        pc.Primitive("point", int(k), points[k : k + 1], pitch_landmarks()[k])
        for k in np.flatnonzero(valid)
    ]
    return points, confidence, vertices, primitives, pixels


def vertex_census(points, confidence, visible, image_to_world, protocol):
    """Per landmark: hit / false / miss / phantom / abstain, with the reference error."""
    world = project(points, image_to_world)
    errors = np.linalg.norm(world - pitch_landmarks(), axis=1)
    rows = {}
    for k in range(pk.NUM_KEYPOINTS):
        fired = bool(confidence[k] >= protocol["landmark_confidence"])
        error = float(errors[k]) if np.isfinite(errors[k]) else None
        if fired:
            verdict = vertex_verdict(error, bool(visible[k]))
        else:
            verdict = "miss" if visible[k] else "abstain"
        rows[f"v{k}"] = {
            "verdict": verdict,
            "confidence": round(float(confidence[k]), 3),
            "error_m": error,
        }
    return rows


def vertex_verdict(error, visible):
    """A fired vertex: hit (< CORRECT_M), offset (under PRUNE_M, the fitter's robust loss
    absorbs it), false (further, prunable); off-screen firings are phantoms unless they
    still land near their landmark (the decoder can place a landmark just outside)."""
    if error is None or error >= pc.PRUNE_M:
        return "false" if visible else "phantom"
    if error < CORRECT_M:
        return "hit" if visible else "hit_offscreen"
    return "offset" if visible else "phantom"


def primitive_verdict(name, row, visible):
    """A detected ridge: correct (< CORRECT_M on its own line), offset (own line still
    the nearest, under PRUNE_M), mislabelled (clearly on another template line),
    noise (on nothing), phantom (its line is not in frame)."""
    if row["error_m"] < CORRECT_M:
        return "correct" if visible else "correct_offscreen"
    if not visible:
        return "phantom"
    if row["lies_on"] != name and row["lies_on_error_m"] < CORRECT_M:
        return "mislabelled"
    if row["lies_on"] == name and row["error_m"] < pc.PRUNE_M:
        return "offset"
    return "noise"


def nearest_primitive(primitive, image_to_world, lines_world):
    """Which template primitive the ridge actually lies on under the reference."""
    best = (None, np.inf)
    for index in range(pt.NUM_PRIMITIVES):
        kind = "line" if index < pt.NUM_LINES else "conic"
        trial = pc.Primitive(kind, index, primitive.image_points)
        error = pc._primitive_errors(image_to_world, [trial], lines_world)[0]
        if error < best[1]:
            best = (NAMES[index], float(error))
    return best


def primitive_census(primitives, visible, image_to_world, lines_world):
    """Per template primitive: correct / mislabelled / miss / phantom / absent."""
    detected = {p.index: p for p in primitives}
    rows = {}
    for index in range(pt.NUM_PRIMITIVES):
        name = NAMES[index]
        primitive = detected.get(index)
        if primitive is None:
            rows[name] = {"verdict": "miss" if visible[index] else "absent"}
            continue
        error = float(pc._primitive_errors(image_to_world, [primitive], lines_world)[0])
        row = {"points": len(primitive.image_points), "error_m": round(error, 2)}
        if error >= CORRECT_M:
            actual, actual_error = nearest_primitive(primitive, image_to_world, lines_world)
            row["lies_on"], row["lies_on_error_m"] = actual, round(actual_error, 2)
        row["verdict"] = primitive_verdict(name, row, bool(visible[index]))
        rows[name] = row
    return rows


def fit_disagreement(matrix, image_to_world, width, height):
    """Median metre disagreement between a fit and the reference over on-pitch image points."""
    if matrix is None:
        return None
    xs, ys = np.meshgrid(np.arange(0, width, GRID_STEP), np.arange(0, height, GRID_STEP))
    grid = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    truth = project(grid, image_to_world)
    on_pitch = (
        np.isfinite(truth).all(axis=1)
        & (truth[:, 0] > -5)
        & (truth[:, 0] < pt.LENGTH + 5)
        & (truth[:, 1] > -5)
        & (truth[:, 1] < pt.WIDTH + 5)
    )
    if not on_pitch.any():
        return None
    fitted = project(grid[on_pitch], matrix)
    gap = np.linalg.norm(fitted - truth[on_pitch], axis=1)
    return float(np.median(np.where(np.isfinite(gap), gap, 1e3)))


def describe(fit):
    return {
        "accepted": fit.accepted,
        "reason": fit.reason,
        "inliers": fit.inlier_primitives,
        "of": fit.input_primitives,
    }


def attribute(row):
    """Stage the failure one dimension at a time; the rule that fired is recorded."""
    if row["model_class"] == "good":
        return "good", "model_class:good"
    if row["reference"] is None:
        return "no_reference", "reference:pnlcalib_not_under_1m"
    vertices, primitives = row["vertices"], row["primitives"]
    correct_v = sum(v["verdict"] in USABLE_VERTEX for v in vertices.values())
    correct_p = sum(p["verdict"] in USABLE_PRIMITIVE for p in primitives.values())
    contaminants = [
        name for name, v in vertices.items() if v["verdict"] in ("false", "phantom")
    ] + [
        name
        for name, p in primitives.items()
        if p["verdict"] in ("mislabelled", "noise", "phantom")
    ]
    near = [
        name
        for name in contaminants
        if (vertices.get(name) or primitives.get(name)).get("error_m") is not None
        and (vertices.get(name) or primitives.get(name))["error_m"] < pc.PRUNE_M
    ]
    oracle = row["oracle_fit"]
    if correct_v + correct_p < pc.MIN_INLIER_PRIMITIVES:
        return (
            "network_insufficient",
            f"evidence:{correct_v}v+{correct_p}p<{pc.MIN_INLIER_PRIMITIVES}",
        )
    if oracle is not None and not oracle["accepted"]:
        return "geometry_degenerate", f"oracle_fit:{oracle['reason']}"
    if oracle is not None and (row["oracle_gap_m"] or np.inf) > BAD_MEDIAN_M:
        return "geometry_degenerate", f"oracle_gap:{row['oracle_gap_m']:.1f}m"
    if near:
        return "network_contaminated", "unpruned_contaminants:" + ",".join(near)
    if contaminants:
        return "fitter_fault", "prunable_contaminants_only:" + ",".join(contaminants)
    return "fitter_fault", "clean_input_oracle_solves"


def census_frame(path, manifest_row, model_result, reference_prediction, reference_result):
    data = np.load(path)
    maps = data["maps"].astype(np.float32)
    width, height = int(data["width"]), int(data["height"])
    protocol = dict(pk.DEFAULT_PROTOCOL)
    lines_world = pt.line_coefficients()
    points, confidence, vertices, primitives, pixels = decode(maps, width, height, protocol)
    everything = vertices + primitives
    baseline = pc.calibrate_primitives(everything)
    row = {
        "frame": path.stem,
        "sequence": manifest_row["sequence"],
        "frame_id": manifest_row["frame_id"],
        "model_class": classify_model_frame(model_result),
        "model_reason": model_result["reason"],
        "model_median_m": (model_result.get("error") or {}).get("median_m"),
        "reference_median_m": (reference_result.get("error") or {}).get("median_m"),
        "replayed_fit": describe(baseline),
        "fired_vertices": len(vertices),
        "detected_primitives": [NAMES[p.index] for p in primitives],
        "primitive_pixels": pixels,
    }
    reference = reference_matrices(reference_prediction, reference_result)
    row["reference"] = None if reference is None else "pnlcalib"
    if reference is None:
        row.update(vertices={}, primitives={}, oracle_fit=None, oracle_gap_m=None)
    else:
        image_to_world, world_to_image = reference
        landmarks_visible = visible_landmarks(world_to_image, width, height)
        primitives_visible = visible_primitives(world_to_image, width, height)
        row["vertices"] = vertex_census(
            points, confidence, landmarks_visible, image_to_world, protocol
        )
        row["primitives"] = primitive_census(
            primitives, primitives_visible, image_to_world, lines_world
        )
        row["visible_landmarks"] = int(landmarks_visible.sum())
        row["visible_primitives"] = [NAMES[i] for i, v in primitives_visible.items() if v]
        correct = [
            v for v in vertices if row["vertices"][f"v{v.index}"]["verdict"] in USABLE_VERTEX
        ] + [
            p
            for p in primitives
            if row["primitives"][NAMES[p.index]]["verdict"] in USABLE_PRIMITIVE
        ]
        oracle = (
            pc.calibrate_primitives(correct) if len(correct) >= pc.MIN_INLIER_PRIMITIVES else None
        )
        row["oracle_fit"] = None if oracle is None else describe(oracle)
        row["oracle_gap_m"] = (
            None
            if oracle is None
            else fit_disagreement(oracle.matrix, image_to_world, width, height)
        )
        row["replayed_gap_m"] = fit_disagreement(baseline.matrix, image_to_world, width, height)
    row["attribution"], row["attribution_rule"] = attribute(row)
    return row


def load_inputs(args):
    manifest = json.loads(Path(args.manifest).read_text())
    model = json.loads(Path(args.model_results).read_text())
    reference_predictions = json.loads(Path(args.reference_predictions).read_text())
    reference_results = json.loads(Path(args.reference_results).read_text())
    by_key = lambda rows: {frame_key(r): r for r in rows}
    return (
        by_key(manifest["frames"]),
        by_key(model["frames"]),
        by_key(reference_predictions["frames"]),
        by_key(reference_results["frames"]),
    )


def aggregate(rows):
    """The census: every dimension counted on its own, the fallback bucket printed."""
    report = {"frames": len(rows), "model_class": Counter(r["model_class"] for r in rows)}
    report["reference_available"] = Counter(
        f"{r['model_class']}:{'yes' if r['reference'] else 'no'}" for r in rows
    )
    report["attribution_by_class"] = {
        cls: Counter(r["attribution"] for r in rows if r["model_class"] == cls)
        for cls in ("rejected", "bad", "good")
    }
    report["attribution_rules"] = Counter(
        f"{r['model_class']}|{r['attribution']}|{r['attribution_rule'].split(':')[0]}" for r in rows
    )
    report["attribution_by_sequence"] = {
        seq: Counter(f"{r['model_class']}|{r['attribution']}" for r in rows if r["sequence"] == seq)
        for seq in sorted({r["sequence"] for r in rows})
    }
    vertex_totals, primitive_totals = defaultdict(Counter), defaultdict(Counter)
    for r in rows:
        for v in r["vertices"].values():
            vertex_totals[r["model_class"]][v["verdict"]] += 1
        for p in r["primitives"].values():
            primitive_totals[r["model_class"]][p["verdict"]] += 1
    report["vertex_verdicts_by_class"] = {k: dict(v) for k, v in vertex_totals.items()}
    report["primitive_verdicts_by_class"] = {k: dict(v) for k, v in primitive_totals.items()}
    report["mislabel_pairs"] = Counter(
        f"{name}->{p['lies_on']}"
        for r in rows
        for name, p in r["primitives"].items()
        if p["verdict"] == "mislabelled"
    )
    report["replay_matches_benchmark"] = Counter(
        "same_verdict"
        if r["replayed_fit"]["reason"] == r["model_reason"]
        else f"differs:{r['model_reason']}->{r['replayed_fit']['reason']}"
        for r in rows
    )
    report["missed_primitives"] = Counter(
        name
        for r in rows
        if r["model_class"] != "good"
        for name, p in r["primitives"].items()
        if p["verdict"] == "miss"
    )
    report["missed_vertices_bad_frames"] = Counter(
        name
        for r in rows
        if r["model_class"] != "good"
        for name, v in r["vertices"].items()
        if v["verdict"] == "miss"
    )
    report["oracle_solves"] = Counter(
        f"{r['model_class']}:{'solved' if r['oracle_fit'] and r['oracle_fit']['accepted'] and (r['oracle_gap_m'] or 99) < BAD_MEDIAN_M else 'not'}"
        for r in rows
        if r["reference"]
    )
    return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in report.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("maps_dir")
    parser.add_argument("manifest")
    parser.add_argument("model_results")
    parser.add_argument("reference_predictions")
    parser.add_argument("reference_results")
    args = parser.parse_args()
    maps_dir = Path(args.maps_dir)
    manifest, model, reference_predictions, reference_results = load_inputs(args)
    rows_path = maps_dir / "census-rows.jsonl"
    done = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            done[row["frame"]] = row
    with rows_path.open("a") as sink:
        for path in sorted(maps_dir.glob("*.npz")):
            key = path.stem
            if key in done:
                continue
            row = census_frame(
                path, manifest[key], model[key], reference_predictions[key], reference_results[key]
            )
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            done[key] = row
            print(
                f"{key} {row['model_class']:<8} {row['model_reason']:<24} -> {row['attribution']} ({row['attribution_rule']})"
            )
    rows = [done[k] for k in sorted(done)]
    for row in rows:  # attribution rules may have changed since a row was written
        row["attribution"], row["attribution_rule"] = attribute(row)
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = aggregate(rows)
    (maps_dir / "census-report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
