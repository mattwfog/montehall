"""Score fitter variants on dumped calibrator maps with the official evaluator.

The network output is fixed (the .npz maps from dump_calibration_maps.py); only the
fit changes. Each variant writes a calibration-predictions/v1 file and its
calibration-benchmark-results/v1 evaluation under <maps_dir>/variants/, so numbers
are directly comparable with the benchmark's own results files. The 2026-09-08
audit's homography variants (tangent gauge, 5 m prune, DLT fallback) were folded into
the library; `library` is that fitter.

2026-09-10, the model question the failure census left open: the same evidence is fit
by three models, the free homography (`library`, 8 parameters), a pinhole camera
(`pinhole`, 7: focal, rotation, position) and a pinhole camera held at a per-sequence
position (`pinhole-held`, 4: focal and rotation; the position is the median camera of
the frames the library fitter itself accepted in that sequence, so no ground truth is
used; `pinhole-held-good` takes the median over the frames the library fitter got
within 2 m of the ground truth and is a labelled diagnostic, not a product setting).
Two evidence tracks: `raw`, the decoded maps as the product sees them, and `oracle`,
only the evidence the census graded correct under the PnLCalib reference, which
isolates the model from the detections. Every fit is appended to
<maps_dir>/variants/frame-fits.jsonl as it completes and a rerun resumes; the
census-conditioned comparison lands in <maps_dir>/variants/census-by-variant.json.

usage: python scripts/benchmarks/evaluate_fit_variants.py <maps_dir> <manifest> <protocol>
           [--census <census-rows.jsonl>] [--reference-predictions <pnl-predictions.json>]
           [--workers N]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from soccerviz.candidates import pitch_keypoints as pk
from soccerviz.candidates.calibration_model import centered_homography, evaluate_calibrations
from soccerviz.core import pinhole_calibration as ph
from soccerviz.core import pitch_template as pt
from soccerviz.core import primitive_calibration as pc
from soccerviz.core.assets import sha256

NAMES = pt.primitive_names()
SHIFT = np.array([[1.0, 0.0, -pt.LENGTH / 2], [0.0, 1.0, -pt.WIDTH / 2], [0.0, 0.0, 1.0]])
MODES = ("raw", "oracle")
HALFWAY_ONLY = {"side_top", "halfway", "centre_circle"}


def _census_module():
    """The census script's verdict vocabulary, imported from the sibling file."""
    spec = importlib.util.spec_from_file_location(
        "census_calibration_failures", Path(__file__).with_name("census_calibration_failures.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError("census_calibration_failures.py not found beside this script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CENSUS = _census_module()


def fit_library(primitives, meta, position):
    return pc.calibrate_primitives(primitives)


def fit_pinhole(primitives, meta, position):
    return ph.calibrate_pinhole(primitives, meta["width"], meta["height"])


def fit_pinhole_held(primitives, meta, position):
    return ph.calibrate_pinhole(primitives, meta["width"], meta["height"], position=position)


VARIANTS = {
    "library": fit_library,
    "pinhole": fit_pinhole,
    "pinhole-held": fit_pinhole_held,
    "pinhole-held-good": fit_pinhole_held,
    "pinhole-held-self": fit_pinhole_held,
    "pinhole-held-game": fit_pinhole_held,
}
# Where each held variant's position comes from: the library fitter's accepted frames, its
# ground-truth-good frames (diagnostic), the free pinhole fitter's accepted frames in the
# same sequence, or in every sequence of the same game (one main-camera tripod per game).
PRIOR_OF = {
    "pinhole-held": ("library", "accepted"),
    "pinhole-held-good": ("library", "good"),
    "pinhole-held-self": ("pinhole", "sequence"),
    "pinhole-held-game": ("pinhole", "game"),
}
PASSES = (
    ("pinhole", "pinhole-held", "pinhole-held-good"),
    ("pinhole-held-self", "pinhole-held-game"),
)


def decode_evidence(path, protocol):
    data = np.load(path)
    maps = data["maps"].astype(np.float32)
    width, height = int(data["width"]), int(data["height"])
    points, confidence = pk.decode_heatmaps(maps[: pk.NUM_KEYPOINTS], width, height)
    valid = confidence >= protocol["landmark_confidence"]
    primitives, _ = pk.decode_primitives(
        maps[pk.NUM_KEYPOINTS :], width, height, protocol["primitive_threshold"]
    )
    vertices = [
        pc.Primitive("point", int(k), points[k : k + 1], pk.pitch_landmarks()[k])
        for k in np.flatnonzero(valid)
    ]
    meta = {
        "sequence": str(data["sequence"]),
        "frame_id": int(data["frame_id"]),
        "width": width,
        "height": height,
        "image_sha256": str(data["image_sha256"]),
    }
    return vertices + primitives, meta


def oracle_subset(evidence, census_row):
    """The evidence the census graded correct under the reference; None without one."""
    if not census_row or census_row.get("reference") is None:
        return None
    kept = []
    for primitive in evidence:
        if primitive.kind == "point":
            verdict = census_row["vertices"].get(f"v{primitive.index}", {}).get("verdict")
            usable = verdict in CENSUS.USABLE_VERTEX
        else:
            verdict = census_row["primitives"].get(NAMES[primitive.index], {}).get("verdict")
            usable = verdict in CENSUS.USABLE_PRIMITIVE
        if usable:
            kept.append(primitive)
    return kept


def camera_summary(fit, meta):
    camera = ph.camera_of(fit, meta["width"], meta["height"]) if fit.accepted else None
    return None if camera is None else camera.describe()


def fit_row(frame, variant, mode, evidence, meta, position):
    started = time.perf_counter()
    if evidence is None:
        fit, homography = None, None
    else:
        fit = VARIANTS[variant](evidence, meta, position)
        homography = centered_homography(fit.matrix).tolist() if fit.accepted else None
    return {
        "frame": frame,
        "variant": variant,
        "evidence": mode,
        **meta,
        "evidence_count": None if evidence is None else len(evidence),
        "held_position": None if position is None else [round(float(v), 2) for v in position],
        "accepted": bool(fit.accepted) if fit else False,
        "reason": fit.reason if fit else "no_reference",
        "homography": homography,
        "diagnostics": fit.diagnostics() if fit else {},
        "camera": camera_summary(fit, meta) if fit else None,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def fit_frame(path, protocol, census_row, jobs):
    """All (variant, mode, position) jobs of one frame; decoded once."""
    evidence, meta = decode_evidence(Path(path), protocol)
    oracle = oracle_subset(evidence, census_row)
    frame = Path(path).stem
    return [
        fit_row(frame, variant, mode, evidence if mode == "raw" else oracle, meta, position)
        for variant, mode, position in jobs
    ]


def run_jobs(paths, protocol, census, jobs_for, done, sink, workers):
    """Run `jobs_for(path)` per frame in a process pool; append rows as they complete."""
    pending = {
        path: [j for j in jobs_for(path) if (path.stem, j[0], j[1]) not in done] for path in paths
    }
    pending = {path: jobs for path, jobs in pending.items() if jobs}
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = {
            pool.submit(fit_frame, str(path), protocol, census.get(path.stem), jobs): path
            for path, jobs in pending.items()
        }
        for count, future in enumerate(as_completed(futures), 1):
            for row in future.result():
                sink.write(json.dumps(row) + "\n")
                done[(row["frame"], row["variant"], row["evidence"])] = row
            sink.flush()
            if count % 30 == 0 or count == len(futures):
                print(f"  {count}/{len(futures)} frames, last {futures[future].stem}")


def library_camera(row):
    """Camera position behind an accepted library fit, when its homography is a pinhole one."""
    if not row["accepted"]:
        return None
    image_to_world = np.linalg.inv(SHIFT) @ np.asarray(row["homography"], float)
    camera = ph.camera_from_homography(np.linalg.inv(image_to_world), row["width"], row["height"])
    return None if camera is None else camera.position


def sequence_priors(done, library_results, reference):
    """Per sequence: median library camera over accepted frames and over ground-truth-good
    frames, their spread and counts, and PnLCalib's median position for comparison."""
    good = {
        (r["sequence"], r["frame_id"])
        for r in library_results["frames"]
        if r["accepted"] and (r["error"].get("median_m") or np.inf) <= CENSUS.BAD_MEDIAN_M
    }
    positions = defaultdict(lambda: {"accepted": [], "good": []})
    for (frame, variant, mode), row in done.items():
        if variant != "library" or mode != "raw":
            continue
        position = library_camera(row)
        if position is None:
            continue
        positions[row["sequence"]]["accepted"].append(position)
        if (row["sequence"], row["frame_id"]) in good:
            positions[row["sequence"]]["good"].append(position)
    priors = {}
    for sequence in sorted({row["sequence"] for row in done.values()}):
        entry = {}
        for kind in ("accepted", "good"):
            stack = np.asarray(positions[sequence][kind]).reshape(-1, 3)
            if len(stack):
                median = np.median(stack, axis=0)
                entry[kind] = {
                    "position": [round(float(v), 2) for v in median],
                    "frames": len(stack),
                    "mad_m": [
                        round(float(v), 2) for v in np.median(np.abs(stack - median), axis=0)
                    ],
                }
            else:
                entry[kind] = {
                    "position": [round(float(v), 2) for v in ph.DEFAULT_POSITION],
                    "frames": 0,
                    "fallback": "DEFAULT_POSITION",
                }
        if reference.get(sequence):
            stack = np.asarray(reference[sequence])
            entry["pnlcalib_median_position"] = [
                round(float(v), 2) for v in np.median(stack, axis=0)
            ]
        priors[sequence] = entry
    return priors


def pinhole_priors(done, game_of):
    """Per sequence: median position of the free pinhole fits accepted in that sequence,
    and over every sequence of the same game."""
    by_sequence = defaultdict(list)
    for (_, variant, mode), row in done.items():
        if variant == "pinhole" and mode == "raw" and row["accepted"] and row["camera"]:
            x, y, height = row["camera"]["position_m"]
            by_sequence[row["sequence"]].append([x, y, -height])
    by_game = defaultdict(list)
    for sequence, positions in by_sequence.items():
        by_game[game_of.get(sequence, sequence)].extend(positions)

    def summary(positions):
        stack = np.asarray(positions).reshape(-1, 3)
        if not len(stack):
            return {
                "position": [round(float(v), 2) for v in ph.DEFAULT_POSITION],
                "frames": 0,
                "fallback": "DEFAULT_POSITION",
            }
        median = np.median(stack, axis=0)
        return {
            "position": [round(float(v), 2) for v in median],
            "frames": len(stack),
            "mad_m": [round(float(v), 2) for v in np.median(np.abs(stack - median), axis=0)],
        }

    return {
        sequence: {
            "sequence": summary(by_sequence[sequence]),
            "game": summary(by_game[game_of.get(sequence, sequence)]),
            "game_id": game_of.get(sequence),
        }
        for sequence in sorted({row["sequence"] for row in done.values()})
    }


def games_of(path):
    """sequence → game id from the corpus summary, if present."""
    if path is None or not Path(path).exists():
        return {}
    summary = json.loads(Path(path).read_text())
    return {entry["sequence"]: str(entry["game_id"]) for entry in summary.get("sequences", [])}


def reference_positions(path):
    """PnLCalib's recorded camera positions per sequence, moved to the top-left frame."""
    if path is None:
        return {}
    by_sequence = defaultdict(list)
    for frame in json.loads(Path(path).read_text())["frames"]:
        parameters = frame.get("camera_parameters") or {}
        if frame["accepted"] and "position_meters" in parameters:
            shifted = np.asarray(parameters["position_meters"], float) + [
                pt.LENGTH / 2,
                pt.WIDTH / 2,
                0,
            ]
            by_sequence[frame["sequence"]].append(shifted)
    return by_sequence


def predictions_for(rows, variant, mode, maps_dir, manifest_path, protocol_path):
    frames = [
        {
            "accepted": row["accepted"],
            "reason": row["reason"],
            "homography": row["homography"],
            "diagnostics": {
                **row["diagnostics"],
                "camera": row["camera"],
                "held_position": row["held_position"],
            },
            "sequence": row["sequence"],
            "frame_id": row["frame_id"],
            "width": row["width"],
            "height": row["height"],
            "image_sha256": row["image_sha256"],
        }
        for row in rows
    ]
    return {
        "schema": "calibration-predictions/v1",
        "manifest_sha256": sha256(manifest_path),
        "protocol_sha256": sha256(protocol_path),
        "backend": {"name": f"replayed-maps/{maps_dir.name}", "variant": variant, "evidence": mode},
        "source_labels_are_model_inputs": mode == "oracle",
        "frames": frames,
        "complete": True,
    }


def evaluate(done, variant, mode, out, maps_dir, manifest, protocol_path):
    rows = sorted(
        (row for (_, v, m), row in done.items() if v == variant and m == mode),
        key=lambda r: (r["sequence"], r["frame_id"]),
    )
    label = variant if mode == "raw" else f"{variant}-oracle"
    pred_path = out / f"{label}-predictions.json"
    pred_path.write_text(
        json.dumps(
            predictions_for(rows, variant, mode, maps_dir, manifest, protocol_path), indent=1
        )
    )
    result = evaluate_calibrations(manifest, protocol_path, pred_path)
    (out / f"{label}-results.json").write_text(json.dumps(result, indent=1))
    return result


def view_of(census_row):
    if not census_row or census_row.get("reference") is None:
        return "no_reference"
    visible = set(census_row.get("visible_primitives") or [])
    if visible <= HALFWAY_ONLY:
        return "halfway"
    if "goal_left" in visible or "goal_right" in visible:
        return "end"
    return "other"


def frame_verdict(result_row):
    """rejected / bad / good by the census's own rule on the official per-frame error."""
    return CENSUS.classify_model_frame(result_row)


def census_table(results, census):
    """Per variant and evidence, per census attribution and per view: frames, accepted,
    solved (accepted and median within 2 m), and gains / losses against the library."""
    verdicts = {
        label: {(r["sequence"], r["frame_id"]): r for r in result["frames"]}
        for label, result in results.items()
    }
    baseline = verdicts["library"]
    keyed_census = {(r["sequence"], r["frame_id"]): r for r in census.values()}
    table = {}
    for label, frames in verdicts.items():
        groups = defaultdict(list)
        for key in frames:
            census_row = keyed_census.get(key)
            attribution = census_row["attribution"] if census_row else "uncensused"
            groups[f"attribution:{attribution}"].append(key)
            groups[f"view:{view_of(census_row)}"].append(key)
            groups["all"].append(key)
        summary = {}
        for group, keys in sorted(groups.items()):
            mine = [frame_verdict(frames[k]) for k in keys]
            theirs = [frame_verdict(baseline[k]) for k in keys]
            medians = [
                frames[k]["error"]["median_m"]
                for k in keys
                if frames[k]["accepted"] and frames[k]["error"]["median_m"] is not None
            ]
            summary[group] = {
                "frames": len(keys),
                "accepted": sum(v != "rejected" for v in mine),
                "solved": sum(v == "good" for v in mine),
                "median_of_frame_medians_m": round(float(np.median(medians)), 2)
                if medians
                else None,
                "gained_vs_library": sum(m == "good" and t != "good" for m, t in zip(mine, theirs)),
                "lost_vs_library": sum(m != "good" and t == "good" for m, t in zip(mine, theirs)),
            }
        overall = results[label]["overall"]
        table[label] = {
            "official": {
                "accepted_frames": overall["accepted_frames"],
                "frames": overall["frames"],
                "all_target_fraction_within_2m": round(overall["all_target_fraction_within_2m"], 3),
                "median_m": round(overall["accepted_point_error"]["median_m"] or float("nan"), 2),
                "p95_m": round(overall["accepted_point_error"]["p95_m"] or float("nan"), 2),
                "failure_reasons": overall["failure_reasons"],
            },
            "groups": summary,
        }
    return table


def print_table(table):
    groups = [
        "all",
        "attribution:good",
        "attribution:geometry_degenerate",
        "attribution:fitter_fault",
        "attribution:network_contaminated",
        "attribution:no_reference",
        "view:halfway",
        "view:end",
    ]
    print(
        f"\n{'variant':<26} {'official acc':>12} {'w2m_all':>8} {'med':>6} {'p95':>6}  "
        + "  ".join(f"{g.split(':')[-1][:12]:>18}" for g in groups)
    )
    print(
        f"{'':<26} {'':>12} {'':>8} {'':>6} {'':>6}  "
        + "  ".join(f"{'acc/solved/n':>18}" for _ in groups)
    )
    for label, entry in table.items():
        o = entry["official"]
        cells = []
        for g in groups:
            s = entry["groups"].get(g)
            cells.append(
                f"{s['accepted']:>5}/{s['solved']:>4}/{s['frames']:<4}" if s else f"{'-':>18}"
            )
        print(
            f"{label:<26} {o['accepted_frames']:>5}/{o['frames']:<6} {o['all_target_fraction_within_2m']:>8.3f} {o['median_m']:>6.2f} {o['p95_m']:>6.2f}  "
            + "  ".join(f"{c:>18}" for c in cells)
        )


def load_census(path):
    if path is None or not Path(path).exists():
        return {}
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {row["frame"]: row for row in rows}


def load_done(path):
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["frame"], row["variant"], row["evidence"])] = row
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("maps_dir")
    parser.add_argument("manifest")
    parser.add_argument("protocol")
    parser.add_argument(
        "--census", default=None, help="census-rows.jsonl (default: <maps_dir>/census-rows.jsonl)"
    )
    parser.add_argument(
        "--reference-predictions",
        default=None,
        help="PnLCalib predictions, positions for comparison only",
    )
    parser.add_argument(
        "--corpus-summary",
        default=None,
        help="corpus-summary.json with game ids (default: <maps_dir>/../../corpus-summary.json)",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    maps_dir = Path(args.maps_dir)
    out = maps_dir / "variants"
    out.mkdir(exist_ok=True)
    protocol = dict(pk.DEFAULT_PROTOCOL)
    census = load_census(args.census or maps_dir / "census-rows.jsonl")
    paths = sorted(maps_dir.glob("*.npz"))
    fits_path = out / "frame-fits.jsonl"
    done = load_done(fits_path)
    with fits_path.open("a") as sink:
        print(f"pass 1: library on {len(paths)} frames, raw and oracle evidence")
        run_jobs(
            paths,
            protocol,
            census,
            lambda p: [("library", m, None) for m in MODES],
            done,
            sink,
            args.workers,
        )
        results = {
            "library": evaluate(done, "library", "raw", out, maps_dir, args.manifest, args.protocol)
        }
        priors = sequence_priors(
            done, results["library"], reference_positions(args.reference_predictions)
        )
        (out / "sequence-camera-priors.json").write_text(json.dumps(priors, indent=1))
        for sequence, entry in priors.items():
            print(
                f"  {sequence}: held {entry['accepted']['position']} over {entry['accepted']['frames']} accepted frames"
                f" (mad {entry['accepted'].get('mad_m')}); good-frame prior {entry['good']['position']} over {entry['good']['frames']};"
                f" pnlcalib {entry.get('pnlcalib_median_position')}"
            )

        game_of = games_of(args.corpus_summary or maps_dir.parent.parent / "corpus-summary.json")
        all_priors = {"library": priors}

        def position_for(variant, sequence):
            if variant == "pinhole":
                return None
            source, kind = PRIOR_OF[variant]
            return np.asarray(all_priors[source][sequence][kind]["position"])

        def jobs_for(variants):
            def jobs(path):
                sequence = str(np.load(path)["sequence"])
                return [
                    (variant, mode, position_for(variant, sequence))
                    for variant in variants
                    for mode in MODES
                ]

            return jobs

        print("pass 2: pinhole variants with library-derived positions")
        run_jobs(paths, protocol, census, jobs_for(PASSES[0]), done, sink, args.workers)
        all_priors["pinhole"] = pinhole_priors(done, game_of)
        (out / "sequence-camera-priors.json").write_text(json.dumps(all_priors, indent=1))
        for sequence, entry in all_priors["pinhole"].items():
            print(
                f"  {sequence} (game {entry['game_id']}): pinhole-derived position {entry['sequence']['position']} over {entry['sequence']['frames']} frames"
                f" (mad {entry['sequence'].get('mad_m')}); game-level {entry['game']['position']} over {entry['game']['frames']}"
            )
        print("pass 3: held at the free pinhole fitter's own positions, per sequence and per game")
        run_jobs(paths, protocol, census, jobs_for(PASSES[1]), done, sink, args.workers)
    for variant in VARIANTS:
        for mode in MODES:
            label = variant if mode == "raw" else f"{variant}-oracle"
            if label not in results:
                results[label] = evaluate(
                    done, variant, mode, out, maps_dir, args.manifest, args.protocol
                )
    table = census_table(results, census)
    (out / "census-by-variant.json").write_text(json.dumps(table, indent=1))
    print_table(table)


if __name__ == "__main__":
    main()
