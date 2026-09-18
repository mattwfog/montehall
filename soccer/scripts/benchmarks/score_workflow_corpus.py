"""Score completed workflow-corpus runs against the frozen corpus ground truth.

Three separate prediction sets per run, scored with the same frozen protocol as the
detector benchmark, so pipeline decisions are never blended with raw detector output:
  persons+ball-raw : every person box (roles collapsed) plus every raw ball candidate box
  ball-confirmed   : ball centers the causal ball track confirmed ("detected")
  ball-estimated   : ball centers the track carried as "tentative" or "estimated"
Scores are the pipeline's own confidences for raw output and 1.0 for the binary track
decisions. Outputs land beside the ground truth in artifacts/workflow-corpus/evaluation
and a compact tracked summary in results/experiments. Existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from soccerviz.candidates.vision_benchmark import evaluate
from soccerviz.core.assets import sha256
from soccerviz.core.data import write_json

CORPUS = Path("artifacts/workflow-corpus")
CONFIRMED = {"detected"}
ESTIMATED = {"tentative", "estimated"}


def frame_index(sequence, run):
    """Map run frame_id to the eval manifest identity via the clip's lossless frame map."""
    mapping = json.loads((CORPUS / "videos5hz" / sequence / "frame-map.json").read_text())
    report = json.loads((run / "report.json").read_text())
    if report["source_sha256"] != mapping["source_sha256"]:
        raise ValueError(f"{sequence}: run source differs from the sampled corpus video")
    by_video_frame = {row["video_frame"]: row for row in mapping["frames"]}
    frames = pd.read_parquet(run / "frames.parquet")
    index = {}
    for row in frames.to_dict("records"):
        source = by_video_frame[int(row["source_frame"])]
        if abs(float(row["timestamp_s"]) - source["timestamp_s"]) > 0.001:
            raise ValueError(f"{sequence}: sampled timestamps differ from the frame map")
        index[int(row["frame_id"])] = {
            "sequence": sequence,
            "frame_id": int(source["source_frame"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "timestamp_s": float(row["timestamp_s"]),
            "image_sha256": source["sha256"],
        }
    return index


def person_and_raw_ball(run, index):
    detections = pd.read_parquet(run / "detections.parquet")
    candidates = pd.read_parquet(run / "ball_candidates.parquet")
    per_frame = {fid: [] for fid in index}
    for row in detections.to_dict("records"):
        per_frame[int(row["frame_id"])].append(
            {
                "bbox_xyxy": [row["bbox_x0"], row["bbox_y0"], row["bbox_x1"], row["bbox_y1"]],
                "score": float(row["confidence"]),
                "label": "person",
                "source_class": row["role_hypothesis"],
            }
        )
    for row in candidates.to_dict("records"):
        per_frame[int(row["frame_id"])].append(
            {
                "bbox_xyxy": [row["x0"], row["y0"], row["x1"], row["y1"]],
                "score": float(row["confidence"]),
                "label": "ball",
                "source_class": "ball_candidate",
                "rank": int(row["rank"]),
            }
        )
    return per_frame


def ball_centers(run, index, statuses):
    tracks = pd.read_parquet(run / "ball_tracks.parquet")
    per_frame = {fid: [] for fid in index}
    for row in tracks.to_dict("records"):
        if row["status"] in statuses and row["center_xy"] is not None:
            per_frame[int(row["frame_id"])].append(
                {
                    "center_xy": [float(v) for v in row["center_xy"]],
                    "score": 1.0,
                    "label": "ball",
                    "source_class": row["status"],
                }
            )
    return per_frame


def envelope(schema, manifest_hash, backend, kind, frames):
    return {
        "schema": schema,
        "manifest_sha256": manifest_hash,
        "purpose": "development",
        "prediction_kind": kind,
        "backend": backend,
        "checkpoint_hashes": {
            name: spec["sha256"] for name, spec in backend.get("checkpoints", {}).items()
        },
        "complete_manifest": True,
        "frames": frames,
    }


def score(jobs_path, evaluation, results_root, tag):
    jobs = json.loads(jobs_path.read_text())
    manifest = evaluation / "manifest.json"
    manifest_hash = sha256(manifest)
    runs = {}
    presets, backends = set(), {}
    for sequence, job in sorted(jobs.items()):
        if job.get("status") != "completed":
            raise ValueError(f"{sequence}: job {job.get('job_id')} is not completed")
        run = Path(job["video_run"])
        configuration = json.loads((run / "configuration.json").read_text())
        presets.add(configuration["preset"])
        backends[sequence] = configuration["detector"]
        runs[sequence] = run
    if len(presets) != 1:
        raise ValueError(f"Runs mix presets {sorted(presets)}; score one preset at a time")
    preset = presets.pop()
    backend = {
        **next(iter(backends.values())),
        "preset": preset,
        "runs": {s: str(r) for s, r in runs.items()},
        "note": "Workflow ran at its operating threshold; no low-score capture floor exists.",
    }
    kinds = {
        "persons-ball-raw": ("vision-predictions/v1", person_and_raw_ball),
        "ball-confirmed": (
            "vision-center-predictions/v1",
            lambda run, index: ball_centers(run, index, CONFIRMED),
        ),
        "ball-estimated": (
            "vision-center-predictions/v1",
            lambda run, index: ball_centers(run, index, ESTIMATED),
        ),
    }
    summary = {
        "schema": "workflow-corpus-score/v1",
        "preset": preset,
        "tag": tag,
        "manifest_sha256": manifest_hash,
        "ground_truth_sha256": sha256(evaluation / "ground-truth.json"),
        "jobs_sha256": sha256(jobs_path),
        "runs": {
            s: {"path": str(r), "report_sha256": sha256(r / "report.json")} for s, r in runs.items()
        },
        "results": {},
    }
    for kind, (schema, builder) in kinds.items():
        frames = []
        for sequence, run in runs.items():
            index = frame_index(sequence, run)
            per_frame = builder(run, index)
            for fid, identity in index.items():
                frames.append({**identity, "detections": per_frame[fid]})
        predictions_path = evaluation / f"{tag}-{kind}-predictions.json"
        results_path = evaluation / f"{tag}-{kind}-results.json"
        for path in (predictions_path, results_path):
            if path.exists():
                raise FileExistsError(f"{path} exists; choose a new --tag")
        write_json(predictions_path, envelope(schema, manifest_hash, backend, kind, frames))
        result = evaluate(manifest, predictions_path, evaluation / "ground-truth.json")
        write_json(results_path, result)
        overall = result["overall"]
        summary["results"][kind] = {
            "predictions_sha256": sha256(predictions_path),
            "results_sha256": sha256(results_path),
            "overall": overall,
            "by_sequence": {
                s: (v["classes"] if v.get("classes") else {"ball_center": v["ball_center"]})
                for s, v in result["by_sequence"].items()
            },
        }
    out = results_root / f"workflow-corpus-{tag}-summary.json"
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new --tag")
    write_json(out, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=CORPUS / "jobs.json")
    parser.add_argument("--evaluation", type=Path, default=CORPUS / "evaluation")
    parser.add_argument("--results-root", type=Path, default=Path("results/experiments"))
    parser.add_argument("--tag", required=True, help="Output prefix, e.g. rf-soccer-20260908")
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum():
        parser.error("tag must be alphanumeric with dashes")
    summary = score(args.jobs, args.evaluation, args.results_root, args.tag)
    for kind, block in summary["results"].items():
        overall = block["overall"]
        if overall.get("classes"):
            person, ball = overall["classes"]["person"], overall["ball_center"]
            print(
                kind,
                "person P/R",
                person["precision"],
                person["recall"],
                "ball-center P/R",
                ball["precision"],
                ball["recall"],
            )
        else:
            ball = overall["ball_center"]
            print(kind, "ball-center P/R", ball["precision"], ball["recall"], "tp", ball.get("tp"))


if __name__ == "__main__":
    main()
