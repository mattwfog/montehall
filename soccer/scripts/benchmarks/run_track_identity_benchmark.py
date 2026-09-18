"""Track-level jersey identity on the frozen SoccerNet sequences, staged and unrolled.

`prepare` cuts one torso crop per ground-truth player box on the benchmark's
frames (oracle tracks: the tracking half is held fixed so the reading half is
measured alone), `infer` runs the classifier on them (on Spark, where torch is),
and `evaluate` reports, unrolled so no stage hides another:

1. per-crop forced number accuracy on labelled crops, open (100 numbers);
2. the same with the lineup restriction;
3. per-track identity with open aggregation;
4. per-track identity with lineup restriction and aggregation, the shipping rule.

Each level reports decided / correct / wrong / undecided over the labelled tracks,
plus accuracy over decided and coverage over labelled. The lineup of a
(sequence, team) is the set of jersey numbers the SoccerNet labels give that team
in the sequence: a stand-in for a provider lineup, stated as such. SoccerNet is
GPL and evaluation-only; nothing here trains on it. Ground truth is read only in
`prepare` (to define the oracle tracks and crops) and `evaluate`.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

from soccerviz.candidates.jersey_model import jersey_label
from soccerviz.core.assets import sha256
from soccerviz.vision import track_identity

SEQUENCES = ("SNGS-021", "SNGS-045", "SNGS-089")
SOURCES = "artifacts/vision-benchmark/sources/valid"
ROLES = ("player", "goalkeeper")


def torso_bounds(x0, y0, x1, y1):
    """The upstream JNR torso rule, in crop coordinates: drop the top sixth and bottom third."""
    h = y1 - y0
    return [0, h // 6, x1 - x0, h - h // 3]


def prepare(root, out, sequences=SEQUENCES):
    """Torso crops of every labelled-role GT box on the available frames, with the oracle tracks."""
    out = Path(out)
    (out / "crops").mkdir(parents=True, exist_ok=False)
    crops, truth_rows, sources = [], [], []
    for sequence in sequences:
        folder = Path(root) / SOURCES / sequence
        labels_path = folder / "Labels-GameState.json"
        data = json.loads(labels_path.read_text())
        sources.append(
            {"sequence": sequence, "path": str(labels_path), "sha256": sha256(labels_path)}
        )
        frame_of = {i["image_id"]: int(Path(i["file_name"]).stem) for i in data["images"]}
        available = {int(p.stem) for p in (folder / "img1").glob("*.jpg")}
        for annotation in data["annotations"]:
            attrs = annotation.get("attributes") or {}
            if annotation.get("supercategory") != "object" or attrs.get("role") not in ROLES:
                continue
            frame_id = frame_of[annotation["image_id"]]
            if frame_id not in available:
                continue
            box = annotation["bbox_image"]
            x0, y0 = int(np.floor(box["x"])), int(np.floor(box["y"]))
            x1, y1 = int(np.ceil(box["x"] + box["w"])), int(np.ceil(box["y"] + box["h"]))
            image = cv2.imread(str(folder / "img1" / f"{frame_id:06d}.jpg"))
            if image is None:
                raise FileNotFoundError(folder / "img1" / f"{frame_id:06d}.jpg")
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
            if x1 - x0 < 4 or y1 - y0 < 8:
                continue
            crop_id = f"{sequence}-{frame_id:06d}-t{annotation['track_id']:03d}"
            path = out / "crops" / f"{crop_id}.png"
            if not cv2.imwrite(str(path), image[y0:y1, x0:x1]):
                raise RuntimeError(f"Failed to write {path}")
            crops.append(
                {
                    "crop_id": crop_id,
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "track_id": int(annotation["track_id"]),
                    "crop_path": f"crops/{path.name}",
                    "crop_sha256": sha256(path),
                    "crop_width": x1 - x0,
                    "crop_height": y1 - y0,
                    "torso_bounds_in_crop_xyxy": torso_bounds(x0, y0, x1, y1),
                }
            )
            truth_rows.append(
                {
                    "crop_id": crop_id,
                    "sequence": sequence,
                    "track_id": int(annotation["track_id"]),
                    "team": attrs.get("team"),
                    "role": attrs.get("role"),
                    "number": jersey_label(attrs.get("jersey")),
                }
            )
    manifest = {
        "schema": "jersey-crops/v1",
        "purpose": "oracle-track identity benchmark: GT boxes define crops and tracks; numbers are never read here",
        "ground_truth_used_in_sampling": True,
        "oracle_tracks": True,
        "sources": sources,
        "torso_rule": "top sixth and bottom third of the person box removed (upstream JNR rule)",
        "crops": crops,
    }
    (out / "crop-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    truth = {
        "schema": "track-identity-truth/v1",
        "crop_manifest_sha256": sha256(out / "crop-manifest.json"),
        "evaluation_only": True,
        "rows": truth_rows,
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    by_sequence = collections.Counter(c["sequence"] for c in crops)
    tracks = {(r["sequence"], r["track_id"]) for r in truth_rows}
    labelled = {(r["sequence"], r["track_id"]) for r in truth_rows if r["number"] is not None}
    summary = {
        "crops": len(crops),
        "crops_by_sequence": dict(by_sequence),
        "tracks": len(tracks),
        "labelled_tracks": len(labelled),
    }
    (out / "prepare-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def lineups(truth_rows):
    """(sequence, team) → sorted jersey numbers the labels give that team in the sequence."""
    numbers = collections.defaultdict(set)
    for row in truth_rows:
        if row["number"] is not None and row["team"] is not None:
            numbers[(row["sequence"], row["team"])].add(row["number"])
    return {key: sorted(values) for key, values in numbers.items()}


def _track_summary(decisions, truth_by_track):
    decided = {k: d for k, d in decisions.items() if d["number"] is not None}
    correct = sum(1 for k, d in decided.items() if d["number"] == truth_by_track[k])
    return {
        "labelled_tracks": len(decisions),
        "decided": len(decided),
        "correct": correct,
        "wrong": len(decided) - correct,
        "undecided": len(decisions) - len(decided),
        "accuracy_over_decided": correct / max(1, len(decided)),
        "coverage_over_labelled": len(decided) / max(1, len(decisions)),
        "correct_coverage": correct / max(1, len(decisions)),
    }


def evaluate(out, predictions_dir, decision=track_identity.DEFAULT_DECISION):
    out, predictions_dir = Path(out), Path(predictions_dir)
    manifest = json.loads((out / "crop-manifest.json").read_text())
    truth = json.loads((out / "truth.json").read_text())
    predictions = json.loads((predictions_dir / "predictions.json").read_text())
    if predictions["crop_manifest_sha256"] != sha256(out / "crop-manifest.json"):
        raise ValueError("Predictions are not bound to this crop manifest")
    if truth["crop_manifest_sha256"] != sha256(out / "crop-manifest.json"):
        raise ValueError("Truth is not bound to this crop manifest")
    stored = np.load(predictions_dir / "distributions.npz")
    distribution_of = dict(zip(stored["crop_ids"].tolist(), stored["probabilities"]))
    missing = [c["crop_id"] for c in manifest["crops"] if c["crop_id"] not in distribution_of]
    if missing:
        raise ValueError(f"{len(missing)} crops have no distribution, e.g. {missing[0]}")
    truth_rows = truth["rows"]
    lineup_of = lineups(truth_rows)
    labelled = [r for r in truth_rows if r["number"] is not None]
    # Levels 1 and 2: per crop.
    open_hits = lineup_hits = 0
    for row in labelled:
        p = distribution_of[row["crop_id"]]
        open_hits += int(np.argmax(p[: track_identity.NUM_NUMBERS]) == row["number"])
        restricted = track_identity.restrict_to_lineup(
            p, lineup_of.get((row["sequence"], row["team"]))
        )
        lineup_hits += int(np.argmax(restricted[: track_identity.NUM_NUMBERS]) == row["number"])
    per_crop = {
        "labelled_crops": len(labelled),
        "open_forced_accuracy": open_hits / max(1, len(labelled)),
        "lineup_forced_accuracy": lineup_hits / max(1, len(labelled)),
        "legibility_median": float(
            np.median(
                [1.0 - distribution_of[r["crop_id"]][track_identity.ILLEGIBLE] for r in labelled]
            )
        ),
    }
    # Levels 3 and 4: per track.
    frames = collections.defaultdict(list)
    for row in truth_rows:
        frames[(row["sequence"], row["track_id"])].append(distribution_of[row["crop_id"]])
    truth_by_track = {}
    team_of_track = {}
    for row in labelled:
        truth_by_track[(row["sequence"], row["track_id"])] = row["number"]
        team_of_track[(row["sequence"], row["track_id"])] = row["team"]
    labelled_frames = {k: v for k, v in frames.items() if k in truth_by_track}
    open_decisions = track_identity.identify_tracks(labelled_frames, {}, decision)
    lineup_decisions = track_identity.identify_tracks(
        labelled_frames,
        {k: lineup_of.get((k[0], team_of_track[k])) for k in labelled_frames},
        decision,
    )
    per_track_rows = [
        {
            "sequence": k[0],
            "track_id": k[1],
            "truth": truth_by_track[k],
            "frames": len(labelled_frames[k]),
            "open": open_decisions[k],
            "lineup": lineup_decisions[k],
        }
        for k in sorted(labelled_frames)
    ]
    result = {
        "schema": "track-identity-evaluation/v1",
        "crop_manifest_sha256": sha256(out / "crop-manifest.json"),
        "predictions_sha256": sha256(predictions_dir / "predictions.json"),
        "truth_sha256": sha256(out / "truth.json"),
        "classifier": predictions.get("classifier"),
        "decision": dict(decision),
        "lineup_source": "jersey numbers the SoccerNet labels give each team in the sequence (stand-in for a provider lineup)",
        "oracle_tracks": True,
        "per_crop": per_crop,
        "per_track_open": _track_summary(open_decisions, truth_by_track),
        "per_track_lineup": _track_summary(lineup_decisions, truth_by_track),
        "by_sequence": {
            s: {
                "open": _track_summary(
                    {k: v for k, v in open_decisions.items() if k[0] == s}, truth_by_track
                ),
                "lineup": _track_summary(
                    {k: v for k, v in lineup_decisions.items() if k[0] == s}, truth_by_track
                ),
            }
            for s in sorted({k[0] for k in labelled_frames})
        },
        "lineups": {f"{s}/{t}": v for (s, t), v in sorted(lineup_of.items())},
        "tracks": per_track_rows,
        "limitations": [
            "Oracle tracks: GT boxes and identities define the tracks, so tracking errors are excluded by design",
            "Lineups come from the labels themselves, not a provider",
            "GT jersey attributes identify the player's number, not its visibility in any crop",
            "Three sequences from three games; 44 labelled tracks",
        ],
    }
    (predictions_dir / "track-identity-evaluation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return {k: result[k] for k in ("per_crop", "per_track_open", "per_track_lineup", "by_sequence")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("infer")
    p.add_argument("--crop-manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("evaluate")
    p.add_argument("--benchmark", type=Path, required=True, help="the prepare output folder")
    p.add_argument("--predictions", type=Path, required=True, help="the infer output folder")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.command == "prepare":
        print(json.dumps(prepare(root, args.out), indent=2))
    elif args.command == "infer":
        from soccerviz.candidates.jersey_classifier import infer

        result = infer(args.crop_manifest, args.checkpoint, args.out, device=args.device)
        print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2))
    else:
        print(json.dumps(evaluate(args.benchmark, args.predictions), indent=2))


if __name__ == "__main__":
    main()
