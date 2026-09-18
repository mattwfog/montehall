"""Print the detector comparison table in docs/experiments/detector-training.md from the JSONs.

Each column is a prefix under artifacts/vision-benchmark/: <prefix>-results.json
(vision_benchmark evaluate) and <prefix>-tracking-results.json
(compare_detector_tracking). Values are read, never typed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "artifacts/vision-benchmark"

ROWS = (
    ("Person precision", lambda r, t: pct(r["overall"]["classes"]["person"]["precision"])),
    ("Person recall", lambda r, t: pct(r["overall"]["classes"]["person"]["recall"])),
    (
        "Recall on small person boxes",
        lambda r, t: pct(r["overall"]["classes"]["person"]["size_buckets"]["small"]["recall"]),
    ),
    (
        "Person false positives per frame",
        lambda r, t: f"{r['overall']['classes']['person']['false_positives_per_frame']:.2f}",
    ),
    ("Ball-center precision", lambda r, t: pct(r["overall"]["ball_center"]["precision"])),
    ("Ball-center recall", lambda r, t: pct(r["overall"]["ball_center"]["recall"])),
    ("Two-class box mAP, IoU .50:.95", lambda r, t: pct(r["coco_bbox"]["map"])),
    ("Person tracking HOTA", lambda r, t: f"{t['overall']['HOTA']:.4f}" if t else "n/a"),
    ("Person tracking IDF1", lambda r, t: pct(t["overall"]["IDF1"]) if t else "n/a"),
)


def pct(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def load(prefix):
    results = json.loads((ROOT / f"{prefix}-results.json").read_text())
    tracking_path = ROOT / f"{prefix}-tracking-results.json"
    tracking = json.loads(tracking_path.read_text()) if tracking_path.exists() else None
    return results, tracking


def table(prefixes, labels):
    columns = [load(prefix) for prefix in prefixes]
    lines = [
        "| Metric | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join("---:" for _ in labels) + " |",
    ]
    for name, cell in ROWS:
        lines.append(f"| {name} | " + " | ".join(cell(r, t) for r, t in columns) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", action="append", required=True, help="results prefix; repeat")
    parser.add_argument("--label", action="append", default=[], help="column label; repeat")
    args = parser.parse_args()
    labels = args.label + args.prefix[len(args.label) :]
    if len(labels) != len(args.prefix):
        raise SystemExit("Give at most one --label per --prefix")
    print(table(args.prefix, labels))


if __name__ == "__main__":
    main()
