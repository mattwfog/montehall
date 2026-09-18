"""Fine-tune evaluation: ball/rim coverage, fine-tuned vs pretrained.

The product metric is COVERAGE (fraction of frames with a ball detection),
not COCO mAP — ball coverage is what gates possession recall and shooter
attribution downstream. Runs both detectors over the same sampled frames
from the clip and reports coverage + agreement.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from montehall_cv.pipeline.detect import RFDetrDetector
from montehall_cv.pipeline.video import decode_frames
from montehall_cv.store.records import DetClass


CONF_SWEEP = (0.2, 0.3, 0.4)


def evaluate(
    video: Path,
    finetuned: Path,
    sample_every: int = 10,
    max_frames: int = 900,
    batch: int = 16,
) -> dict:
    """Coverage at every CONF_SWEEP threshold, so a fine-tune whose confidence
    calibration shifted (normal with a fresh head) shows a moved curve instead
    of being indistinguishable from a dead model at one hard threshold."""
    started = time.monotonic()
    frames = [
        f for f in decode_frames(video, every_n=sample_every, max_frames=max_frames)
    ]

    counts: dict[str, dict[str, dict[str, float]]] = {}
    for name, weights in (("pretrained", None), ("finetuned", finetuned)):
        detector = RFDetrDetector(
            threshold=min(CONF_SWEEP), batch_size=batch, weights=weights
        )
        tallies = {
            conf: {"ball_frames": 0, "rim_frames": 0, "person_total": 0}
            for conf in CONF_SWEEP
        }
        max_conf: dict[int, float] = {}
        for start in range(0, len(frames), batch):
            chunk = frames[start : start + batch]
            for dets in detector.detect([f.image for f in chunk]):
                for c, conf in zip(dets.cls, dets.conf):
                    max_conf[int(c)] = max(max_conf.get(int(c), 0.0), float(conf))
                for conf in CONF_SWEEP:
                    keep = dets.conf >= conf
                    cls = dets.cls[keep]
                    tallies[conf]["ball_frames"] += int(
                        (cls == int(DetClass.BALL)).any()
                    )
                    tallies[conf]["person_total"] += int(
                        (cls == int(DetClass.PERSON)).sum()
                    )
                    tallies[conf]["rim_frames"] += int((cls == int(DetClass.RIM)).any())
        counts[name] = {
            f"conf_{conf}": {
                "ball_coverage": round(t["ball_frames"] / len(frames), 3),
                "rim_coverage": round(t["rim_frames"] / len(frames), 3),
                "persons_per_frame": round(t["person_total"] / len(frames), 1),
            }
            for conf, t in tallies.items()
        }
        counts[name]["max_conf_by_class"] = {
            DetClass(c).name: round(v, 3) for c, v in sorted(max_conf.items())
        }
        del detector

    return {
        "frames_evaluated": len(frames),
        **counts,
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=900)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.video, args.finetuned, args.sample_every, args.max_frames),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
