"""Conservative OCR evidence on sampled shirt crops; roster identity remains separate."""

import shutil
import subprocess
import tempfile
from collections import Counter
from io import StringIO
from pathlib import Path

import cv2
import pandas as pd

from soccerviz.core.data import write_json


def run_jersey(video_run: Path, out: Path):
    executable = shutil.which("tesseract")
    if executable is None:
        raise RuntimeError("Install Tesseract to run the optional local jersey OCR baseline")
    detections = pd.read_parquet(video_run / "detections.parquet")
    records = []
    with tempfile.TemporaryDirectory(prefix="soccerviz-jersey-") as tmp:
        crop_path = Path(tmp) / "shirt.png"
        for source in sorted((video_run / "frames").glob("*.jpg")):
            if "annotated" in source.stem:
                continue
            frame = int(source.stem)
            image = cv2.imread(str(source))
            rows = detections[
                (detections.frame_id == frame) & (detections.role_hypothesis == "player")
            ]
            for row in rows.itertuples():
                width, height = row.bbox_x1 - row.bbox_x0, row.bbox_y1 - row.bbox_y0
                x0, x1 = (
                    max(0, int(row.bbox_x0 + 0.15 * width)),
                    min(image.shape[1], int(row.bbox_x1 - 0.15 * width)),
                )
                y0, y1 = (
                    max(0, int(row.bbox_y0 + 0.15 * height)),
                    min(image.shape[0], int(row.bbox_y0 + 0.65 * height)),
                )
                crop = image[y0:y1, x0:x1]
                token, confidence = None, None
                if crop.shape[0] >= 12 and crop.shape[1] >= 8:
                    gray = cv2.cvtColor(cv2.resize(crop, None, fx=4, fy=4), cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(str(crop_path), gray)
                    result = subprocess.run(
                        [
                            executable,
                            str(crop_path),
                            "stdout",
                            "--psm",
                            "7",
                            "-c",
                            "tessedit_char_whitelist=0123456789",
                            "tsv",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=15,
                    )
                    parsed = pd.read_csv(StringIO(result.stdout), sep="\t", dtype={"text": str})
                    valid = parsed[
                        parsed.text.fillna("").str.fullmatch(r"\d{1,2}") & (parsed.conf >= 0)
                    ]
                    if len(valid):
                        best = valid.sort_values("conf", ascending=False).iloc[0]
                        token, confidence = str(best.text), float(best.conf)
                records.append(
                    {
                        "frame_id": frame,
                        "tracklet_id": row.tracklet_id,
                        "detection_id": row.detection_id,
                        "jersey_token": token,
                        "ocr_confidence": confidence,
                        "source": "tesseract_shirt_crop",
                        "verified": False,
                    }
                )
    table = pd.DataFrame(records)
    hypotheses = []
    for tid, group in table.groupby("tracklet_id"):
        strong = group[(group.ocr_confidence >= 80) & group.jersey_token.notna()]
        votes = Counter(strong.jersey_token)
        winner = votes.most_common(1)
        accepted = bool(winner and winner[0][1] >= 2 and len(votes) == 1)
        hypotheses.append(
            {
                "tracklet_id": int(tid),
                "jersey_hypothesis": winner[0][0] if accepted else None,
                "supporting_frames": winner[0][1] if winner else 0,
                "status": "provisional" if accepted else "abstain",
                "player_identity": None,
                "reason": "two consistent high-confidence OCR frames required; roster unavailable",
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "observations.parquet", index=False)
    pd.DataFrame(hypotheses).to_parquet(out / "hypotheses.parquet", index=False)
    report = {
        "crops": len(table),
        "crops_with_digit_candidate": int(table.jersey_token.notna().sum()),
        "tracklets_reviewed": len(hypotheses),
        "provisional_jersey_hypotheses": sum(h["status"] == "provisional" for h in hypotheses),
        "named_identities": 0,
        "ocr_accuracy": None,
        "limitations": [
            "Tiny broadcast crops often lack a visible number",
            "OCR confidence is not calibrated correctness; all accepted numbers remain provisional",
            "No roster is available and no player names are inferred",
        ],
    }
    write_json(out / "report.json", report)
    return report
