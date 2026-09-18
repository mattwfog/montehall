"""Exercise video → tactical features with explicit unresolved orientation and domain shift."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from soccerviz.core.analysis import FEATURES, snapshot_features
from soccerviz.core.data import Match, write_json


def audit_video_features(video_run: Path, artifacts: Path):
    frames = pd.read_parquet(video_run / "frames.parquet")
    state = pd.read_parquet(video_run / "state.parquet")
    model = joblib.load(artifacts / "baseline.joblib")["model"]
    times = frames.timestamp_s.to_numpy()
    ball = np.full((len(frames), 2), np.nan)
    for row in state[state.entity == "ball_candidate"].itertuples():
        ball[int(row.frame_id)] = [row.x_m, row.y_m]
    rows = []
    for frame in frames.itertuples():
        people = state[
            (state.frame_id == frame.frame_id)
            & (state.entity == "player")
            & state.team_cluster.notna()
        ].drop_duplicates("tracklet_id")
        if people.empty:
            continue
        xy = np.full((len(frames), len(people), 2), np.nan)
        xy[frame.frame_id] = people[["x_m", "y_m"]].to_numpy()
        for direction in (1, -1):
            match = Match(
                0,
                frames.source_frame.to_numpy(),
                times,
                np.ones(len(frames), dtype=int),
                xy,
                ball,
                people.team_cluster.to_numpy(int),
                people.tracklet_id.astype(str).tolist(),
                np.tile([direction, -direction], (len(frames), 1)),
                pd.DataFrame(),
            )
            for team in (0, 1):
                features = snapshot_features(match, int(frame.frame_id), team)
                if features is None:
                    continue
                prediction = float(model.predict_proba(pd.DataFrame([features])[FEATURES])[0, 1])
                rows.append(
                    {
                        "frame_id": frame.frame_id,
                        "timestamp_s": frame.timestamp_s,
                        "anonymous_team_cluster": team,
                        "assumed_attack_direction": direction if team == 0 else -direction,
                        "unvalidated_progression_score": prediction,
                        "status": "domain_transfer_diagnostic",
                        "possession_team": None,
                        **features,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_parquet(video_run / "tactical-feature-audit.parquet", index=False)
    report = {
        "frames": len(frames),
        "frames_with_usable_features": int(result.frame_id.nunique()) if len(result) else 0,
        "orientation_scenarios_scored": len(result),
        "validated_tactical_predictions": 0,
        "limitations": [
            "Team identity, attacking direction and possession are unresolved",
            "Both directions are scored for each anonymous team as a sensitivity audit",
            "Ball detections and projected positions are not ground-truth validated",
            "Classifier was trained on provider tracking; video-domain calibration is unmeasured",
            "These scores do not drive recommendations or enter training",
        ],
    }
    write_json(video_run / "bridge-report.json", report)
    return report
