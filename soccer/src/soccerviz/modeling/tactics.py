"""Bounded tactical baselines on provider tracking with match-disjoint evaluation."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from soccerviz.core.analysis import FEATURES, orient
from soccerviz.core.assets import sha256
from soccerviz.core.data import load_match, write_json
from soccerviz.core.model import metrics

PASS_FEATURES = [
    "distance_m",
    "forward_gain_m",
    "lateral_distance_m",
    "lane_clearance_m",
    "receiver_opponent_distance_m",
    "passer_pressure_m",
    "receiver_x_m",
]
FORMATIONS = {
    "4-3-3": [(0, y) for y in np.linspace(0, 1, 4)]
    + [(0.5, y) for y in [0.15, 0.5, 0.85]]
    + [(1, y) for y in [0.1, 0.5, 0.9]],
    "4-4-2": [(0, y) for y in np.linspace(0, 1, 4)]
    + [(0.5, y) for y in np.linspace(0, 1, 4)]
    + [(1, y) for y in [0.3, 0.7]],
    "4-2-3-1": [(0, y) for y in np.linspace(0, 1, 4)]
    + [(0.35, y) for y in [0.3, 0.7]]
    + [(0.7, y) for y in [0.1, 0.5, 0.9]]
    + [(1, 0.5)],
}


def shape_assignment(points):
    """Nearest normalized formation template; descriptive geometry, not role truth."""
    points = np.asarray(points, float)
    finite = np.flatnonzero(np.isfinite(points).all(axis=1))
    if len(finite) != 11:
        return {"shape": "unknown", "template_error": None, "roles": {}}
    keeper = finite[np.argmin(points[finite, 0])]
    players = finite[finite != keeper]
    p = points[players]
    scaled = (p - p.min(axis=0)) / np.maximum(np.ptp(p, axis=0), 1)
    candidates = []
    for name, template in FORMATIONS.items():
        cost = np.linalg.norm(scaled[:, None] - np.asarray(template)[None], axis=2)
        rows, columns = linear_sum_assignment(cost)
        candidates.append((float(cost[rows, columns].mean()), name, columns))
    error, name, columns = min(candidates, key=lambda row: row[0])
    roles = {int(keeper): "deepest_player_proxy"}
    roles.update({int(players[i]): f"{name}:slot-{slot + 1}" for i, slot in enumerate(columns)})
    return {"shape": name, "template_error": error, "roles": roles}


def pass_geometry(ball, receiver, opponents):
    ball, receiver = np.asarray(ball), np.asarray(receiver)
    opponents = opponents[np.isfinite(opponents).all(axis=1)]
    delta = receiver - ball
    length = float(np.linalg.norm(delta))
    if length < 1 or not len(opponents):
        return None
    t = np.clip((opponents - ball) @ delta / length**2, 0, 1)
    closest = ball + t[:, None] * delta
    return dict(
        zip(
            PASS_FEATURES,
            [
                length,
                float(delta[0]),
                float(abs(delta[1])),
                float(np.linalg.norm(opponents - closest, axis=1).min()),
                float(np.linalg.norm(opponents - receiver, axis=1).min()),
                float(np.linalg.norm(opponents - ball, axis=1).min()),
                float(receiver[0]),
            ],
            strict=True,
        )
    )


def lane_simulation(ball, receiver, opponents, draws=64, seed=22):
    """Toy ground-pass simulation. Frequency is conditional on assumed kinematics."""
    rng = np.random.default_rng(seed)
    opponents = opponents[np.isfinite(opponents).all(axis=1)]
    points = np.asarray(ball) + (np.asarray(receiver) - ball) * np.linspace(0.05, 1, 20)[:, None]
    distance = np.linalg.norm(points - ball, axis=1)
    ball_speed = rng.uniform(12, 22, draws)
    defender_speed = rng.uniform(4, 7, (draws, len(opponents)))
    reaction = rng.uniform(0.2, 0.6, (draws, len(opponents)))
    travel = np.linalg.norm(opponents[:, None] - points[None], axis=2)
    opponent_time = reaction[:, :, None] + travel[None] / defender_speed[:, :, None]
    ball_time = distance[None] / ball_speed[:, None]
    intercepted = (opponent_time.min(axis=1) < ball_time).any(axis=1)
    return float((~intercepted).mean())


def candidate_table(match, index, team, passer=None):
    direction = int(match.direction[index, team])
    positions = orient(match.xy[index], direction)
    ball = orient(match.ball[index], direction)
    opponents = positions[match.teams != team]
    if not np.isfinite(ball).all() or np.isfinite(opponents).all(axis=1).sum() < 7:
        return pd.DataFrame()
    own = np.flatnonzero((match.teams == team) & np.isfinite(positions).all(axis=1))
    if len(own) < 7:
        return pd.DataFrame()
    if passer is None:
        passer = int(own[np.argmin(np.linalg.norm(positions[own] - ball, axis=1))])
    rows = []
    for receiver in own:
        if receiver == passer:
            continue
        features = pass_geometry(ball, positions[receiver], opponents)
        if features is not None:
            rows.append(
                {
                    "receiver": match.players[receiver],
                    "receiver_index": int(receiver),
                    "x_m": float(match.xy[index, receiver, 0]),
                    "y_m": float(match.xy[index, receiver, 1]),
                    **features,
                }
            )
    return pd.DataFrame(rows)


def pass_dataset(match):
    rows = []
    for event_id, event in match.events[match.events.Type == "PASS"].iterrows():
        team = 0 if event.Team == "Home" else 1
        prefix = "Home:" if team == 0 else "Away:"
        passer, target = prefix + str(event.From), prefix + str(event.To)
        if passer not in match.players or target not in match.players:
            continue
        index = int(np.searchsorted(match.times, event["Start Time [s]"], side="right")) - 1
        if index < 0 or match.periods[index] != event.Period:
            continue
        candidates = candidate_table(match, index, team, match.players.index(passer))
        if candidates.empty or target not in set(candidates.receiver):
            continue
        candidates["chosen"] = (candidates.receiver == target).astype(int)
        candidates["event_id"] = f"g{match.game}-e{event_id}"
        candidates["game"] = match.game
        candidates["time_s"] = float(match.times[index])
        candidates["index"] = index
        rows.append(candidates)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ranking_metrics(table, score):
    ranked = table.assign(score=score).sort_values("score", ascending=False, kind="stable")
    ranks = []
    for _, group in ranked.groupby("event_id", sort=False):
        ranks.append(int(np.flatnonzero(group.chosen.to_numpy())[0]) + 1)
    ranks = np.array(ranks)
    return {
        "events": len(ranks),
        "top1": float((ranks == 1).mean()),
        "top3": float((ranks <= 3).mean()),
        "mrr": float((1 / ranks).mean()),
    }


def future_events(match, samples, segments):
    labels, shots = [], []
    event_records = match.events.to_dict("records")
    for sample in samples.itertuples():
        end = segments.loc[sample.possession_id, "end_s"]
        team_name = "Home" if sample.team == 0 else "Away"
        future = [
            e
            for e in event_records
            if e["Team"] == team_name
            and e["Period"] == sample.period
            and sample.time_s < e["Start Time [s]"] <= min(sample.time_s + 5, end)
            and e["Type"] in {"PASS", "SHOT", "BALL LOST"}
        ]
        labels.append(future[0]["Type"] if future else "NO_ACTION")
        shots.append(
            int(
                any(
                    e["Team"] == team_name
                    and e["Period"] == sample.period
                    and sample.time_s < e["Start Time [s]"] <= min(sample.time_s + 10, end)
                    and e["Type"] == "SHOT"
                    for e in event_records
                )
            )
        )
        if (
            segments.loc[sample.possession_id, "end_reason"] in {"period_end", "recording_end"}
            and end < sample.time_s + 10
        ):
            shots[-1] = -1  # Unknown future, excluded from shot-model fitting and evaluation.
    return np.array(labels), np.array(shots)


def run_tactics(data: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    matches = {g: load_match(data, g) for g in (1, 2)}
    samples = {
        g: pd.read_parquet(data / "processed" / f"game_{g}" / "samples.parquet") for g in (1, 2)
    }
    possessions = {
        g: pd.read_parquet(data / "processed" / f"game_{g}" / "possessions.parquet").set_index(
            "possession_id"
        )
        for g in (1, 2)
    }
    future = {g: future_events(matches[g], samples[g], possessions[g]) for g in (1, 2)}
    action = RandomForestClassifier(
        n_estimators=120,
        max_depth=5,
        min_samples_leaf=12,
        class_weight="balanced",
        random_state=22,
        n_jobs=4,
    )
    action.fit(samples[1][FEATURES], future[1][0])
    predicted = action.predict(samples[2][FEATURES])
    classes = sorted(set(future[1][0]) | set(future[2][0]))
    majority = pd.Series(future[1][0]).mode().iloc[0]
    action_report = {
        "target": "first same-team pass, shot, loss within 5s or NO_ACTION",
        "accuracy": float(accuracy_score(future[2][0], predicted)),
        "macro_f1": float(f1_score(future[2][0], predicted, average="macro", zero_division=0)),
        "majority_accuracy": float((future[2][0] == majority).mean()),
        "majority_macro_f1": float(
            f1_score(
                future[2][0], np.full(len(predicted), majority), average="macro", zero_division=0
            )
        ),
        "classes": classes,
        "confusion_matrix": confusion_matrix(future[2][0], predicted, labels=classes).tolist(),
        "per_class": classification_report(
            future[2][0], predicted, output_dict=True, zero_division=0
        ),
    }
    value = RandomForestClassifier(
        n_estimators=120, max_depth=4, min_samples_leaf=15, random_state=22, n_jobs=4
    )
    valid_train, valid_eval = future[1][1] >= 0, future[2][1] >= 0
    value.fit(samples[1].loc[valid_train, FEATURES], future[1][1][valid_train])
    positive = list(value.classes_).index(1)
    p = value.predict_proba(samples[2][FEATURES])[:, positive]
    prior = float(future[1][1][valid_train].mean())
    value_report = {
        "target": "shot before 10s or event-derived possession end, not xG or EPV",
        "training_positives": int((future[1][1] == 1).sum()),
        "evaluation_positives": int((future[2][1] == 1).sum()),
        "censored_training": int((~valid_train).sum()),
        "censored_evaluation": int((~valid_eval).sum()),
        "model": metrics(future[2][1][valid_eval], p[valid_eval]),
        "prior": metrics(future[2][1][valid_eval], np.full(valid_eval.sum(), prior)),
    }
    pass_train, pass_eval = pass_dataset(matches[1]), pass_dataset(matches[2])
    ranker = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=22,
        n_jobs=4,
    )
    ranker.fit(pass_train[PASS_FEATURES], pass_train.chosen)
    ranks = ranker.predict_proba(pass_eval[PASS_FEATURES])[:, 1]
    pass_report = {
        "target": "successful-pass recipient imitation; not pass success or optimal action",
        "training_events": int(pass_train.event_id.nunique()),
        "model": ranking_metrics(pass_eval, ranks),
        "nearest_teammate": ranking_metrics(pass_eval, -pass_eval.distance_m.to_numpy()),
        "limitations": [
            "Only annotated completed passes with visible intended recipient",
            "Choice model scores are not calibrated completion probabilities",
        ],
    }
    pass_eval.assign(choice_score=ranks).to_parquet(out / "pass-evaluation.parquet", index=False)
    scaler = StandardScaler().fit(samples[1][FEATURES])
    search = NearestNeighbors(n_neighbors=20).fit(scaler.transform(samples[1][FEATURES]))
    distances, indices = search.kneighbors(scaler.transform(samples[2][FEATURES]))
    retrieval = []
    for i, sample in enumerate(samples[2].itertuples()):
        seen = set()
        for distance, index in zip(distances[i], indices[i], strict=True):
            neighbor = samples[1].iloc[index]
            if neighbor.possession_id in seen:
                continue
            seen.add(neighbor.possession_id)
            retrieval.append(
                {
                    "query_sample": sample.sample_id,
                    "query_game": 2,
                    "neighbor_sample": neighbor.sample_id,
                    "neighbor_game": 1,
                    "neighbor_possession": neighbor.possession_id,
                    "neighbor_time_s": float(neighbor.time_s),
                    "standardized_distance": float(distance),
                    "rank": len(seen),
                }
            )
            if len(seen) == 3:
                break
    pd.DataFrame(retrieval).to_parquet(out / "retrieval.parquet", index=False)
    snapshots, roles = [], []
    options = []
    for game, match in matches.items():
        for sample in samples[game].itertuples():
            i, team = sample.index, sample.team
            direction = int(match.direction[i, team])
            own = orient(match.xy[i, match.teams == team], direction)
            shape = shape_assignment(own)
            own_ids = [
                player
                for player, side in zip(match.players, match.teams, strict=True)
                if side == team
            ]
            roles.extend(
                {
                    "sample_id": sample.sample_id,
                    "game": game,
                    "player_id": own_ids[k],
                    "role_proxy": role,
                    "shape_error": shape["template_error"],
                }
                for k, role in shape["roles"].items()
            )
            phase = (
                "build_up"
                if sample.ball_progress_m < 35
                else "middle_third"
                if sample.ball_progress_m < 70
                else "final_third"
            )
            snapshots.append(
                {
                    "sample_id": sample.sample_id,
                    "game": game,
                    "time_s": sample.time_s,
                    "possession_id": sample.possession_id,
                    "team": team,
                    "phase_proxy": phase,
                    "shape_template": shape["shape"],
                    "shape_error": shape["template_error"],
                    "pressure_proxy": sample.nearest_opponent_m <= 3.0,
                    "support_ahead": sample.teammates_ahead,
                    "width_m": sample.team_width_m,
                    "depth_m": sample.team_depth_m,
                    "advanced_space_proxy": sample.advanced_space_share,
                    "rest_defense_count_proxy": sample.observed_teammates - sample.teammates_ahead,
                    "observed_players": sample.observed_teammates + sample.observed_opponents,
                }
            )
            if game == 2:
                candidates = candidate_table(match, i, team)
                if len(candidates):
                    candidates["choice_score"] = ranker.predict_proba(candidates[PASS_FEATURES])[
                        :, 1
                    ]
                    candidates = candidates.nlargest(3, "choice_score").copy()
                    ball = orient(match.ball[i], direction)
                    opponents = orient(match.xy[i, match.teams != team], direction)
                    candidates["simulated_open_lane_fraction"] = [
                        lane_simulation(ball, orient(match.xy[i, j], direction), opponents)
                        for j in candidates.receiver_index
                    ]
                    candidates["sample_id"] = sample.sample_id
                    options.append(candidates)
    pd.DataFrame(snapshots).to_parquet(out / "snapshots.parquet", index=False)
    pd.DataFrame(roles).to_parquet(out / "role-assignments.parquet", index=False)
    pd.concat(options, ignore_index=True).to_parquet(out / "pass-options.parquet", index=False)
    next_output = samples[2][["sample_id", "possession_id", "time_s"]].copy()
    next_output["observed_next_action"], next_output["predicted_next_action"] = (
        future[2][0],
        predicted,
    )
    next_output["observed_shot_within_10s"], next_output["shot_probability"] = future[2][1], p
    next_output.to_parquet(out / "next-event-predictions.parquet", index=False)
    joblib.dump(
        {
            "action_model": action,
            "shot_model": value,
            "pass_ranker": ranker,
            "retrieval_scaler": scaler,
            "retrieval_index": search,
            "retrieval_samples": samples[1],
            "features": FEATURES,
            "pass_features": PASS_FEATURES,
            "train_game": 1,
            "evaluation_game": 2,
        },
        out / "models.joblib",
    )
    set_pieces = []
    for game, match in matches.items():
        for event_id, event in match.events[match.events.Type == "SET PIECE"].iterrows():
            team = 0 if event.Team == "Home" else 1
            time_s = float(event["Start Time [s]"])
            relevant = possessions[game][
                (possessions[game].team == team)
                & (possessions[game].start_s >= time_s - 0.01)
                & (possessions[game].start_s <= time_s + 0.01)
            ]
            end = float(relevant.iloc[0].end_s) if len(relevant) else time_s
            shot = match.events[
                (match.events.Team == event.Team)
                & (match.events.Type == "SHOT")
                & (match.events.Period == event.Period)
                & (match.events["Start Time [s]"] > time_s)
                & (match.events["Start Time [s]"] <= min(end, time_s + 10))
            ]
            set_pieces.append(
                {
                    "game": game,
                    "event_id": f"g{game}-e{event_id}",
                    "team": team,
                    "time_s": time_s,
                    "restart_type": str(event.Subtype),
                    "shot_within_10s_of_interval": bool(len(shot)),
                    "interval_found": bool(len(relevant)),
                }
            )
    pd.DataFrame(set_pieces).to_parquet(out / "set-pieces.parquet", index=False)
    report = {
        "train_game": 1,
        "evaluation_game": 2,
        "input_sha256": {
            f"game_{g}/{name}": sha256(data / "processed" / f"game_{g}" / name)
            for g in (1, 2)
            for name in ["samples.parquet", "events.parquet", "tracking.npz"]
        },
        "next_action": action_report,
        "shot_surrogate": value_report,
        "pass_choice": pass_report,
        "set_pieces": {
            "intervals": len(set_pieces),
            "with_following_shot": sum(s["shot_within_10s_of_interval"] for s in set_pieces),
            "interpretation": "Descriptive restart retrieval; no corner intervention model",
        },
        "retrieval": {
            "queries": len(samples[2]),
            "neighbors": len(retrieval),
            "cross_match_only": True,
            "analyst_relevance": "unmeasured",
        },
        "descriptive_counts": pd.DataFrame(snapshots)
        .groupby("game")
        .phase_proxy.value_counts()
        .to_dict(),
        "limitations": [
            "Phase, pressing, shape and rest-defense labels are transparent proxies",
            "Formation template fit is not validated tactical role or nominal formation recognition",
            "Shot surrogate is sparse; no goal quality, game score, or managerial context",
            "Similar sequences are feature neighbors, not causal evidence",
            "Pass alternatives assume stationary receivers and simplified ground-ball/defender motion",
        ],
    }
    report["descriptive_counts"] = {
        f"game_{g}/{phase}": int(count)
        for (g, phase), count in report["descriptive_counts"].items()
    }
    write_json(out / "report.json", report)
    return report
