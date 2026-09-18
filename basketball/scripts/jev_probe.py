"""Live probe of the jev adjudicator backend on hand-built possession traces.

Not a benchmark: six synthetic traces chosen to show how the typed judgments
behave on a clean make, a miss, a turnover, an ambiguous scorer, a sparse trace
and a self-contradicting trace. It prints the probabilities the model returned
and the verdict the code composed from them. Needs TYPESAFE_API_KEY.

Usage:
    python scripts/jev_probe.py
"""

from __future__ import annotations

from montehall_cv.harness import adjudicators as adj


def _control(ts: float, entity: int, team: int, x: float, y: float = 25.0) -> dict:
    return {"ts_s": ts, "entity": entity, "team": team, "court_x": x, "court_y": y}


_BASE = {
    "possession_id": 1,
    "start_s": 100.0,
    "end_s": 112.0,
    "duration_s": 12.0,
    "offense_team_cluster": 0,
    "entities_involved": [],
}

CASES: dict[str, dict] = {
    "clean make off a pass": {
        **_BASE,
        "ball_controls": [
            _control(101.0, 3, 0, 40.0, 20.0),
            _control(104.5, 3, 0, 62.0, 24.0),
            _control(108.0, 5, 0, 88.0),
            _control(109.6, 5, 0, 89.5),
        ],
        "shot_events": [{"ts_s": 110.0, "attempt": True, "made": True, "confidence": 0.92}],
        "passes": [
            {"ts_s": 108.0, "from_entity": 3, "to_entity": 5, "to_court_x": 88.0, "to_court_y": 25.0}
        ],
    },
    "miss, then the defense controls": {
        **_BASE,
        "ball_controls": [_control(101, 3, 0, 60), _control(106, 5, 0, 86), _control(111, 9, 1, 84)],
        "shot_events": [{"ts_s": 108.0, "attempt": True, "made": False, "confidence": 0.85}],
        "passes": [
            {"ts_s": 106.0, "from_entity": 3, "to_entity": 5, "to_court_x": 86.0, "to_court_y": 25.0}
        ],
    },
    "no shot, ball changes teams": {
        **_BASE,
        "ball_controls": [
            _control(101, 3, 0, 50),
            _control(104, 3, 0, 60),
            _control(106, 9, 1, 58),
            _control(109, 9, 1, 40),
        ],
        "shot_events": [],
        "passes": [],
    },
    "make with two offensive bodies at the rim": {
        **_BASE,
        "ball_controls": [
            _control(108.8, 5, 0, 88.5),
            _control(109.0, 3, 0, 88.0),
            _control(109.2, 5, 0, 88.6),
            _control(109.4, 3, 0, 88.2),
        ],
        "shot_events": [{"ts_s": 110.0, "attempt": True, "made": True, "confidence": 0.6}],
        "passes": [],
    },
    "sparse: one control sample, low-confidence attempt": {
        **_BASE,
        "ball_controls": [_control(103, 3, 0, 55)],
        "shot_events": [{"ts_s": 111.0, "attempt": True, "made": None, "confidence": 0.35}],
        "passes": [],
    },
    "contradiction: made shot, only the defense has the ball": {
        **_BASE,
        "ball_controls": [_control(101, 9, 1, 30), _control(105, 9, 1, 35), _control(109, 8, 1, 40)],
        "shot_events": [{"ts_s": 110.0, "attempt": True, "made": True, "confidence": 0.9}],
        "passes": [],
    },
}


def main() -> None:
    adjudicator = adj.JevAdjudicator()
    tokens = 0
    for name, trace in CASES.items():
        state, questions = adj.jev_questions(trace)
        response = adjudicator._client.system_one(state, questions)
        tokens += response.usage.input_tokens or 0
        answers = {n: a.model_dump() for n, a in response.answers.items()}
        verdict = adj.compose_jev_verdict(trace, answers)
        outcome_p = {
            k: round(p, 2)
            for k, p in sorted(answers["outcome"]["probabilities"].items(), key=lambda kv: -kv[1])
            if p >= 0.01
        }
        scorer_p = (
            {k: round(p, 2) for k, p in answers["scorer"]["probabilities"].items()}
            if "scorer" in answers
            else None
        )
        print(f"\n[{name}]")
        print(f"  outcome p: {outcome_p}")
        print(f"  scorer p:  {scorer_p}")
        print(f"  model anomaly p: {answers['anomaly']['noul']:.2f}")
        shown = ("outcome", "scorer_entity", "assist_entity", "anomalies")
        print("  verdict:", {k: verdict[k] for k in shown}, "confidence", round(verdict["confidence"], 2))
    print(f"\n{response.model}: {tokens} input tokens for {len(CASES)} possessions")


if __name__ == "__main__":
    main()
