"""Possession adjudicator backends: one trace in, one verdict out.

The harness treats the adjudicator as a swappable likelihood channel, like every
other sensor in the pipeline. Two backends ship:

- ``haiku``: one generative call per possession; the model writes a JSON verdict
  and reports its own confidence.
- ``jev``: TypeSafe's System One model. It generates no text. Code asks narrow
  typed questions over the same trace and reads back probabilities, so the
  verdict's confidence is a distribution the scorer can check for calibration,
  and abstention is a threshold rather than a parsing failure.

Both return the same verdict dict, so the stage schema, the cache and the box
score are backend-agnostic. SDKs are imported lazily; neither is a base
dependency.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from montehall_cv.harness.traces import trace_json
from montehall_cv.store.vlm_cache import VlmCache, content_key

OUTCOMES = ("made_fg", "missed_fg_dreb", "missed_fg_oreb", "turnover", "unclear")

HAIKU_MODEL = "claude-haiku-4-5-20251001"
JEV_MODEL = "jev-latest"

HAIKU_SYSTEM = """You are a basketball statistician applying official scoring conventions to \
possession traces from a computer-vision pipeline. Rules you enforce:

- OUTCOME per FIBA possession definition: a possession ends by made FG, defensive \
rebound after a miss, or turnover; an offensive rebound CONTINUES the possession.
- ASSIST per the FIBA mechanical test: only the LAST pass before the shot counts; \
a pass to a player who scores from the paint is always an assist; a pass to a \
player outside the paint who scores without dribbling is always an assist; with \
dribbles it is an assist only if the shooter did not have to beat their own \
defender. If ball-control data cannot establish a qualifying last pass, there is \
NO assist.
- DOUBT DEFAULTS: doubt about rebound control -> assume control; doubt about act \
of shooting -> assume NOT shooting; unattributable stats belong to the TEAM, \
never to a guessed player.
- The trace may be incomplete (sparse ball detection). Flag anomalies rather than \
inventing facts. Trust shot_events (VLM-adjudicated) over raw ball_controls when \
they conflict.

Respond ONLY with JSON:
{"outcome": "made_fg"|"missed_fg_dreb"|"missed_fg_oreb"|"turnover"|"unclear",
 "scorer_entity": int|null, "assist_entity": int|null,
 "confidence": 0.0-1.0, "anomalies": ["..."], "rationale": "<=2 sentences"}"""


class Adjudicator(Protocol):
    name: str

    def adjudicate(self, trace: dict, cache: VlmCache | None = None) -> dict: ...


class HaikuAdjudicator:
    name = "haiku"

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            client = anthropic.Anthropic(api_key=api_key)
        self._client = client

    def adjudicate(self, trace: dict, cache: VlmCache | None = None) -> dict:
        payload = trace_json(trace)
        key = content_key(HAIKU_MODEL, HAIKU_SYSTEM, payload)
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                return hit
        response = self._client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=500,
            system=HAIKU_SYSTEM,
            messages=[{"role": "user", "content": payload}],
        )
        text = next((b.text for b in response.content if hasattr(b, "text")), "")
        verdict: dict = {}
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                verdict = parsed
        if cache is not None:
            cache.put(key, verdict)
        return verdict


# Decision thresholds for the jev backend. They are starting points, not tuned
# values: score them with harness/calibration.py on truth before trusting them.
OUTCOME_MIN_P = 0.5  # below this the outcome is reported as "unclear"
SCORER_MIN_P = 0.6  # below this the basket stays a team stat (doubt default)
ASSIST_MIN_P = 0.6
ANOMALY_MIN_P = 0.5

TEAM_OPTION = "team_unattributed"

_OUTCOME_CRITERIA = {
    "made_fg": "A field goal attempt in this possession went in. Trust `shot_events` "
    "over raw `ball_controls` when they conflict.",
    "missed_fg_dreb": "A field goal attempt missed and the defense gained control, "
    "ending the possession. When rebound control is in doubt, assume control.",
    "missed_fg_oreb": "A field goal attempt missed and the offense kept control, so "
    "the possession continued.",
    "turnover": "The offense lost the ball to the other team with no field goal "
    "attempt ending the possession.",
    "unclear": "The trace is too sparse or contradictory to support any of the "
    "other outcomes. When the act of shooting is in doubt, assume no shot.",
}


def _last_pass_before_shot(trace: dict) -> dict | None:
    """The only pass the FIBA assist test considers. Picked in code: it is a
    lookup, not a judgment."""
    passes = trace.get("passes") or []
    if not passes:
        return None
    made = [s for s in trace.get("shot_events") or [] if s.get("made")]
    if made:
        shot_ts = made[-1]["ts_s"]
        before = [p for p in passes if p["ts_s"] <= shot_ts]
        return before[-1] if before else None
    return passes[-1]


def _offense_entities(trace: dict) -> list[int]:
    offense = trace.get("offense_team_cluster")
    seen: list[int] = []
    for control in trace.get("ball_controls") or []:
        if offense is not None and control.get("team") != offense:
            continue
        if control["entity"] not in seen:
            seen.append(control["entity"])
    return seen


def structural_anomalies(trace: dict) -> list[str]:
    """Contradictions that are computable from the trace are found in code, not
    asked of a model. (A live probe on 2026-09-18 showed why: given a made shot
    in a possession where only the defense ever controlled the ball, the model's
    self-contradiction judgment came back at 0.11.)"""
    found: list[str] = []
    offense = trace.get("offense_team_cluster")
    controls = trace.get("ball_controls") or []
    made = [s for s in trace.get("shot_events") or [] if s.get("made")]
    if made and offense is not None and controls:
        if not any(c.get("team") == offense for c in controls):
            found.append("made_shot_without_offense_control")
    if made and not controls:
        found.append("made_shot_without_any_ball_control")
    return found


def jev_questions(trace: dict) -> tuple[dict, dict[str, dict]]:
    """State and typed questions for one possession, as plain dicts.

    Every question is asked in one request (speculative fan-out): the scorer and
    assist questions state their premise, and code consumes them only when the
    outcome answer makes them applicable.
    """
    state = dict(trace)
    questions: dict[str, dict] = {
        "outcome": {
            "type": "choice",
            "instructions": "This is a symbolic trace of one basketball possession from "
            "a computer-vision pipeline; ball detection is sparse, so the trace may be "
            "incomplete. Applying FIBA statistician conventions, how did the possession "
            "end?",
            "criteria": _OUTCOME_CRITERIA,
        },
        "anomaly": {
            "type": "noul",
            "instructions": "Does this trace contradict itself, for example a made shot "
            "in `shot_events` with no offensive ball control near it, or ball controls "
            "that place the ball with both teams at the same time?",
        },
    }
    candidates = _offense_entities(trace)
    if candidates:
        criteria: dict[str, str] = {
            f"entity_{eid}": f"Entity {eid} took the made shot: it controlled the ball "
            "at or just before the made shot event."
            for eid in candidates
        }
        criteria[TEAM_OPTION] = (
            "The trace cannot establish which entity shot. Unattributable stats belong "
            "to the team, never to a guessed player."
        )
        questions["scorer"] = {
            "type": "choice",
            "instructions": "Assume this possession ended with a made field goal. Which "
            "entity scored it?",
            "criteria": criteria,
        }
    last_pass = _last_pass_before_shot(trace)
    if last_pass is not None:
        state["last_pass_before_shot"] = last_pass
        questions["assist"] = {
            "type": "noul",
            "instructions": "Assume this possession ended with a made field goal by the "
            "receiver of `last_pass_before_shot`. Under the FIBA mechanical test, is "
            "that pass an assist? A pass to a player who scores from the paint is an "
            "assist. A pass to a player outside the paint who scores without dribbling "
            "is an assist. With dribbles it is an assist only if the shooter did not "
            "have to beat their own defender. If the ball-control data cannot establish "
            "a qualifying pass, it is not an assist.",
        }
    return state, questions


def compose_jev_verdict(trace: dict, answers: dict[str, dict]) -> dict:
    """Typed answers -> the harness verdict. All policy lives here, in code."""
    outcome_answer = answers["outcome"]
    probabilities = {k: float(v) for k, v in outcome_answer["probabilities"].items()}
    top = max(probabilities, key=lambda k: probabilities[k])
    outcome = top if probabilities[top] >= OUTCOME_MIN_P else "unclear"

    scorer = assist = None
    if outcome == "made_fg":
        scorer_answer = answers.get("scorer")
        if scorer_answer is not None:
            choice = scorer_answer["choice"]
            p_choice = float(scorer_answer["probabilities"][choice])
            if choice != TEAM_OPTION and p_choice >= SCORER_MIN_P:
                scorer = int(choice.removeprefix("entity_"))
        assist_answer = answers.get("assist")
        last_pass = _last_pass_before_shot(trace)
        if (
            assist_answer is not None
            and last_pass is not None
            and scorer is not None
            and last_pass["to_entity"] == scorer
            and float(assist_answer["noul"]) >= ASSIST_MIN_P
        ):
            assist = int(last_pass["from_entity"])

    p_anomaly = float(answers["anomaly"]["noul"])
    return {
        "outcome": outcome,
        "scorer_entity": scorer,
        "assist_entity": assist,
        "confidence": probabilities[outcome],
        "anomalies": structural_anomalies(trace)
        + (["trace_self_contradiction"] if p_anomaly >= ANOMALY_MIN_P else []),
        "rationale": None,
        "outcome_probabilities": probabilities,
    }


class JevAdjudicator:
    name = "jev"

    def __init__(self, client: Any | None = None, model: str = JEV_MODEL) -> None:
        if client is None:
            from typesafe_sdk import TypeSafeClient

            if not os.environ.get("TYPESAFE_API_KEY"):
                raise RuntimeError("TYPESAFE_API_KEY not set")
            client = TypeSafeClient(model=model)
        self._client = client
        self._model = model

    def adjudicate(self, trace: dict, cache: VlmCache | None = None) -> dict:
        state, questions = jev_questions(trace)
        key = content_key(
            self._model,
            json.dumps(questions, sort_keys=True),
            json.dumps(state, sort_keys=True, separators=(",", ":")),
        )
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                return hit
        response = self._client.system_one(state, questions)
        answers = {name: _answer_dict(answer) for name, answer in response.answers.items()}
        verdict = compose_jev_verdict(trace, answers)
        if cache is not None:
            cache.put(key, verdict)
        return verdict


def _answer_dict(answer: Any) -> dict:
    return answer.model_dump() if hasattr(answer, "model_dump") else dict(answer)


BACKENDS = {"haiku": HaikuAdjudicator, "jev": JevAdjudicator}


def make_adjudicator(name: str) -> Adjudicator:
    if name not in BACKENDS:
        raise ValueError(f"unknown adjudicator backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name]()
