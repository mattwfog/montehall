"""Adjudicator backends: question construction, verdict composition, caching.
No network: both SDK clients are stubbed."""

import json
from types import SimpleNamespace

import pytest

from montehall_cv.harness import adjudicators as adj
from montehall_cv.store.vlm_cache import VlmCache


def _trace(**overrides):
    trace = {
        "possession_id": 7,
        "start_s": 100.0,
        "end_s": 112.0,
        "duration_s": 12.0,
        "offense_team_cluster": 0,
        "ball_controls": [
            {"ts_s": 101.0, "entity": 3, "team": 0, "court_x": 40.0, "court_y": 20.0},
            {"ts_s": 104.0, "entity": 9, "team": 1, "court_x": 42.0, "court_y": 22.0},
            {"ts_s": 108.0, "entity": 5, "team": 0, "court_x": 88.0, "court_y": 25.0},
        ],
        "shot_events": [{"ts_s": 110.0, "attempt": True, "made": True, "confidence": 0.9}],
        "passes": [
            {"ts_s": 108.0, "from_entity": 3, "to_entity": 5, "to_court_x": 88.0, "to_court_y": 25.0},
            {"ts_s": 111.5, "from_entity": 5, "to_entity": 3, "to_court_x": 60.0, "to_court_y": 25.0},
        ],
        "entities_involved": [{"entity": 3, "jersey": "24"}, {"entity": 5, "jersey": None}],
    }
    trace.update(overrides)
    return trace


def _answers(outcome_p, scorer=None, assist=None, anomaly=0.05):
    top = max(outcome_p, key=lambda k: outcome_p[k])
    answers = {
        "outcome": {"type": "choice", "choice": top, "confidence": 0.9, "probabilities": outcome_p},
        "anomaly": {"type": "noul", "noul": anomaly},
    }
    if scorer is not None:
        top_s = max(scorer, key=lambda k: scorer[k])
        answers["scorer"] = {"type": "choice", "choice": top_s, "confidence": 0.9, "probabilities": scorer}
    if assist is not None:
        answers["assist"] = {"type": "noul", "noul": assist}
    return answers


MADE = {"made_fg": 0.9, "missed_fg_dreb": 0.04, "missed_fg_oreb": 0.02, "turnover": 0.02, "unclear": 0.02}


def test_scorer_options_are_offense_entities_plus_team():
    _, questions = adj.jev_questions(_trace())
    assert list(questions["scorer"]["criteria"]) == ["entity_3", "entity_5", adj.TEAM_OPTION]


def test_assist_question_uses_last_pass_before_the_made_shot():
    state, questions = adj.jev_questions(_trace())
    # the 111.5s pass is after the 110.0s make and must not be the candidate
    assert state["last_pass_before_shot"]["ts_s"] == 108.0
    assert "assist" in questions


def test_no_passes_means_no_assist_question():
    state, questions = adj.jev_questions(_trace(passes=[]))
    assert "assist" not in questions
    assert "last_pass_before_shot" not in state


def test_made_fg_with_scorer_and_assist():
    verdict = adj.compose_jev_verdict(
        _trace(), _answers(MADE, scorer={"entity_5": 0.8, "entity_3": 0.1, adj.TEAM_OPTION: 0.1}, assist=0.85)
    )
    assert verdict["outcome"] == "made_fg"
    assert verdict["scorer_entity"] == 5
    assert verdict["assist_entity"] == 3
    assert verdict["confidence"] == pytest.approx(0.9)
    assert verdict["outcome_probabilities"] == MADE


def test_flat_outcome_distribution_abstains():
    flat = {"made_fg": 0.3, "missed_fg_dreb": 0.3, "missed_fg_oreb": 0.2, "turnover": 0.1, "unclear": 0.1}
    verdict = adj.compose_jev_verdict(_trace(), _answers(flat, scorer={"entity_5": 0.9, adj.TEAM_OPTION: 0.1}))
    assert verdict["outcome"] == "unclear"
    assert verdict["scorer_entity"] is None
    assert verdict["confidence"] == pytest.approx(0.1)


def test_uncertain_scorer_stays_a_team_stat():
    verdict = adj.compose_jev_verdict(
        _trace(), _answers(MADE, scorer={"entity_5": 0.45, "entity_3": 0.4, adj.TEAM_OPTION: 0.15}, assist=0.9)
    )
    assert verdict["scorer_entity"] is None
    assert verdict["assist_entity"] is None  # no assist without an attributed scorer


def test_assist_requires_the_pass_receiver_to_be_the_scorer():
    verdict = adj.compose_jev_verdict(
        _trace(), _answers(MADE, scorer={"entity_3": 0.9, "entity_5": 0.05, adj.TEAM_OPTION: 0.05}, assist=0.95)
    )
    assert verdict["scorer_entity"] == 3
    assert verdict["assist_entity"] is None


def test_scorer_ignored_when_outcome_is_not_a_make():
    missed = {"made_fg": 0.05, "missed_fg_dreb": 0.85, "missed_fg_oreb": 0.04, "turnover": 0.03, "unclear": 0.03}
    verdict = adj.compose_jev_verdict(_trace(), _answers(missed, scorer={"entity_5": 0.99, adj.TEAM_OPTION: 0.01}))
    assert verdict["outcome"] == "missed_fg_dreb"
    assert verdict["scorer_entity"] is None


def test_anomaly_threshold():
    assert adj.compose_jev_verdict(_trace(), _answers(MADE, anomaly=0.7))["anomalies"] == ["trace_self_contradiction"]
    assert adj.compose_jev_verdict(_trace(), _answers(MADE, anomaly=0.2))["anomalies"] == []


class _JevStub:
    def __init__(self, answers):
        self.answers, self.calls = answers, 0

    def system_one(self, state, questions):
        self.calls += 1
        self.last = (state, questions)
        return SimpleNamespace(answers=self.answers)


def test_jev_adjudicator_sends_one_request_and_caches(tmp_path):
    stub = _JevStub(_answers(MADE, scorer={"entity_5": 0.8, adj.TEAM_OPTION: 0.2}, assist=0.9))
    cache = VlmCache(tmp_path / "cache.jsonl")
    adjudicator = adj.JevAdjudicator(client=stub)
    first = adjudicator.adjudicate(_trace(), cache=cache)
    second = adjudicator.adjudicate(_trace(), cache=cache)
    assert stub.calls == 1
    assert first == second
    assert set(stub.last[1]) == {"outcome", "anomaly", "scorer", "assist"}


class _HaikuStub:
    def __init__(self, text):
        self.text = text
        self.messages = self

    def create(self, **_):
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)])


def test_haiku_adjudicator_parses_json_verdict():
    body = {"outcome": "turnover", "scorer_entity": None, "assist_entity": None, "confidence": 0.7}
    verdict = adj.HaikuAdjudicator(client=_HaikuStub("ok: " + json.dumps(body))).adjudicate(_trace())
    assert verdict == body


def test_haiku_adjudicator_returns_empty_on_unparseable_text():
    assert adj.HaikuAdjudicator(client=_HaikuStub("no json here")).adjudicate(_trace()) == {}


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError):
        adj.make_adjudicator("gpt")


def test_jev_backend_round_trips_through_the_real_sdk_client():
    """The real TypeSafeClient, with only the HTTP transport mocked: proves the
    request shape the SDK accepts and the response shape the adjudicator reads."""
    sdk = pytest.importorskip("typesafe_sdk")
    httpx2 = pytest.importorskip("httpx2")
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        answers = {}
        for name, question in sent["questions"].items():
            if question["type"] == "choice":
                options = list(question["criteria"])
                rest = 0.1 / max(len(options) - 1, 1)
                probabilities = {o: (0.9 if i == 0 else rest) for i, o in enumerate(options)}
                answers[name] = {
                    "type": "choice",
                    "choice": options[0],
                    "confidence": 0.8,
                    "probabilities": probabilities,
                }
            else:
                answers[name] = {"type": "noul", "noul": 0.9}
        body = {"model": "jev-1.13.0", "usage": {"input_tokens": 1, "output_tokens": 0}, "answers": answers}
        return httpx2.Response(200, json=body)

    client = sdk.TypeSafeClient(api_key="test", transport=httpx2.MockTransport(handler))
    verdict = adj.JevAdjudicator(client=client).adjudicate(_trace())
    assert set(sent["questions"]) == {"outcome", "anomaly", "scorer", "assist"}
    assert sent["state"]["last_pass_before_shot"]["ts_s"] == 108.0
    assert verdict["outcome"] == "made_fg"
    assert verdict["scorer_entity"] == 3  # first offense entity in the stubbed distribution
    assert verdict["assist_entity"] is None  # the last pass went to entity 5, not the scorer
