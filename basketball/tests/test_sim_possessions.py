"""Simulated possessions: truth comes from the simulator, the trace is degraded."""

from montehall_cv.harness import sim_possessions as sp


def test_sample_is_deterministic_and_sized():
    a, b = sp.sample(40, seed=3), sp.sample(40, seed=3)
    assert len(a) == 40 and a == b
    assert [trace["possession_id"] for trace, _ in a] == list(range(40))


def test_truth_covers_every_way_a_possession_ends():
    outcomes = {truth["outcome"] for _, truth in sp.sample(120, seed=5)}
    assert outcomes == {"made_fg", "missed_fg_dreb", "turnover"}


def test_only_makes_carry_a_scorer():
    for _, truth in sp.sample(120, seed=5):
        assert (truth["scorer_entity"] is not None) == (truth["outcome"] == "made_fg")


def test_trace_is_sparse_and_inside_its_span():
    pairs = sp.sample(120, seed=5)
    for trace, _ in pairs:
        for control in trace["ball_controls"]:
            assert trace["start_s"] <= control["ts_s"] <= trace["end_s"]
        assert trace["duration_s"] >= sp.MIN_POSSESSION_S
    ticks = sum(trace["duration_s"] * 5 for trace, _ in pairs)
    seen = sum(len(trace["ball_controls"]) for trace, _ in pairs)
    assert seen / ticks < 0.3  # the ball is mostly unobserved, as on real film


def test_the_made_flag_alone_does_not_solve_it():
    """If trusting the last shot flag were enough, the eval would measure nothing."""
    pairs = sp.sample(300, seed=7)

    def naive(trace):
        if not trace["shot_events"]:
            return "turnover"
        return "made_fg" if trace["shot_events"][-1]["made"] else "missed_fg_dreb"

    accuracy = sum(naive(trace) == truth["outcome"] for trace, truth in pairs) / len(pairs)
    assert 0.5 < accuracy < 0.8
