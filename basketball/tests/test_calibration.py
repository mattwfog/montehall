import pytest

from montehall_cv.harness import calibration as cal


def test_perfectly_calibrated_bins_have_zero_ece():
    confidence = [0.75] * 4 + [0.25] * 4
    correct = [True, True, True, False, True, False, False, False]
    assert cal.expected_calibration_error(confidence, correct) == pytest.approx(0.0)


def test_overconfidence_shows_up_as_ece():
    confidence = [0.95] * 10
    correct = [True] * 5 + [False] * 5
    assert cal.expected_calibration_error(confidence, correct) == pytest.approx(0.45)
    assert cal.brier_score(confidence, correct) == pytest.approx((0.05**2 * 5 + 0.95**2 * 5) / 10)


def test_reliability_bins_skip_empty_and_clamp_one():
    bins = cal.reliability_bins([1.0, 1.0, 0.05], [True, False, False])
    assert [(b["lo"], b["n"]) for b in bins] == [(0.0, 1), (0.9, 2)]
    assert bins[1]["accuracy"] == pytest.approx(0.5)


def test_precision_rises_as_coverage_falls_for_an_informative_confidence():
    confidence = [0.9, 0.9, 0.6, 0.6, 0.3, 0.3]
    correct = [True, True, True, False, False, False]
    curve = cal.precision_at_coverage(confidence, correct)
    assert [round(p["coverage"], 3) for p in curve] == [1.0, 0.667, 0.333]
    assert [round(p["precision"], 3) for p in curve] == [0.5, 0.75, 1.0]


def test_summarize_reports_every_metric():
    summary = cal.summarize([0.8, 0.6, 0.4], [True, True, False])
    assert summary["n"] == 3
    assert {"accuracy", "mean_confidence", "ece", "brier", "reliability", "precision_at_coverage"} <= set(summary)


@pytest.mark.parametrize("confidence,correct", [([], []), ([0.5], [True, False]), ([1.2], [True])])
def test_bad_input_is_refused(confidence, correct):
    with pytest.raises(ValueError):
        cal.summarize(confidence, correct)
