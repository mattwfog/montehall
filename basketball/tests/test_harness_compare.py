"""compare(): same traces through each backend, scored against outside truth."""

from montehall_cv.harness import compare as cmp


class _Fixed:
    def __init__(self, name, verdicts):
        self.name, self._verdicts = name, verdicts

    def adjudicate(self, trace, cache=None):
        return self._verdicts[trace["possession_id"]]


def test_compare_scores_each_backend_and_counts_abstentions(tmp_path, monkeypatch):
    traces = [{"possession_id": i} for i in (1, 2, 3, 4)]  # 4 has no truth: excluded
    truth = {1: "made_fg", 2: "turnover", 3: "made_fg"}
    backends = {
        "sure": _Fixed("sure", {
            1: {"outcome": "made_fg", "confidence": 0.9},
            2: {"outcome": "made_fg", "confidence": 0.9},  # confidently wrong
            3: {"outcome": "made_fg", "confidence": 0.9},
        }),
        "careful": _Fixed("careful", {
            1: {"outcome": "made_fg", "confidence": 0.9},
            2: {"outcome": "unclear", "confidence": 0.3},  # abstains instead
            3: {"outcome": "made_fg", "confidence": 0.8},
        }),
    }
    monkeypatch.setattr(cmp, "build_traces", lambda job_dir: traces)
    monkeypatch.setattr(cmp, "make_adjudicator", lambda name: backends[name])

    report = cmp.compare(tmp_path, truth, ["sure", "careful"])

    assert report["n_possessions"] == 3
    sure, careful = report["backends"]["sure"], report["backends"]["careful"]
    assert (sure["answered"], sure["abstained"]) == (3, 0)
    assert round(sure["accuracy"], 3) == 0.667
    assert (careful["answered"], careful["abstained"]) == (2, 1)
    assert careful["accuracy"] == 1.0
    assert careful["ece"] < sure["ece"]
