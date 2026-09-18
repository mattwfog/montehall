import pytest
from pydantic import ValidationError

from montehall_cv.store.records import (
    DetClass,
    Detection,
    IdentityHypothesis,
    TrackletSummary,
)


def _detection(**overrides):
    base = dict(
        job_id="j1", frame_idx=0, ts_ms=0, det_idx=0, cls=DetClass.PERSON,
        x1=0.0, y1=0.0, x2=10.0, y2=10.0, conf=0.9,
    )
    base.update(overrides)
    return Detection(**base)


def test_detection_defaults():
    d = _detection()
    assert d.track_id is None
    assert d.is_detected is True
    assert d.court_x is None


def test_detection_conf_bounds():
    with pytest.raises(ValidationError):
        _detection(conf=1.5)
    with pytest.raises(ValidationError):
        _detection(conf=-0.1)


def test_detection_frozen():
    d = _detection()
    with pytest.raises(ValidationError):
        d.conf = 0.5


def test_identity_hypothesis_stage_vocabulary():
    IdentityHypothesis(job_id="j1", track_id=1, candidate="23", prob=0.8, bound_at_stage="llm")
    with pytest.raises(ValidationError):
        IdentityHypothesis(
            job_id="j1", track_id=1, candidate="23", prob=0.8, bound_at_stage="vibes"
        )


def test_tracklet_requires_detections():
    with pytest.raises(ValidationError):
        TrackletSummary(
            job_id="j1", track_id=1, cls=DetClass.PERSON,
            frame_start=0, frame_end=10, ts_start_ms=0, ts_end_ms=333,
            n_detections=0, mean_conf=0.5,
        )
