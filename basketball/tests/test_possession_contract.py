"""Finding-2 contract: possessions store the observed team cluster as an
int; the left/right costume survives only as a legacy read path."""

from __future__ import annotations

from pathlib import Path

from montehall_cv.store.artifacts import ArtifactWriter, read_stage
from montehall_cv.store.records import Possession, possession_offense_cluster
from montehall_cv.store.schemas import POSSESSIONS_SCHEMA


def test_helper_reads_honest_int() -> None:
    assert possession_offense_cluster({"offense_team_cluster": 1}) == 1
    assert possession_offense_cluster({"offense_team_cluster": 0}) == 0


def test_helper_reads_legacy_costume() -> None:
    assert possession_offense_cluster({"offense_team": "left"}) == 0
    assert possession_offense_cluster({"offense_team": "right"}) == 1


def test_helper_prefers_int_over_legacy() -> None:
    row = {"offense_team_cluster": 1, "offense_team": "left"}
    assert possession_offense_cluster(row) == 1


def test_helper_none_on_no_evidence() -> None:
    assert possession_offense_cluster({}) is None
    assert possession_offense_cluster({"offense_team": "midcourt"}) is None
    assert possession_offense_cluster({"offense_team_cluster": None}) is None


def test_possession_model_takes_int_cluster() -> None:
    p = Possession(
        job_id="j", possession_id=0, frame_start=0, frame_end=10,
        ts_start_ms=0, ts_end_ms=3000, offense_team_cluster=1,
    )
    assert p.offense_team_cluster == 1


def test_schema_round_trip(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path / "possessions", POSSESSIONS_SCHEMA)
    writer.add(
        {
            "job_id": "j", "possession_id": 0, "frame_start": 0,
            "frame_end": 10, "ts_start_ms": 0, "ts_end_ms": 3000,
            "offense_team_cluster": 1, "outcome": None,
        }
    )
    writer.close()
    rows = read_stage(tmp_path / "possessions").to_pylist()
    assert possession_offense_cluster(rows[0]) == 1
