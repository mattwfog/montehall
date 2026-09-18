import pyarrow as pa

from montehall_cv.pipeline.run_boxscore import BOX_EVENTS_SCHEMA, BOX_SCORE_SCHEMA
from montehall_cv.runner.adapter import build_results
from montehall_cv.store.artifacts import ArtifactWriter

SHOT_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("frame_start", pa.int32()),
        pa.field("ts_start_ms", pa.int64()),
        pa.field("verdict_attempt", pa.bool_()),
        pa.field("verdict_made", pa.bool_()),
    ]
)


def _write(stage_dir, schema, rows):
    writer = ArtifactWriter(stage_dir, schema)
    writer.add_many(rows)
    writer.close()


def _job(tmp_path):
    job = tmp_path / "job1"
    _write(job / "box_score", BOX_SCORE_SCHEMA, [
        {"job_id": "job1", "team_cluster": 0, "player_key": "24", "entity_id": 7,
         "fga": 2, "fgm": 1, "fga3": 1, "fgm3": 0, "oreb": 1, "dreb": 0,
         "points": 2, "confidence": 0.8},
        {"job_id": "job1", "team_cluster": 0, "player_key": "team", "entity_id": None,
         "fga": 1, "fgm": 0, "fga3": 0, "fgm3": 0, "oreb": 0, "dreb": 0,
         "points": 0, "confidence": 0.5},
    ])
    _write(job / "box_events", BOX_EVENTS_SCHEMA, [
        {"job_id": "job1", "frame_idx": 100, "ts_ms": 4000, "team_cluster": 0,
         "player_key": "24", "event_type": "FGA", "subtype": "FGA2"},
        {"job_id": "job1", "frame_idx": 100, "ts_ms": 4000, "team_cluster": 0,
         "player_key": "24", "event_type": "FGM", "subtype": "FGM2"},
        {"job_id": "job1", "frame_idx": 300, "ts_ms": 12000, "team_cluster": 0,
         "player_key": "24", "event_type": "REB", "subtype": None},
        {"job_id": "job1", "frame_idx": 500, "ts_ms": 20000, "team_cluster": 1,
         "player_key": "team", "event_type": "TOV", "subtype": None},
    ])
    _write(job / "shot_events", SHOT_EVENTS_SCHEMA, [
        {"job_id": "job1", "frame_start": 100, "ts_start_ms": 4000,
         "verdict_attempt": True, "verdict_made": True},
    ])
    return job


def test_timelines_use_the_transformers_two_consumption_shapes(tmp_path):
    results = build_results(_job(tmp_path), fps=30.0)
    stats = results["stats"]
    # FGA/FGM ride as {frame: subtype} dicts; REB/TOV as frame lists
    assert stats["player_events"]["team_a"]["24"]["FGA"] == {100: "FGA2"}
    assert stats["player_events"]["team_a"]["24"]["FGM"] == {100: "FGM2"}
    assert stats["player_events"]["team_a"]["24"]["REB"] == [300]
    assert stats["team_events"]["team_a"]["FGA"] == {100: "FGA2"}
    assert stats["team_events"]["team_b"]["TOV"] == [500]
    # team-line rows never leak into player_events
    assert "team" not in stats["player_events"]["team_a"]
    assert "team" not in stats["player_events"]["team_b"]


def test_turnover_totals_accumulate_from_events(tmp_path):
    results = build_results(_job(tmp_path), fps=30.0)
    assert results["stats"]["team_totals"]["team_b"]["turnover"] == 1
    assert results["stats"]["team_totals"]["team_a"]["turnover"] == 0


def test_player_rows_carry_jersey_meta(tmp_path):
    results = build_results(_job(tmp_path), fps=30.0)
    # the app's transformer sets is_unknown = (jersey is None)
    assert results["stats"]["player_totals"]["team_a"]["24"]["jersey"] == "24"


def test_made_basket_carries_scorer_and_shot_class(tmp_path):
    results = build_results(_job(tmp_path), fps=30.0)
    (basket,) = results["made_basket_play_analyses"]
    assert basket["team"] == "team_a"
    assert basket["scorer"] == {"jersey": "24"}
    assert basket["shot_class"] == "FGM2"
    assert basket["goal_frame_idx"] == 100


def test_missing_box_events_stage_degrades_to_totals_only(tmp_path):
    job = _job(tmp_path)
    for part in (job / "box_events").iterdir():
        part.unlink()
    (job / "box_events").rmdir()
    results = build_results(job, fps=30.0)
    assert results["stats"]["team_events"] == {"team_a": {}, "team_b": {}}
    assert results["stats"]["player_events"] == {"team_a": {}, "team_b": {}}
    (basket,) = results["made_basket_play_analyses"]
    assert basket["scorer"] is None and basket["shot_class"] is None


def _job_unnamed(tmp_path):
    """Entity-first artifacts: two unnamed entity rows on team_b (e50 out-
    stats e60) plus a named row, an e-key FGM, and an e-key REB event."""
    job = tmp_path / "job2"
    _write(job / "box_score", BOX_SCORE_SCHEMA, [
        {"job_id": "job2", "team_cluster": 0, "player_key": "24", "entity_id": 7,
         "fga": 1, "fgm": 0, "fga3": 0, "fgm3": 0, "oreb": 0, "dreb": 0,
         "points": 0, "confidence": 0.8},
        {"job_id": "job2", "team_cluster": 1, "player_key": "e50", "entity_id": 50,
         "fga": 2, "fgm": 1, "fga3": 0, "fgm3": 0, "oreb": 1, "dreb": 0,
         "points": 2, "confidence": 0.7, "position": "guard"},
        {"job_id": "job2", "team_cluster": 1, "player_key": "e60", "entity_id": 60,
         "fga": 0, "fgm": 0, "fga3": 0, "fgm3": 0, "oreb": 0, "dreb": 1,
         "points": 0, "confidence": 0.6},
    ])
    _write(job / "box_events", BOX_EVENTS_SCHEMA, [
        {"job_id": "job2", "frame_idx": 100, "ts_ms": 4000, "team_cluster": 1,
         "player_key": "e50", "event_type": "FGA", "subtype": "FGA2"},
        {"job_id": "job2", "frame_idx": 100, "ts_ms": 4000, "team_cluster": 1,
         "player_key": "e50", "event_type": "FGM", "subtype": "FGM2"},
        {"job_id": "job2", "frame_idx": 300, "ts_ms": 12000, "team_cluster": 1,
         "player_key": "e60", "event_type": "REB", "subtype": None},
    ])
    _write(job / "shot_events", SHOT_EVENTS_SCHEMA, [
        {"job_id": "job2", "frame_start": 100, "ts_start_ms": 4000,
         "verdict_attempt": True, "verdict_made": True},
    ])
    return job


def test_unnamed_entity_rows_project_as_is_unknown_players(tmp_path):
    results = build_results(_job_unnamed(tmp_path), fps=30.0)
    players = results["stats"]["player_totals"]["team_b"]
    # no jersey key -> the app transformer sets is_unknown; track_id binds
    # the row to the entity for identity votes
    top = players["e50"]
    assert "jersey" not in top
    assert top["track_id"] == 50
    assert top["name"] == "Player A (guard)"  # stat leader + gated position
    assert players["e60"]["name"] == "Player B"
    # e-key events ride player_events under the SAME key as the totals row
    assert results["stats"]["player_events"]["team_b"]["e50"]["FGM"] == {100: "FGM2"}
    assert results["stats"]["player_events"]["team_b"]["e60"]["REB"] == [300]
    # team totals still accumulate every entity row
    assert results["stats"]["team_totals"]["team_b"]["fga"] == 2


def test_contact_sheet_suggestions_ride_unnamed_rows(tmp_path):
    import pyarrow as pa

    from montehall_cv.pipeline.contact_sheet import CONTACT_READS_SCHEMA

    job = _job_unnamed(tmp_path)
    _write(job / "entities", pa.schema([
        pa.field("job_id", pa.string()),
        pa.field("track_id", pa.int32()),
        pa.field("entity_id", pa.int32()),
        pa.field("team_cluster", pa.int8()),
    ]), [
        {"job_id": "job2", "track_id": 7, "entity_id": 50, "team_cluster": 1},
        {"job_id": "job2", "track_id": 8, "entity_id": 50, "team_cluster": 1},
    ])
    _write(job / "contact_reads", CONTACT_READS_SCHEMA, [
        {"job_id": "job2", "track_id": 7, "number": "12",
         "confidence": 0.7, "n_crops": 10},
        {"job_id": "job2", "track_id": 8, "number": "12",
         "confidence": 0.9, "n_crops": 8},
    ])
    results = build_results(job, fps=30.0)
    top = results["stats"]["player_totals"]["team_b"]["e50"]
    # highest-confidence tracklet read rolls up; still NO jersey key
    assert top["suggested_number"] == "12"
    assert abs(top["suggested_confidence"] - 0.9) < 1e-6
    assert "jersey" not in top


def test_made_basket_scorer_for_unnamed_entity_carries_track_id(tmp_path):
    results = build_results(_job_unnamed(tmp_path), fps=30.0)
    (basket,) = results["made_basket_play_analyses"]
    assert basket["scorer"] == {"track_id": 50, "label": "Player A (guard)"}
    assert basket["team"] == "team_b"
