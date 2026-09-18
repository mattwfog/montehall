"""Entity-first box-score keying (identity design-of-record, 2026-07-10):
attributed-but-unnamed entities take their own "e<id>" stat rows instead of
folding into the team line; roster-illegal jerseys demote to unnamed rows;
position descriptors are confidence-gated."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from montehall_cv.pipeline import run_boxscore
from montehall_cv.pipeline.run_boxscore import _position_map, is_entity_key
from montehall_cv.store.artifacts import ArtifactWriter, read_stage
from montehall_cv.store.schemas import (
    BALL_CONTROLS_SCHEMA,
    IDENTITY_SCHEMA,
    POSITIONS_SCHEMA,
)
from montehall_cv.zones import THREE_PT_RADIUS_FT, rim_distance_ft

ENTITIES_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("track_id", pa.int32()),
        pa.field("entity_id", pa.int32()),
        pa.field("team_cluster", pa.int8()),
    ]
)

SHOT_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("job_id", pa.string()),
        pa.field("event_id", pa.int32()),
        pa.field("frame_start", pa.int32()),
        pa.field("frame_end", pa.int32()),
        pa.field("ts_start_ms", pa.int64()),
        pa.field("ts_end_ms", pa.int64()),
        pa.field("court_end", pa.string(), nullable=True),
        pa.field("n_ball_obs", pa.int32()),
        pa.field("verdict_attempt", pa.bool_(), nullable=True),
        pa.field("verdict_made", pa.bool_(), nullable=True),
        pa.field("confidence", pa.float32(), nullable=True),
        pa.field("rationale", pa.string(), nullable=True),
    ]
)


def _write(stage_dir: Path, schema: pa.Schema, rows: list[dict]) -> None:
    writer = ArtifactWriter(stage_dir, schema)
    writer.add_many(rows)
    writer.close()


def _control(ts: int, entity: int, team: int) -> dict:
    return {
        "job_id": "j", "frame_idx": ts // 33, "ts_ms": ts, "entity_id": entity,
        "team_cluster": team, "court_x": 12.0, "court_y": 25.0,
        "ball_x": 12.5, "ball_y": 25.0, "dist_ft": 0.5,
    }


def _shot(event_id: int, ts: int, made: bool) -> dict:
    return {
        "job_id": "j", "event_id": event_id, "frame_start": ts // 33,
        "frame_end": ts // 33 + 10, "ts_start_ms": ts, "ts_end_ms": ts + 400,
        "court_end": "left", "n_ball_obs": 4, "verdict_attempt": True,
        "verdict_made": made, "confidence": 0.9, "rationale": "test",
    }


def _entities(job: Path, rows: list[tuple[int, int, int]]) -> None:
    _write(job / "entities", ENTITIES_SCHEMA, [
        {"job_id": "j", "track_id": t, "entity_id": e, "team_cluster": c}
        for t, e, c in rows
    ])


def _identity(job: Path, reads: list[tuple[int, str, float]]) -> None:
    _write(job / "identity", IDENTITY_SCHEMA, [
        {"job_id": "j", "track_id": t, "candidate": cand, "prob": p,
         "bound_at_stage": "ocr_vote", "bound_at_frame": None}
        for t, cand, p in reads
    ])


def test_unnamed_entities_take_their_own_rows_not_the_team_line(tmp_path: Path):
    job = tmp_path / "j"
    # entity 400 (team 0) has NO jersey read and shoots a miss;
    # entity 500 (team 1) has NO jersey read and grabs the board
    _entities(job, [(1, 400, 0), (2, 500, 1)])
    _identity(job, [])
    _write(job / "ball_controls", BALL_CONTROLS_SCHEMA, [
        _control(1_500, 400, 0),
        _control(2_800, 500, 1),
    ])
    _write(job / "shot_events", SHOT_EVENTS_SCHEMA, [_shot(0, 2_000, made=False)])

    summary = run_boxscore.run(tmp_path, "j")
    rows = {
        (r["team_cluster"], r["player_key"]): r
        for r in read_stage(job / "box_score").to_pylist()
    }
    shooter = rows[(0, "e400")]
    assert shooter["fga"] == 1 and shooter["entity_id"] == 400
    rebounder = rows[(1, "e500")]
    assert rebounder["dreb"] == 1 and rebounder["entity_id"] == 500
    assert (0, "team") not in rows and (1, "team") not in rows
    assert summary["entity_rows"] == 2 and summary["named_rows"] == 0
    events = read_stage(job / "box_events").to_pylist()
    assert {e["player_key"] for e in events} == {"e400", "e500"}


def test_assist_credits_an_unnamed_passer(tmp_path: Path):
    job = tmp_path / "j"
    # named shooter (jersey 24), unnamed passer (entity 100, no read)
    _entities(job, [(1, 100, 0), (2, 200, 0)])
    _identity(job, [(2, "24", 0.9)])
    _write(job / "ball_controls", BALL_CONTROLS_SCHEMA, [
        _control(500, 100, 0),
        _control(1_200, 200, 0),
    ])
    _write(job / "shot_events", SHOT_EVENTS_SCHEMA, [_shot(0, 2_000, made=True)])

    summary = run_boxscore.run(tmp_path, "j")
    assert summary["assists"] == 1
    rows = {
        (r["team_cluster"], r["player_key"]): r
        for r in read_stage(job / "box_score").to_pylist()
    }
    assert rows[(0, "e100")]["ast"] == 1
    assert rows[(0, "24")]["fgm"] == 1


def test_roster_illegal_jersey_demotes_to_unnamed_row(tmp_path: Path):
    job = tmp_path / "j"
    # reads "10" and "24" bind cluster 0 to the roster; entity 600's read
    # "99" is not a roster number -> it must NOT name the line, and the
    # stats must NOT fold into the team line either
    _entities(job, [(1, 100, 0), (2, 200, 0), (3, 600, 0)])
    _identity(job, [(1, "10", 0.9), (2, "24", 0.9), (3, "99", 0.9)])
    _write(job / "ball_controls", BALL_CONTROLS_SCHEMA, [_control(1_500, 600, 0)])
    _write(job / "shot_events", SHOT_EVENTS_SCHEMA, [_shot(0, 2_000, made=False)])
    roster = job.parent / "roster.json"
    roster.write_text(
        '{"j": {"players": [{"number": "10"}, {"number": "24"}, {"number": "30"}]}}'
    )

    summary = run_boxscore.run(tmp_path, "j", roster_map=roster)
    assert summary["roster_bound_cluster"] == 0
    assert summary["roster_filtered_candidates"] == 1
    rows = {
        (r["team_cluster"], r["player_key"]): r
        for r in read_stage(job / "box_score").to_pylist()
    }
    assert (0, "99") not in rows
    assert rows[(0, "e600")]["fga"] == 1  # demoted, not deleted


def test_position_map_is_confidence_gated(tmp_path: Path):
    job = tmp_path / "j"
    perimeter_xy, paint_xy, mid_xy = (30.0, 25.0), (7.0, 25.0), (16.0, 25.0)
    # preconditions pin the geometric intent against constant drift
    assert min(rim_distance_ft(*perimeter_xy, "left"),
               rim_distance_ft(*perimeter_xy, "right")) >= THREE_PT_RADIUS_FT
    assert min(rim_distance_ft(*paint_xy, "left"),
               rim_distance_ft(*paint_xy, "right")) <= 10.0

    def rows(entity: int, xy: tuple[float, float], n: int) -> list[dict]:
        return [
            {"job_id": "j", "frame_idx": i, "ts_ms": i * 33, "cls": 0,
             "entity_id": entity, "team_cluster": 0,
             "court_x": xy[0], "court_y": xy[1], "court_conf": 0.9}
            for i in range(n)
        ]

    _write(job / "positions", POSITIONS_SCHEMA,
           rows(1, perimeter_xy, 200) + rows(2, paint_xy, 200)
           + rows(3, perimeter_xy, 50)   # too few samples -> abstain
           + rows(4, mid_xy, 200))       # neither share dominates -> abstain

    assert _position_map(job) == {1: "guard", 2: "big"}


def test_box_score_row_carries_position(tmp_path: Path):
    job = tmp_path / "j"
    _entities(job, [(1, 400, 0)])
    _identity(job, [])
    _write(job / "ball_controls", BALL_CONTROLS_SCHEMA, [_control(1_500, 400, 0)])
    _write(job / "shot_events", SHOT_EVENTS_SCHEMA, [_shot(0, 2_000, made=False)])
    _write(job / "positions", POSITIONS_SCHEMA, [
        {"job_id": "j", "frame_idx": i, "ts_ms": i * 33, "cls": 0,
         "entity_id": 400, "team_cluster": 0,
         "court_x": 30.0, "court_y": 25.0, "court_conf": 0.9}
        for i in range(200)
    ])

    summary = run_boxscore.run(tmp_path, "j")
    assert summary["positions_labeled"] == 1
    (row,) = read_stage(job / "box_score").to_pylist()
    assert row["player_key"] == "e400" and row["position"] == "guard"


@pytest.mark.parametrize(
    ("key", "expected"),
    [("e400", True), ("24", False), ("team", False), ("5", False)],
)
def test_is_entity_key(key: str, expected: bool):
    assert is_entity_key(key) is expected


class TestJerseyNaming:
    def _identity(self, tmp_path, rows):
        from montehall_cv.store.schemas import IDENTITY_SCHEMA
        writer = ArtifactWriter(tmp_path / "identity", IDENTITY_SCHEMA)
        for tid, cand, prob in rows:
            writer.add({"job_id": "j", "track_id": tid, "candidate": cand,
                        "prob": prob, "n_reads": 3, "bound_at_stage": "ocr_vote"})
        writer.close()

    def test_same_team_fragments_share_the_number(self, tmp_path: Path) -> None:
        # One player fragmented across entities (UConn #24, 2026-07-12):
        # duplicates are the same person, both keep the name.
        from montehall_cv.pipeline.run_boxscore import _jersey_map
        self._identity(tmp_path, [(1, "24", 0.9), (2, "24", 0.8)])
        entity_of = {
            1: {"entity_id": 10, "team_cluster": 0},
            2: {"entity_id": 20, "team_cluster": 0},
        }
        jersey_of, _, n_conflicted = _jersey_map(tmp_path, entity_of)
        assert jersey_of == {10: "24", 20: "24"}
        assert n_conflicted == 0

    def test_internally_conflicted_entity_gets_no_name(self, tmp_path: Path) -> None:
        # A merged blob whose tracklets read two different numbers is a
        # bad merge — it must stay anonymous.
        from montehall_cv.pipeline.run_boxscore import _jersey_map
        self._identity(tmp_path, [(1, "24", 0.9), (2, "12", 0.9)])
        entity_of = {
            1: {"entity_id": 10, "team_cluster": 0},
            2: {"entity_id": 10, "team_cluster": 0},
        }
        jersey_of, _, n_conflicted = _jersey_map(tmp_path, entity_of)
        assert jersey_of == {}
        assert n_conflicted == 1


class TestConcurrencyVeto:
    def test_number_live_on_another_track_vetoes(self, tmp_path: Path) -> None:
        from montehall_cv.pipeline.event_identity import TrackIndex
        from montehall_cv.store.schemas import IDENTITY_SCHEMA
        loc = pa.schema([
            pa.field("job_id", pa.string()), pa.field("frame_idx", pa.int32()),
            pa.field("ts_ms", pa.int64()), pa.field("det_idx", pa.int32()),
            pa.field("cls", pa.int8()), pa.field("x1", pa.float32()),
            pa.field("y1", pa.float32()), pa.field("x2", pa.float32()),
            pa.field("y2", pa.float32()), pa.field("conf", pa.float32()),
            pa.field("track_id", pa.int32()), pa.field("is_detected", pa.bool_()),
            pa.field("court_x", pa.float32(), nullable=True),
            pa.field("court_y", pa.float32(), nullable=True),
            pa.field("court_conf", pa.float32(), nullable=True),
        ])
        w = ArtifactWriter(tmp_path / "localized", loc)
        for tid, ts in [(1, 50_000), (1, 60_000), (2, 50_000), (2, 60_000)]:
            w.add({"job_id": "j", "frame_idx": int(ts // 33), "ts_ms": ts,
                   "det_idx": 0, "cls": 0, "x1": 0.0, "y1": 0.0, "x2": 10.0,
                   "y2": 20.0, "conf": 0.9, "track_id": tid,
                   "is_detected": True, "court_x": None, "court_y": None,
                   "court_conf": None})
        w.close()
        w = ArtifactWriter(tmp_path / "identity", IDENTITY_SCHEMA)
        w.add({"job_id": "j", "track_id": 1, "candidate": "1", "prob": 0.9,
               "n_reads": 3, "bound_at_stage": "ocr_vote"})
        w.close()
        idx = TrackIndex(tmp_path, 0.5)
        # shooter's own tracks = [2]; the '1'-reading track 1 is live at ts
        assert idx.concurrent_reader_elsewhere("1", [2], 55_000) is True
        # but not when the reading track IS the shooter's
        assert idx.concurrent_reader_elsewhere("1", [1, 2], 55_000) is False
        # and not when it is off-court at ts
        assert idx.concurrent_reader_elsewhere("1", [2], 90_000) is False


class TestActiveTrackNames:
    def _stage(self, tmp_path, loc_rows, id_rows):
        from montehall_cv.store.schemas import IDENTITY_SCHEMA
        loc = pa.schema([
            pa.field("job_id", pa.string()), pa.field("frame_idx", pa.int32()),
            pa.field("ts_ms", pa.int64()), pa.field("det_idx", pa.int32()),
            pa.field("cls", pa.int8()), pa.field("x1", pa.float32()),
            pa.field("y1", pa.float32()), pa.field("x2", pa.float32()),
            pa.field("y2", pa.float32()), pa.field("conf", pa.float32()),
            pa.field("track_id", pa.int32()), pa.field("is_detected", pa.bool_()),
            pa.field("court_x", pa.float32(), nullable=True),
            pa.field("court_y", pa.float32(), nullable=True),
            pa.field("court_conf", pa.float32(), nullable=True),
        ])
        w = ArtifactWriter(tmp_path / "localized", loc)
        for tid, ts in loc_rows:
            w.add({"job_id": "j", "frame_idx": int(ts // 33), "ts_ms": ts,
                   "det_idx": 0, "cls": 0, "x1": 0.0, "y1": 0.0, "x2": 10.0,
                   "y2": 20.0, "conf": 0.9, "track_id": tid,
                   "is_detected": True, "court_x": None, "court_y": None,
                   "court_conf": None})
        w.close()
        w = ArtifactWriter(tmp_path / "identity", IDENTITY_SCHEMA)
        for tid, cand, prob in id_rows:
            w.add({"job_id": "j", "track_id": tid, "candidate": cand,
                   "prob": prob, "n_reads": 3, "bound_at_stage": "ocr_vote"})
        w.close()

    def test_active_tracklet_posterior_names_the_event(self, tmp_path) -> None:
        from montehall_cv.pipeline.event_identity import active_track_names
        # entity 5 = tracks 1 (early, reads '24') + 2 (late, no reads);
        # the event overlaps track 1 only -> named '24'
        self._stage(tmp_path,
                    loc_rows=[(1, 1000), (1, 5000), (2, 60000), (2, 65000)],
                    id_rows=[(1, "24", 0.9)])
        out = active_track_names(tmp_path, [(0, 5, 4000)], {5: [1, 2]}, 0.5)
        assert out == {0: "24"}

    def test_inactive_tracklets_name_never_smears(self, tmp_path) -> None:
        from montehall_cv.pipeline.event_identity import active_track_names
        # the '1'-read track is NOT active at the event -> no name
        self._stage(tmp_path,
                    loc_rows=[(1, 1000), (1, 5000), (2, 60000), (2, 65000)],
                    id_rows=[(1, "1", 0.9)])
        out = active_track_names(tmp_path, [(0, 5, 62000)], {5: [1, 2]}, 0.5)
        assert out == {}

    def test_conflicting_active_reads_abstain(self, tmp_path) -> None:
        from montehall_cv.pipeline.event_identity import active_track_names
        self._stage(tmp_path,
                    loc_rows=[(1, 1000), (1, 5000), (2, 4000), (2, 6000)],
                    id_rows=[(1, "24", 0.9), (2, "12", 0.9)])
        out = active_track_names(tmp_path, [(0, 5, 4500)], {5: [1, 2]}, 0.5)
        assert out == {}
