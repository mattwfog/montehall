"""Brain token stream: channel coverage, ordering, resume, refusal."""

from __future__ import annotations

import json

import pytest

from montehall_cv.brain.name_calls import match_segment, roster_surnames
from montehall_cv.brain.tokenize_game import tokenize_game
from montehall_cv.brain.tokens import (
    CH_BALL,
    CH_CLOCK,
    CH_CUT,
    CH_JERSEY_READ,
    CH_PBP_ANCHOR,
    CH_PLAYER,
    CH_SCORE_DELTA,
    TRAIN_ONLY_CHANNELS,
)
from montehall_cv.eval.pbp_align import ALIGNMENT_SCHEMA, CLOCK_READS_SCHEMA
from montehall_cv.pipeline.run_scoreboard import SCORE_EVENTS_SCHEMA
from montehall_cv.store.artifacts import ArtifactWriter, read_stage
from montehall_cv.store.schemas import (
    IDENTITY_SCHEMA,
    POSITIONS_SCHEMA,
    TRACKLETS_SCHEMA,
)


def _write(stage_dir, schema, rows):
    writer = ArtifactWriter(stage_dir, schema)
    writer.add_many(rows)
    writer.close()


def _align_only_game(tmp_path):
    game = tmp_path / "eid_401000001"
    reads = [
        {"video_t": float(t), "text": "19:00", "conf": 0.99,
         "clock_s": 1140.0 - t, "period": 1}
        for t in range(0, 20, 2)
    ]
    # a 30s confident-read gap -> one cut token
    reads.append({"video_t": 50.0, "text": "18:00", "conf": 0.99,
                  "clock_s": 1080.0, "period": 1})
    _write(game / "clock_reads", CLOCK_READS_SCHEMA, reads)
    _write(game / "pbp_alignment", ALIGNMENT_SCHEMA, [
        {"play_id": "p1", "sequence": 0, "period": 1, "clock_s": 1130.0,
         "video_t": 10.0, "type_text": "JumpShot", "text": "X made jumper",
         "shooting_play": True, "scoring_play": True, "score_value": 2,
         "athlete_ids": ["101"], "shooter_jersey": "24",
         "coord_x": 25.0, "coord_y": 10.0, "align_gap_s": 0.4},
    ])
    return game


def _full_job(tmp_path):
    job = tmp_path / "job_a"
    _write(job / "positions", POSITIONS_SCHEMA, [
        # entity 7: two rows inside one 200ms bucket -> one player token
        {"job_id": "job_a", "frame_idx": 1, "ts_ms": 1000, "cls": 0,
         "entity_id": 7, "team_cluster": 0, "court_x": 10.0, "court_y": 20.0,
         "court_conf": 0.9},
        {"job_id": "job_a", "frame_idx": 2, "ts_ms": 1100, "cls": 0,
         "entity_id": 7, "team_cluster": 0, "court_x": 10.5, "court_y": 20.0,
         "court_conf": 0.9},
        {"job_id": "job_a", "frame_idx": 7, "ts_ms": 1400, "cls": 0,
         "entity_id": 7, "team_cluster": 0, "court_x": 11.0, "court_y": 21.0,
         "court_conf": 0.9},
        # the ball
        {"job_id": "job_a", "frame_idx": 1, "ts_ms": 1000, "cls": 1,
         "entity_id": None, "team_cluster": None, "court_x": 12.0,
         "court_y": 22.0, "court_conf": 0.8},
    ])
    _write(job / "tracklets", TRACKLETS_SCHEMA, [
        {"job_id": "job_a", "track_id": 3, "cls": 0, "frame_start": 0,
         "frame_end": 100, "ts_start_ms": 0, "ts_end_ms": 4000,
         "n_detections": 100, "mean_conf": 0.9},
    ])
    _write(job / "identity", IDENTITY_SCHEMA, [
        {"job_id": "job_a", "track_id": 3, "candidate": "24", "prob": 0.97,
         "bound_at_stage": "ocr_vote", "bound_at_frame": 50},
    ])
    _write(job / "score_events", SCORE_EVENTS_SCHEMA, [
        {"job_id": "job_a", "event_id": 0, "side": "home", "ts_ms": 3000,
         "before_val": 10, "after_val": 12, "delta": 2, "read_conf": 0.95},
    ])
    return job


class TestTokenizeGame:
    def test_align_only_channels(self, tmp_path) -> None:
        game = _align_only_game(tmp_path)
        meta = tokenize_game(game)
        assert meta["channels"][CH_CLOCK] == 11
        assert meta["channels"][CH_CUT] == 1
        assert meta["channels"][CH_PBP_ANCHOR] == 1
        table = read_stage(game / "tokens")
        rows = table.to_pylist()
        assert [r["token_idx"] for r in rows] == list(range(len(rows)))
        assert all(a["t_ms"] <= b["t_ms"] for a, b in zip(rows, rows[1:]))
        anchor = next(r for r in rows if r["channel"] == CH_PBP_ANCHOR)
        payload = json.loads(anchor["payload_json"])
        assert payload["shooter_jersey"] == "24"
        assert anchor["text"] == "JumpShot"

    def test_full_job_channels_and_downsampling(self, tmp_path) -> None:
        job = _full_job(tmp_path)
        meta = tokenize_game(job)
        # 1000+1100 collapse into one bucket; 1400 is its own -> 2 tokens
        assert meta["channels"][CH_PLAYER] == 2
        assert meta["channels"][CH_BALL] == 1
        assert meta["channels"][CH_JERSEY_READ] == 1
        assert meta["channels"][CH_SCORE_DELTA] == 1
        rows = read_stage(job / "tokens").to_pylist()
        jersey = next(r for r in rows if r["channel"] == CH_JERSEY_READ)
        # frame 50 of a 0..100/0..4000ms span -> 2000ms
        assert jersey["t_ms"] == 2000
        assert jersey["text"] == "24"

    def test_resume_skips_completed(self, tmp_path) -> None:
        game = _align_only_game(tmp_path)
        first = tokenize_game(game)
        again = tokenize_game(game)
        assert again == first

    def test_refuses_empty(self, tmp_path) -> None:
        barren = tmp_path / "eid_nothing"
        barren.mkdir()
        with pytest.raises(RuntimeError, match="no tokens"):
            tokenize_game(barren)

    def test_train_only_registry(self) -> None:
        assert CH_PBP_ANCHOR in TRAIN_ONLY_CHANNELS
        assert CH_CLOCK not in TRAIN_ONLY_CHANNELS


class TestNameCallMatching:
    SUMMARY = {
        "boxscore": {"players": [
            {"statistics": [{"athletes": [
                {"athlete": {"id": "101", "displayName": "Jalen Carter"},
                 "jersey": "24"},
                {"athlete": {"id": "102", "displayName": "Mike Carter"},
                 "jersey": "3"},
                {"athlete": {"id": "103", "displayName": "Sam Okafor"},
                 "jersey": "11"},
                {"athlete": {"id": "104", "displayName": "Bo Li"},
                 "jersey": "5"},
            ]}]},
        ]},
    }

    def test_exact_surname_hit(self) -> None:
        surnames = roster_surnames(self.SUMMARY)
        hits = match_segment("Okafor drills the three!", surnames)
        assert [s for s, _ in hits] == ["okafor"]
        assert hits[0][1][0]["athlete_id"] == "103"

    def test_shared_surname_is_ambiguous_not_a_guess(self) -> None:
        surnames = roster_surnames(self.SUMMARY)
        hits = match_segment("Carter with the rebound", surnames)
        assert len(hits) == 1
        assert {e["athlete_id"] for e in hits[0][1]} == {"101", "102"}

    def test_short_surname_skipped(self) -> None:
        surnames = roster_surnames(self.SUMMARY)
        assert "li" not in surnames

    def test_substring_never_matches(self) -> None:
        surnames = roster_surnames(self.SUMMARY)
        assert match_segment("the carters society", surnames) == []
