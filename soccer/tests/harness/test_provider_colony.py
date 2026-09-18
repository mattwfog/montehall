"""Integration checks over existing local public imports; never modify those imports."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from soccerviz.core.assets import sha256
from soccerviz.harness.colony_runtime import checked_tactics, execute_run
from soccerviz.harness.consistency import run_consistency
from soccerviz.harness.engine import Harness, digest
from soccerviz.harness.provider_colony import import_provider, provider_evidence, provider_state

ROOT = Path(__file__).resolve().parents[2]
METRICA = ROOT / "artifacts/integrations/kloppy-game1"
SKILLCORNER = ROOT / "artifacts/integrations/skillcorner/game-1886347"


@pytest.fixture(scope="module")
def metrica():
    if not (METRICA / "observations.parquet").exists():
        pytest.skip("Local Metrica integration artifacts unavailable")
    return import_provider(METRICA, 0.04, 1.04)


@pytest.fixture(scope="module")
def skillcorner():
    if not (SKILLCORNER / "observations.parquet").exists():
        pytest.skip("Local SkillCorner integration artifacts unavailable")
    return import_provider(SKILLCORNER, 0, 2)


def state_of(source):
    evidence = provider_evidence(source, {}, [])
    state = provider_state(source, {"evidence": evidence}, [])
    return evidence, state


def payload(harness, run, stage, revision=None):
    row = next(r for r in harness.status(run, revision)["results"] if r["stage"] == stage)
    return harness.get(row["output_id"])["payload"]


def test_metrica_preserves_native_coordinates_frame_clock_and_source_hashes(metrica):
    table = pd.read_parquet(METRICA / "observations.parquet")
    observation = next(r for r in metrica["observations"] if r["xy_m"] is not None)
    raw = table[
        (table.frame_id == observation["frame_id"])
        & (table.provider_entity_id == observation["provider_entity_id"])
    ].iloc[0]
    assert observation["source_xy_m"] == pytest.approx([raw.x_m, raw.y_m])
    assert observation["xy_m"] == pytest.approx([raw.x_m, raw.y_m])
    assert metrica["frames"][0]["timestamp_s"] == 0.04
    assert metrica["frames"][0]["frame_id"] == 1
    assert metrica["pitch"]["translation_m"] == [0, 0]
    assert metrica["artifact_sha256"] == {
        "observations": sha256(METRICA / "observations.parquet"),
        "report": sha256(METRICA / "report.json"),
    }
    assert metrica["source_report"]["source_sha256"]


def test_skillcorner_translation_preserves_raw_axes_and_separate_match_clock(skillcorner):
    table = pd.read_parquet(SKILLCORNER / "observations.parquet")
    observation = next(r for r in skillcorner["observations"] if r["xy_m"] is not None)
    raw = table[
        (table.frame_id == observation["frame_id"])
        & (table.provider_entity_id == observation["provider_entity_id"])
    ].iloc[0]
    dx, dy = skillcorner["pitch"]["translation_m"]
    assert observation["source_xy_m"] == pytest.approx([raw.x_m, raw.y_m])
    assert observation["xy_m"] == pytest.approx([raw.x_m + dx, raw.y_m + dy])
    assert observation["timestamp_s"] == 1.0
    assert observation["match_timestamp_s"] == 0.0
    assert skillcorner["frames"][0]["timestamp_s"] == 0.0
    early = [r for r in skillcorner["observations"] if r["timestamp_s"] < 1]
    assert early and all(r["match_timestamp_s"] is None for r in early)
    assert all(r["position_status"] == "unavailable" for r in early)


def test_provider_extrapolations_keep_status_and_do_not_count_as_observations(skillcorner):
    evidence, state = state_of(skillcorner)
    estimates = [r for r in skillcorner["observations"] if r["provider_status"] == "extrapolated"]
    assert estimates
    assert all(r["position_status"] == "estimated" and r["xy_m"] is not None for r in estimates)
    frame = next(
        f for f in state["frames"] if any(p["position_status"] == "estimated" for p in f["players"])
    )
    frame["possession"] = {
        "team": 0,
        "status": "reviewed",
        "hypotheses": [
            {"value": 0, "source": "review", "evidence_ids": [frame["players"][0]["evidence_id"]]}
        ],
    }
    for other in state["frames"]:
        if other is not frame:
            other["possession"]["team"] = None
    before = copy.deepcopy(state)
    checks = run_consistency(skillcorner, {"state": state, "evidence": evidence}, [])
    findings = checked_tactics(skillcorner, {"state": state, "checks": checks}, [])
    finding = next(r for r in findings["findings"] if r["frame_id"] == frame["frame_id"])
    observed = [
        p
        for p in frame["players"]
        if p["team"] == 0 and p["position_status"] == "observed" and p["xy_m"] is not None
    ]
    assert finding["observed_players"] == len(observed)
    assert finding["evidence_ids"] == [p["evidence_id"] for p in observed]
    assert state == before


def test_immutable_snapshot_survives_mutated_caller_and_replay(tmp_path, metrica):
    source = copy.deepcopy(metrica)
    harness = Harness(tmp_path / "store")
    run = harness.create(source)
    input_id = harness.context(run)["input_id"]
    assert input_id == digest(source)
    source["observations"][0]["xy_m"] = [999, 999]
    assert harness.get(input_id) == metrica
    first = execute_run(harness, run)
    second = execute_run(harness, run)
    assert first["status"] == "completed"
    assert all(s["action"] == "reused" for s in second["stages"])
    assert [s["stage"] for s in first["stages"]] == [
        "evidence",
        "state",
        "checks",
        "tactics",
        "brief",
    ]
    assert "Provider tracking review" in payload(harness, run, "brief")["markdown"]
    assert payload(harness, run, "state")["source_kind"] == "provider"
    assert payload(harness, run, "checks")["rules"]["calibration_gate"]["status"] == "skipped"


def test_consistency_duplicate_track_blocks_findings_without_rewriting_evidence(metrica):
    evidence, state = state_of(metrica)
    for frame in state["frames"]:
        frame["possession"] = {
            "team": 0,
            "status": "reviewed",
            "hypotheses": [
                {
                    "value": 0,
                    "source": "review",
                    "evidence_ids": [frame["players"][0]["evidence_id"]],
                }
            ],
        }
    before = checked_tactics(
        metrica,
        {
            "state": state,
            "checks": run_consistency(metrica, {"state": state, "evidence": evidence}, []),
        },
        [],
    )
    first = state["frames"][0]
    assert any(r["frame_id"] == first["frame_id"] for r in before["findings"])
    first["players"].append(copy.deepcopy(first["players"][0]))
    frozen = copy.deepcopy(state)
    checks = run_consistency(metrica, {"state": state, "evidence": evidence}, [])
    assert first["frame_id"] in checks["blocking_frame_ids"]
    findings = checked_tactics(metrica, {"state": state, "checks": checks}, [])
    assert all(r["frame_id"] != first["frame_id"] for r in findings["findings"])
    assert state == frozen
    assert any(v["code"] == "duplicate_simultaneous_tracklet" for v in checks["violations"])


def test_provider_review_creates_revision_preserves_original_and_invalidates_dependents(
    tmp_path, metrica
):
    harness = Harness(tmp_path / "store")
    run = harness.create(metrica)
    execute_run(harness, run)
    old = payload(harness, run, "state", 0)
    row = next(r for r in metrica["observations"] if r["entity"] == "player")
    review = {
        "field": "team",
        "tracklet_id": row["tracklet_id"],
        "value": 1 - row["team"],
        "start_s": metrica["window"]["start_s"],
        "end_s": metrica["window"]["end_s"],
        "reviewer": "integration-test-fixture",
        "reason": "Test revision mechanics; not a real provider correction",
        "evidence_ids": [row["evidence_id"]],
    }
    assert harness.review(run, review) == 1
    result = execute_run(harness, run)
    assert result["stages"][0]["action"] == "reused"
    assert all(s["action"] == "executed" for s in result["stages"][1:])
    revised = payload(harness, run, "state", 1)
    player = next(
        p for p in revised["frames"][0]["players"] if p["tracklet_id"] == row["tracklet_id"]
    )
    assert player["team"] == 1 - row["team"]
    assert [h["value"] for h in player["team_hypotheses"]] == [row["team"], 1 - row["team"]]
    assert payload(harness, run, "state", 0) == old
    assert harness.get(harness.context(run)["input_id"]) == metrica


def test_skillcorner_rejects_changed_observations_against_recorded_output_hash(
    tmp_path, skillcorner
):
    folder = tmp_path / "provider"
    folder.mkdir()
    shutil.copy2(SKILLCORNER / "report.json", folder / "report.json")
    table = pd.read_parquet(SKILLCORNER / "observations.parquet")
    index = table.x_m.first_valid_index()
    table.at[index, "x_m"] += 1
    table.to_parquet(folder / "observations.parquet", index=False)
    with pytest.raises(ValueError, match="hash|checksum|changed|integrity"):
        import_provider(folder, 0, 2)


def test_provider_rejects_concurrent_artifact_changes(tmp_path, metrica, monkeypatch):
    folder = tmp_path / "provider"
    folder.mkdir()
    for name in ("report.json", "observations.parquet"):
        shutil.copy2(METRICA / name, folder / name)
    original_read = pd.read_parquet

    def changed_read(path, *args, **kwargs):
        result = original_read(path, *args, **kwargs)
        report = json.loads((folder / "report.json").read_text())
        report["test_concurrent_change"] = True
        (folder / "report.json").write_text(json.dumps(report))
        return result

    monkeypatch.setattr(pd, "read_parquet", changed_read)
    with pytest.raises(ValueError, match="changed during import"):
        import_provider(folder, 0.04, 1.04)


def test_skillcorner_unknown_detection_status_preserves_coordinates_without_using_them(
    tmp_path, skillcorner
):
    folder = tmp_path / "unknown-status-fixture"
    folder.mkdir()
    table = pd.read_parquet(SKILLCORNER / "observations.parquet")
    ball = table[(table.entity == "ball") & (table.status == "detected") & table.x_m.notna()].iloc[
        0
    ]
    player_index = table[
        (table.frame_id == ball.frame_id)
        & (table.entity == "player")
        & (table.status == "detected")
        & table.x_m.notna()
    ].index[0]
    # Force this player's finite position to be nearest to the detected ball.
    # Its unknown detection status must still prevent possession support.
    table.loc[player_index, ["x_m", "y_m"]] = [ball.x_m, ball.y_m]
    table.loc[player_index, "status"] = "unknown_detection_status"
    table.loc[player_index, "is_detected"] = None
    table.to_parquet(folder / "observations.parquet", index=False)
    report = json.loads((SKILLCORNER / "report.json").read_text())
    report["output_sha256"]["observations"] = sha256(folder / "observations.parquet")
    report["test_fixture"] = (
        "One unknown-status player; altered coordinates are not real observations"
    )
    (folder / "report.json").write_text(json.dumps(report))
    start = float(ball.source_frame_time_s)
    source = import_provider(folder, start, start + 0.05)
    unknown = next(r for r in source["observations"] if r["position_status"] == "unknown")
    assert unknown["provider_status"] == "unknown_detection_status"
    assert unknown["source_xy_m"] == pytest.approx([ball.x_m, ball.y_m])
    assert unknown["xy_m"] is not None
    evidence, state = state_of(source)
    assert evidence["coverage"]["unknown"] == 1
    frame = state["frames"][0]
    assert all(
        unknown["evidence_id"] not in hypothesis.get("evidence_ids", [])
        for hypothesis in frame["possession"]["hypotheses"]
    )
    frame["possession"] = {
        "team": unknown["team"],
        "status": "reviewed",
        "hypotheses": [
            {"value": unknown["team"], "source": "review", "evidence_ids": [unknown["evidence_id"]]}
        ],
    }
    checks = run_consistency(source, {"state": state, "evidence": evidence}, [])
    findings = checked_tactics(source, {"state": state, "checks": checks}, [])
    finding = findings["findings"][0]
    assert unknown["evidence_id"] not in finding["evidence_ids"]
    assert finding["observed_players"] == sum(
        p["team"] == unknown["team"]
        and p["position_status"] == "observed"
        and p["xy_m"] is not None
        for p in frame["players"]
    )
    retained = next(p for p in frame["players"] if p["evidence_id"] == unknown["evidence_id"])
    assert retained["position_status"] == "unknown" and retained["xy_m"] is not None
