import copy
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from soccerviz.harness.engine import Harness
from soccerviz.vision.specialists import import_video, specialists, validate_input


@pytest.fixture
def evidence():
    tables = {k: [] for k in ("frames", "detections", "state", "calibration", "ball_candidates")}
    for fid, time in enumerate((0.0, 0.2, 0.4)):
        tables["frames"].append({"frame_id": fid, "timestamp_s": time, "shot_id": 0})
        tables["calibration"].append(
            {
                "frame_id": fid,
                "timestamp_s": time,
                "accepted": True,
                "homography": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            }
        )
        tables["ball_candidates"].append(
            {
                "frame_id": fid,
                "timestamp_s": time,
                "rank": 0,
                "confidence": 0.9,
                "x0": 19,
                "y0": 29,
                "x1": 21,
                "y1": 31,
            }
        )
        for j in range(14):
            team = int(j >= 7)
            x, y = (20 + j * 3, 30 + j) if team == 0 else (65 + (j - 7) * 3, 20 + j)
            eid = f"f{fid}-d{j}"
            tables["detections"].append(
                {
                    "frame_id": fid,
                    "timestamp_s": time,
                    "detection_id": eid,
                    "tracklet_id": j + 1,
                    "role_hypothesis": "player",
                    "bbox_x0": x,
                    "bbox_y0": y - 2,
                    "bbox_x1": x + 1,
                    "bbox_y1": y,
                    "confidence": 0.9,
                    "source": "synthetic_test_fixture",
                }
            )
            tables["state"].append(
                {
                    "frame_id": fid,
                    "timestamp_s": time,
                    "evidence_id": eid,
                    "entity": "player",
                    "tracklet_id": j + 1,
                    "team_cluster": team,
                    "x_m": x,
                    "y_m": y,
                    "calibration_accepted": True,
                }
            )
    return {
        "schema": "video-input/v1",
        "window": {"start_s": 0, "end_s": 0.6},
        "source_sha256": "synthetic-fixture",
        "model_sha256": {},
        "tables": tables,
    }


def correction(**kwargs):
    return {
        "field": "team",
        "tracklet_id": 1,
        "value": 1,
        "start_s": 0,
        "end_s": 0.4,
        "reviewer": "synthetic-test-reviewer",
        "reason": "Test correction, not real annotation",
        "evidence_ids": ["f0-d0"],
        **kwargs,
    }


def output(harness, run, stage, revision=None):
    row = next(r for r in harness.status(run, revision)["results"] if r["stage"] == stage)
    return harness.get(row["output_id"])["payload"]


def test_resume_reuses_completed_work_and_exports_lineage(tmp_path, evidence):
    harness = Harness(tmp_path)
    run = harness.create(evidence)
    paused = harness.execute(run, specialists(), max_stages=2)
    assert paused["status"] == "paused" and paused["next_stage"] == "camera"
    assert {row["stage"]: row["action"] for row in paused["stages"]} == {
        "evidence": "executed",
        "calibration": "executed",
    }
    resumed = Harness(tmp_path).execute(run, specialists())
    assert {row["stage"]: row["action"] for row in resumed["stages"]} == {
        "evidence": "reused",
        "calibration": "reused",
        "camera": "executed",
        "players": "executed",
        "ball": "executed",
        "context": "executed",
        "state": "executed",
        "tactics": "executed",
        "brief": "executed",
    }
    assert all(s["action"] == "reused" for s in harness.execute(run, specialists())["stages"])
    attempts = harness.status(run)["attempts"]
    for spec in specialists():
        assert [row["status"] for row in attempts if row["stage"] == spec.name] == ["completed"]
    exported = harness.export(run)
    assert "Candidate possession" in (exported / "brief.md").read_text()
    for record in harness.status(run)["results"]:
        artifact = json.loads((exported / f"{record['output_id']}.json").read_text())
        for key in artifact["signature"]["parents"].values():
            assert (exported / f"{key}.json").exists()


def test_correction_recomputes_dependents_preserves_alternatives_and_old_knowledge(
    tmp_path, evidence
):
    harness = Harness(tmp_path)
    run = harness.create(evidence)
    original = harness.execute(run, specialists())
    old_state = output(harness, run, "state", 0)
    assert old_state["frames"][0]["possession"]["team"] == 0
    assert harness.review(run, correction()) == 1
    # Latest revision has no results until its dependent work completes.
    assert harness.status(run)["results"] == []
    revised = harness.execute(run, specialists())
    assert {row["stage"]: row["action"] for row in revised["stages"]} == {
        "evidence": "reused",
        "calibration": "reused",
        "camera": "reused",
        "players": "reused",
        "ball": "reused",
        "context": "executed",
        "state": "executed",
        "tactics": "executed",
        "brief": "executed",
    }
    new_state = output(harness, run, "state")
    assert [f["possession"]["team"] for f in new_state["frames"]] == [1, 1, 0]
    assert [h["value"] for h in new_state["frames"][0]["players"][0]["team_hypotheses"]] == [0, 1]
    assert output(harness, run, "state", 0) == old_state
    replay = harness.execute(run, specialists(), revision=0)
    assert [s["artifact"] for s in replay["stages"]] == [s["artifact"] for s in original["stages"]]
    assert all(s["action"] == "reused" for s in replay["stages"])
    assert harness.context(run, 0)["reviews"] == []
    assert (
        harness.context(run, 1)["reviews"][0]["created_at"] >= harness.context(run, 0)["known_at"]
    )


def test_identity_and_orientation_reviews_keep_unknown_and_prior_candidates(tmp_path, evidence):
    h = Harness(tmp_path)
    run = h.create(evidence)
    h.review(run, correction(field="identity", value="Player A"))
    h.review(run, correction(field="orientation", value=-1, team_cluster=0))
    h.execute(run, specialists())
    frame = output(h, run, "state")["frames"][0]
    assert frame["players"][0]["identity"] == "Player A"
    assert frame["players"][0]["identity_hypotheses"][0]["value"] is None
    assert [c["value"] for c in frame["orientation"]["0"]["hypotheses"]] == [-1, 1, -1]
    h.review(run, correction(field="identity", value=None))
    h.execute(run, specialists())
    assert output(h, run, "state")["frames"][0]["players"][0]["identity"] is None


def test_ambiguity_and_bad_geometry_gate_tactical_claims(tmp_path, evidence):
    ambiguous = copy.deepcopy(evidence)
    ambiguous["tables"]["ball_candidates"].append(
        dict(ambiguous["tables"]["ball_candidates"][0], rank=1, confidence=0.8)
    )
    ambiguous["tables"]["calibration"][1]["accepted"] = False
    h = Harness(tmp_path)
    run = h.create(ambiguous)
    h.execute(run, specialists())
    state = output(h, run, "state")
    assert len(state["frames"][0]["ball_hypotheses"]) == 2
    assert state["frames"][0]["possession"]["team"] is None
    assert state["frames"][1]["players"][0]["xy_m"] is None
    assert output(h, run, "tactics")["unknown_possession_frames"] == 2
    h.review(run, correction(field="possession", value=0, start_s=0.2, evidence_ids=["f1-d0"]))
    h.execute(run, specialists())
    finding = next(f for f in output(h, run, "tactics")["findings"] if f["frame_id"] == 1)
    assert finding["geometry"] is None
    assert finding["tactical_recommendation"]["status"] == "abstain"


def test_review_validation_and_immutable_records(tmp_path, evidence):
    h = Harness(tmp_path)
    run = h.create(evidence)
    for invalid in (
        correction(end_s=5),
        correction(evidence_ids=["invented"]),
        correction(tracklet_id=999),
        correction(value=True),
        correction(reviewer=""),
    ):
        with pytest.raises(ValueError):
            h.review(run, invalid)
    h.review(run, correction())
    with h.connect() as db:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute("UPDATE objects SET body='{}'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM revisions")


def fail_stage(source, parents, reviews):
    raise RuntimeError("Deliberate worker failure")


def test_failed_recomputation_does_not_export_stale_brief(tmp_path, evidence):
    h = Harness(tmp_path)
    run = h.create(evidence)
    h.execute(run, specialists())
    before = h.export(run)
    specs = [
        replace(spec, run=fail_stage) if spec.name == "tactics" else spec for spec in specialists()
    ]
    with pytest.raises(RuntimeError, match="Deliberate"):
        h.execute(run, specs)
    assert {row["stage"] for row in h.status(run)["results"]} == {
        "evidence",
        "calibration",
        "camera",
        "players",
        "ball",
        "context",
        "state",
    }
    assert h.status(run)["attempts"][-1]["stage"] == "tactics"
    assert h.status(run)["attempts"][-1]["status"] == "failed"
    after = h.export(run)
    assert not (after / "brief.md").exists()
    assert (before / "brief.md").exists()
    h.execute(run, specialists())
    assert (h.export(run) / "brief.md").exists()


def test_process_death_recovery(tmp_path, evidence):
    store = tmp_path / "store"
    h = Harness(store)
    run = h.create(evidence)
    worker = tmp_path / "crash_worker.py"
    worker.write_text("""import os
import sys
from pathlib import Path
from dataclasses import replace
from soccerviz.harness.engine import Harness
from soccerviz.vision.specialists import specialists, state_stage

def crash_once(source, parents, reviews):
    marker = Path(sys.argv[1]) / "crash-marker"
    if not marker.exists():
        marker.touch()
        os._exit(19)
    return state_stage(source, parents, reviews)

specs = specialists()
specs = [replace(spec, run=crash_once) if spec.name == "state" else spec for spec in specs]
print(Harness(Path(sys.argv[1])).execute(sys.argv[2], specs))
""")
    command = [sys.executable, str(worker), str(store), run]
    env = os.environ | {"PYTHONPATH": str(Path(__file__).parents[2] / "src")}
    assert (
        subprocess.run(command, env=env, capture_output=True, timeout=20, check=False).returncode
        == 19
    )
    interrupted = h.status(run)["attempts"]
    assert interrupted[-1]["stage"] == "state" and interrupted[-1]["status"] == "running"
    assert {row["stage"] for row in interrupted if row["status"] == "completed"} == {
        "evidence",
        "calibration",
        "camera",
        "players",
        "ball",
        "context",
    }
    recovered = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=20, check=False
    )
    assert recovered.returncode == 0, recovered.stderr
    attempts = h.status(run)["attempts"]
    for spec in specialists():
        expected = ["interrupted", "completed"] if spec.name == "state" else ["completed"]
        assert [row["status"] for row in attempts if row["stage"] == spec.name] == expected
    assert {row["stage"] for row in h.status(run)["results"]} == {
        spec.name for spec in specialists()
    }


def test_evidence_rejects_orphan_and_future_timestamps(evidence):
    validate_input(evidence)
    bad = copy.deepcopy(evidence)
    bad["tables"]["state"][0]["timestamp_s"] = 0.4
    with pytest.raises(ValueError, match="timestamp"):
        validate_input(bad)
    bad = copy.deepcopy(evidence)
    bad["tables"]["frames"][0]["timestamp_s"] = -1
    with pytest.raises(ValueError, match="outside"):
        validate_input(bad)


def test_import_freezes_source_and_detects_changed_inputs(tmp_path, evidence):
    video = tmp_path / "video"
    video.mkdir()
    for name, rows in evidence["tables"].items():
        pd.DataFrame(rows).to_parquet(video / f"{name}.parquet", index=False)
    (video / "report.json").write_text(
        json.dumps({"source_sha256": "fixture", "model_sha256": {}, "sample_hz": 5})
    )
    source = import_video(video, 0, 0.4)
    h = Harness(tmp_path / "store")
    run = h.create(source)
    changed = pd.read_parquet(video / "state.parquet")
    changed.loc[0, "team_cluster"] = 1
    changed.to_parquet(video / "state.parquet", index=False)
    h.execute(run, specialists())
    assert output(h, run, "state")["frames"][0]["players"][0]["team"] == 0
    updated = import_video(video, 0, 0.4)
    assert updated["artifact_sha256"] != source["artifact_sha256"]
    second = h.create(updated)
    h.execute(second, specialists())
    assert output(h, second, "state")["frames"][0]["players"][0]["team"] == 1


def test_version_change_invalidates_descendants_and_single_executor_lock(tmp_path, evidence):
    h = Harness(tmp_path)
    run = h.create(evidence)
    h.execute(run, specialists())
    specs = [
        replace(spec, version="state-test-next-version") if spec.name == "state" else spec
        for spec in specialists()
    ]
    changed = h.execute(run, specs)
    assert {row["stage"]: row["action"] for row in changed["stages"]} == {
        "evidence": "reused",
        "calibration": "reused",
        "camera": "reused",
        "players": "reused",
        "ball": "reused",
        "context": "reused",
        "state": "executed",
        "tactics": "executed",
        "brief": "executed",
    }
    with h.execution_lock(), pytest.raises(RuntimeError, match="Another specialist"):
        Harness(tmp_path).execute(run, specialists())


def test_gaps_and_cuts_break_possession_candidates(tmp_path, evidence):
    changed = copy.deepcopy(evidence)
    changed["window"]["end_s"] = 2
    for rows in changed["tables"].values():
        for row in rows:
            if row["frame_id"] == 2:
                row["timestamp_s"] = 1.0
    changed["tables"]["frames"][1]["shot_id"] = 1
    changed["tables"]["frames"][2]["shot_id"] = 1
    h = Harness(tmp_path)
    run = h.create(changed)
    h.execute(run, specialists())
    assert len(output(h, run, "tactics")["candidate_possessions"]) == 3


def test_batch_review_is_atomic_idempotent_and_rejects_stale_base(tmp_path, evidence):
    h = Harness(tmp_path)
    run = h.create(evidence)
    with pytest.raises(ValueError):
        h.review_batch(run, [correction(), correction(evidence_ids=["missing"])], source_key="bad")
    assert h.context(run)["revision"] == 0
    receipt = h.review_batch(
        run,
        [correction(), correction(field="identity", value="A")],
        source_key="reviewed-file-hash",
        expected_revision=0,
    )
    assert receipt["revision"] == 2
    repeat = h.review_batch(
        run,
        [correction(), correction(field="identity", value="A")],
        source_key="reviewed-file-hash",
        expected_revision=0,
    )
    assert repeat["reused"] and repeat["revision"] == 2
    with pytest.raises(ValueError, match="different corrections"):
        h.review_batch(run, [correction(value=0)], source_key="reviewed-file-hash")
    with pytest.raises(ValueError, match="newer reviews"):
        h.review_batch(run, [correction(value=0)], source_key="new-file", expected_revision=0)
    assert h.context(run)["revision"] == 2


def test_attachments_and_panel_reviews_preserve_source_and_revision(tmp_path, evidence):
    import zipfile

    from soccerviz.ui.harness_ui import HarnessPanel

    h = Harness(tmp_path)
    run = h.create(evidence)
    h.execute(run, specialists())
    panel = HarnessPanel(tmp_path)
    assert panel.choices()[0][1] == run
    new = panel.apply(run, 0, "team", "f0-d0", "1", 0, 0.4, "fixture reviewer", "fixture reason", 0)
    assert new == 1
    key = h.attach(run, "reviewed-ground-truth", {"fixture": True})
    assert h.attach(run, "reviewed-ground-truth", {"fixture": True}) == key
    assert len(h.status(run)["attachments"]) == 1
    assert h.status(run, 0)["attachments"] == []
    view = panel.view(run, new)
    assert view["reviews"][0]["Reviewer"] == "fixture reviewer"
    assert view["evidence"][0]["Team alternatives"] == "[0, 1]"
    with zipfile.ZipFile(panel.download(run, new)) as bundle:
        assert f"{key}.json" in bundle.namelist()
        assert "brief.md" in bundle.namelist()
    assert panel.view(run, 0)["reviews"] == []
    newer = panel.apply(
        run,
        1,
        "identity",
        "f0-d0",
        "fixture",
        0,
        0.4,
        "fixture reviewer",
        "second fixture review",
        0,
    )
    assert newer == 2
    inherited = h.status(run, newer)["attachments"]
    assert len(inherited) == 1 and inherited[0]["revision"] == 1
    assert panel.view(run, newer)["attachments"][0]["attached_at"] == inherited[0]["created_at"]
