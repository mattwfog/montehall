from dataclasses import replace

import pytest

from soccerviz.harness.colony import ColonyRegistry
from soccerviz.harness.engine import Harness, Specialist, digest
from soccerviz.harness.experiment_gates import AcceptanceContract, Criterion, evaluate_acceptance


def source_run(source, parents, reviews):
    return {"schema": "source/v1", "value": source["value"]}


def optional_run(source, parents, reviews):
    return {"schema": "optional/v1", "value": source["value"] * 2}


def consumer_run(source, parents, reviews):
    extra = parents.get("optional")
    return {
        "schema": "consumer/v1",
        "value": parents["source"]["value"] + (extra["value"] if extra else 0),
        "optional_absent": extra is None,
    }


def invalid_run(source, parents, reviews):
    return {"schema": "wrong/v1"}


def specs(optional_available=False):
    return [
        Specialist(
            "consumer",
            "1",
            ("source",),
            "consumer/v1",
            "cpu",
            "fixture",
            consumer_run,
            optional_requires=(("optional", "Use source-only output and report reduced coverage"),),
        ),
        Specialist(
            "optional",
            "1",
            (),
            "optional/v1",
            "cpu",
            "fixture",
            optional_run,
            consumers=("consumer",),
            available=optional_available,
            unavailable_reason="" if optional_available else "No optional provider configured",
        ),
        Specialist(
            "source", "1", (), "source/v1", "cpu", "fixture", source_run, consumers=("consumer",)
        ),
    ]


def test_manifest_is_deterministic_and_orders_available_optional_edges():
    registry = ColonyRegistry(specs(True))
    assert [s.name for s in registry.execution_order()] == ["optional", "source", "consumer"]
    assert registry.manifest() == ColonyRegistry(reversed(specs(True))).manifest()
    consumer = next(row for row in registry.manifest()["organisms"] if row["name"] == "consumer")
    assert consumer["optional"][0]["absent_behavior"]
    assert consumer["output_schema"] == "consumer/v1"


def test_missing_required_and_unknown_optional_are_rejected():
    with pytest.raises(ValueError, match="Unknown dependencies"):
        ColonyRegistry(specs()[:2])
    graph = specs()
    graph[0] = replace(graph[0], requires=("source", "optional"), optional_requires=())
    with pytest.raises(ValueError, match="Required dependencies unavailable"):
        ColonyRegistry(graph)
    graph = specs()
    graph[0] = replace(graph[0], optional_requires=(("unknown", "Unavailable policy"),))
    with pytest.raises(ValueError, match="Unknown dependencies"):
        ColonyRegistry(graph)


def test_required_and_optional_cycles_are_rejected():
    graph = specs()
    graph[2] = replace(graph[2], requires=("consumer",))
    with pytest.raises(ValueError, match="cycle"):
        ColonyRegistry(graph)
    graph = specs(True)
    graph[1] = replace(graph[1], optional_requires=(("consumer", "Explicit fallback"),))
    with pytest.raises(ValueError, match="cycle"):
        ColonyRegistry(graph)


def test_optional_requires_explicit_absence_policy():
    graph = specs()
    graph[0] = replace(graph[0], optional_requires=(("optional", ""),))
    with pytest.raises(ValueError, match="absent behavior"):
        ColonyRegistry(graph)


def test_absent_optional_runs_and_availability_invalidates_only_consumer(tmp_path):
    harness = Harness(tmp_path)
    run = harness.create({"value": 3})
    first = harness.execute(run, specs())
    assert first["status"] == "completed"
    assert first["scientific_status"] == "unmeasured"
    assert [s["stage"] for s in first["stages"]] == ["source", "consumer"]
    envelope = harness.get(first["stages"][-1]["artifact"])
    assert envelope["payload"]["value"] == 3
    assert envelope["signature"]["optional_inputs"]["optional"]["status"] == "absent"
    assert envelope["signature"]["optional_inputs"]["optional"]["artifact"] is None
    second = harness.execute(run, specs(True))
    assert {s["stage"]: s["action"] for s in second["stages"]} == {
        "source": "reused",
        "optional": "executed",
        "consumer": "executed",
    }
    envelope = harness.get(second["stages"][-1]["artifact"])
    assert envelope["payload"]["value"] == 9
    assert envelope["signature"]["optional_inputs"]["optional"]["status"] == "present"
    third = harness.execute(run, specs())
    assert all(s["action"] == "reused" for s in third["stages"])
    assert third["stages"][-1]["artifact"] == first["stages"][-1]["artifact"]
    exported = harness.export(run)
    assert (exported / f"{first['colony_manifest_id']}.json").exists()
    assert (exported / f"{second['colony_manifest_id']}.json").exists()
    assert len(harness.status(run)["colony_manifests"]) == 2


def test_optional_contract_change_invalidates_consumer(tmp_path):
    harness = Harness(tmp_path)
    run = harness.create({"value": 3})
    harness.execute(run, specs())
    changed = specs()
    changed[0] = replace(
        changed[0], optional_requires=(("optional", "A changed declared fallback"),)
    )
    result = harness.execute(run, changed)
    assert [s["action"] for s in result["stages"]] == ["reused", "executed"]


def test_registry_schema_failure_and_resume_remain_durable(tmp_path):
    harness = Harness(tmp_path)
    run = harness.create({"value": 3})
    graph = specs()
    graph[0] = replace(graph[0], run=invalid_run)
    with pytest.raises(ValueError, match="Invalid output schema"):
        harness.execute(run, ColonyRegistry(graph))
    assert harness.status(run)["attempts"][-1]["status"] == "failed"
    result = harness.execute(run, specs())
    assert [s["action"] for s in result["stages"]] == ["reused", "executed"]


def test_empty_colony_and_pause_are_explicit(tmp_path):
    harness = Harness(tmp_path)
    run = harness.create({"value": 3})
    result = harness.execute(run, [])
    assert result["status"] == "completed" and result["stages"] == []
    assert result["scientific_status"] == "unmeasured"
    paused = harness.execute(run, specs(True), max_stages=1)
    assert paused["status"] == "paused"
    resumed = harness.execute(run, specs(True))
    assert [s["action"] for s in resumed["stages"]] == ["reused", "executed", "executed"]


def measured_report(value=0.6):
    return {
        "schema": "acceptance-evidence/v1",
        "provenance": {"source_sha256": "a" * 64, "version_sha256": "b" * 64},
        "metrics": {"accuracy": value},
        "agent_status": "complete",
    }


def contract_for(report, threshold=0.8):
    return AcceptanceContract(
        digest(report), "a" * 64, "b" * 64, (Criterion("metrics.accuracy", ">=", threshold),)
    )


def test_runtime_completion_and_agent_claim_do_not_grant_acceptance():
    report = measured_report()
    assert evaluate_acceptance(report, None)["scientific_status"] == "unmeasured"
    result = evaluate_acceptance(report, contract_for(report))
    assert result["runtime_status"] == "completed"
    assert result["scientific_status"] == "rejected"
    assert result["criteria"][0]["observed"] == 0.6
    assert result["criteria"][0]["passed"] is False


def test_acceptance_requires_exact_artifact_and_source_version_provenance():
    report = measured_report(0.9)
    contract = contract_for(report)
    assert evaluate_acceptance(report, contract)["scientific_status"] == "accepted"
    altered = measured_report(0.95)
    assert "report artifact hash mismatch" in evaluate_acceptance(altered, contract)["reasons"]
    wrong_source = replace(contract, source_sha256="c" * 64)
    assert "source hash mismatch" in evaluate_acceptance(report, wrong_source)["reasons"]
    wrong_version = replace(contract, version_sha256="c" * 64)
    assert "version hash mismatch" in evaluate_acceptance(report, wrong_version)["reasons"]
    assert (
        evaluate_acceptance(report, contract, runtime_status="paused")["scientific_status"]
        == "unmeasured"
    )


def test_acceptance_rejects_missing_and_boolean_measurements():
    for value in (None, True, "0.99"):
        report = measured_report(value)
        result = evaluate_acceptance(report, contract_for(report))
        assert result["scientific_status"] == "rejected"
        assert "non-numeric" in result["reasons"][0]
    with pytest.raises(ValueError, match="explicit numeric"):
        AcceptanceContract("a" * 64, "b" * 64, "c" * 64, ())
    with pytest.raises(ValueError, match="SHA256"):
        AcceptanceContract(
            "not-a-hash", "b" * 64, "c" * 64, (Criterion("metrics.accuracy", ">=", 0.8),)
        )
