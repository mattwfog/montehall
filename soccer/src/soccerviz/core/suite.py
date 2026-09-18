"""One-command local research suite and an honest component-results inventory."""

import json
from pathlib import Path

from soccerviz.core.assets import sha256
from soccerviz.core.data import write_json


def collect_results(artifacts: Path):
    specs = [
        (
            "Video ingestion / detection / ball",
            "video/demo-v2/report.json",
            "real video diagnostics",
            "Labeled video accuracy still required",
        ),
        (
            "Pitch calibration",
            "geometry-report.json",
            "synthetic benchmark + video fits",
            "Real metric accuracy unmeasured",
        ),
        (
            "Tracklet / team association",
            "identity-report.json",
            "synthetic benchmark + video spot-check",
            "Long-term identity unvalidated",
        ),
        (
            "Jersey OCR / named identity",
            "jersey/report.json",
            "real crop experiment; abstains",
            "No reliable jersey evidence or roster",
        ),
        (
            "Video → tactical state",
            "video/demo-v2/bridge-report.json",
            "domain-transfer diagnostics",
            "Orientation and possession unresolved",
        ),
        (
            "Missing-player reconstruction",
            "uncertainty/report.json",
            "simulated crop benchmark",
            "Perfect observed identities assumed",
        ),
        (
            "Progression / SHAP",
            "baseline-report.json",
            "match-disjoint development result",
            "Not causal value",
        ),
        (
            "Deterministic trajectories",
            "forecast-report.json",
            "Spark GPU development result",
            "Player motion only",
        ),
        (
            "Probabilistic trajectories",
            "probabilistic/report.json",
            "Spark GPU + half-disjoint calibration",
            "Undercoverage on evaluation match",
        ),
        (
            "Ball trajectories",
            "ball/report.json",
            "Spark GPU development result",
            "One-second 2D motion on complete windows",
        ),
        (
            "Event actor / possession hypotheses",
            "actor/report.json",
            "provider-event comparison",
            "Nearest actor is not continuous possession ground truth",
        ),
        (
            "Tactics / pass options / set pieces / retrieval",
            "tactics/report.json",
            "measured baselines and descriptive proxies",
            "Next-action accuracy below majority baseline",
        ),
        (
            "Manager brief / scenarios / lineup",
            "manager/report.json",
            "evidence-linked brief and bounded solver",
            "Real lineup advice abstains without context",
        ),
        (
            "Evidence contract / annotation evaluation",
            "video/demo-v2/stage-manifest.json",
            "validated stage artifacts",
            "Independent labels pending",
        ),
    ]
    components = []
    for name, file, status, limit in specs:
        path = artifacts / file
        components.append(
            {
                "component": name,
                "status": status if path.exists() else "not run",
                "report": file,
                "limitation": limit,
                "sha256": sha256(path) if path.exists() else None,
            }
        )
    report = {
        "schema_version": "0.2",
        "components": components,
        "components_with_artifacts": sum(c["sha256"] is not None for c in components),
        "components_total": len(components),
        "elite_performance_established": False,
    }
    write_json(artifacts / "suite-report.json", report)
    return report


def run_suite(data: Path, artifacts: Path):
    from soccerviz.core.bridge import audit_video_features
    from soccerviz.core.geometry import synthetic_calibration_benchmark
    from soccerviz.core.identity import synthetic_identity_benchmark
    from soccerviz.datasets.evaluation import audit_video_contract, export_annotation_task
    from soccerviz.modeling.manager import run_manager
    from soccerviz.modeling.tactics import run_tactics
    from soccerviz.modeling.uncertainty import run_uncertainty
    from soccerviz.vision.ball import evaluate_actors

    print("Tactical models and retrieval", flush=True)
    run_tactics(data, artifacts / "tactics")
    print("Missing-player benchmark", flush=True)
    run_uncertainty(data, artifacts / "uncertainty")
    print("Geometry, identity, and manager checks", flush=True)
    write_json(artifacts / "geometry-report.json", synthetic_calibration_benchmark())
    write_json(artifacts / "identity-report.json", synthetic_identity_benchmark())
    run_manager(data, artifacts)
    evaluate_actors(data, artifacts / "actor")
    video = artifacts / "video/demo-v2"
    if (video / "report.json").exists():
        audit_video_features(video, artifacts)
        export_annotation_task(video)
        audit_video_contract(video)
    return collect_results(artifacts)


if __name__ == "__main__":
    print(json.dumps(run_suite(Path("data"), Path("artifacts")), indent=2))
