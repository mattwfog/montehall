"""Mechanical acceptance of immutable measured reports against explicit criteria.

Execution completion is separate from scientific acceptance. This module supplies
no thresholds and does not accept an agent's self-reported 'complete' as evidence.
"""

from __future__ import annotations

import math
import operator
import re
from dataclasses import asdict, dataclass

from soccerviz.harness.engine import digest

OPERATORS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}


@dataclass(frozen=True)
class Criterion:
    path: str
    operator: str
    threshold: float

    def __post_init__(self):
        if not self.path or self.operator not in OPERATORS:
            raise ValueError("Criterion needs a metric path and supported numeric operator")
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(self.threshold)
        ):
            raise ValueError("Acceptance threshold must be finite numeric data")


@dataclass(frozen=True)
class AcceptanceContract:
    report_sha256: str
    source_sha256: str
    version_sha256: str
    criteria: tuple[Criterion, ...]
    report_schema: str = "acceptance-evidence/v1"

    def __post_init__(self):
        for value in (self.report_sha256, self.source_sha256, self.version_sha256):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("Acceptance requires exact SHA256 report/source/version hashes")
        if not self.criteria:
            raise ValueError("Acceptance requires explicit numeric criteria")
        if len({c.path for c in self.criteria}) != len(self.criteria):
            raise ValueError("Acceptance metric paths must be unique")

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "criteria": tuple(Criterion(**c) for c in value["criteria"])})


def evaluate_acceptance(
    report: dict, contract: AcceptanceContract | None, *, runtime_status="completed"
):
    report_id = digest(report)
    result = {
        "schema": "scientific-acceptance/v1",
        "runtime_status": runtime_status,
        "scientific_status": "unmeasured",
        "report_sha256": report_id,
        "criteria": [],
        "reasons": [],
    }
    if contract is None:
        result["reasons"] = [
            "No explicit acceptance contract; runtime completion is not scientific acceptance."
        ]
        return result
    result["contract"] = asdict(contract)
    result["contract_sha256"] = digest(asdict(contract))
    if runtime_status != "completed":
        result["reasons"] = ["Runtime has not completed."]
        return result
    provenance = report.get("provenance", {})
    checks = {
        "report artifact hash": report_id == contract.report_sha256,
        "report schema": report.get("schema") == contract.report_schema,
        "source hash": provenance.get("source_sha256") == contract.source_sha256,
        "version hash": provenance.get("version_sha256") == contract.version_sha256,
    }
    failures = [f"{label} mismatch" for label, passed in checks.items() if not passed]
    if failures:
        result.update(scientific_status="rejected", reasons=failures)
        return result
    for criterion in contract.criteria:
        observed = report
        for key in criterion.path.split("."):
            observed = observed.get(key) if isinstance(observed, dict) else None
        numeric = (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and math.isfinite(observed)
        )
        passed = numeric and OPERATORS[criterion.operator](observed, criterion.threshold)
        result["criteria"].append(
            {**asdict(criterion), "observed": observed, "passed": bool(passed)}
        )
        if not passed:
            result["reasons"].append(
                f"{criterion.path}: "
                + ("missing/non-numeric measurement" if not numeric else "criterion not met")
            )
    result["scientific_status"] = "rejected" if result["reasons"] else "accepted"
    return result
