"""One executable colony for video or provider state, checks, findings and review."""

from __future__ import annotations

import copy
from dataclasses import replace

from soccerviz.harness.engine import Specialist


def checked_tactics(source, parents, reviews):
    from soccerviz.vision.specialists import tactics_stage

    checks = parents["checks"]
    state = copy.deepcopy(parents["state"])
    blocked = set(checks["blocking_frame_ids"])
    boundaries = set(checks["break_before_frame_ids"])
    epoch = 0
    for frame in state["frames"]:
        if frame["frame_id"] in boundaries:
            epoch += 1
        frame["shot_id"] = f"{frame['shot_id']}:{epoch}"
        if frame["frame_id"] in blocked:
            frame["possession"]["team"] = None
            frame["possession"]["status"] = "consistency_blocked"
        # Provider-extrapolated positions remain in the shared state but cannot
        # inflate a finding explicitly named visible-team width/depth.
        for player in frame["players"]:
            if player.get("position_status") in {"estimated", "unknown", "unavailable"}:
                player["xy_m"] = None
    result = tactics_stage(source, {"state": state}, reviews)
    result["consistency"] = {
        "status": checks["status"],
        "blocking_frame_ids": sorted(blocked),
        "break_before_frame_ids": sorted(boundaries),
        "policy": "Blocking inconsistencies prevent tactical findings; estimates do not count as observations",
    }
    return result


def checked_brief(source, parents, reviews):
    from soccerviz.vision.specialists import brief_stage

    result = brief_stage(source, parents, reviews)
    checks = parents["checks"]
    if source.get("schema") == "provider-input/v1":
        result["markdown"] = (
            result["markdown"]
            .replace("# Video possession review", "# Provider tracking review")
            .replace("Source video SHA-256", "Source bundle SHA-256")
            .replace("Inspected source PTS", "Inspected provider clock")
            .replace(
                "No coaching recommendation is issued. Ball projection assumes the ground plane; "
                "off-screen players, identity, and metric uncertainty remain unresolved where evidence is absent.",
                "No coaching recommendation is issued. Provider estimates remain separate from observations; "
                "player names, attacking direction and positional uncertainty remain unresolved where absent.",
            )
        )
        result["markdown"] += f"\n\nClock: {source['clock']}."
    result["markdown"] += (
        f"\n\nConsistency checks: **{checks['status']}**; "
        f"{len(checks['violations'])} violations, "
        f"{len(checks['blocking_frame_ids'])} frames blocked from tactical findings. "
        "Skipped checks represent missing evidence, not a pass."
    )
    result["consistency"] = checks
    result["model_acceptance"] = "unmeasured"
    return result


def build_colony(source=None):
    from soccerviz.harness.colony import ColonyRegistry
    from soccerviz.harness.consistency import run_consistency
    from soccerviz.vision.specialists import brief_stage, specialists, tactics_stage

    if (source or {}).get("schema") == "provider-input/v1":
        from soccerviz.harness.provider_colony import provider_evidence, provider_state
        from soccerviz.vision.observation_channels import alternatives

        specs = [
            Specialist(
                "evidence",
                "1",
                (),
                "provider-evidence/v1",
                "cpu",
                "Frozen provider coordinates, source clock and missingness",
                provider_evidence,
            ),
            Specialist(
                "state",
                "1",
                ("evidence",),
                "match-state/v1",
                "cpu",
                "Provider identities stay anonymous; inferred possession remains a hypothesis",
                provider_state,
                uses_reviews=True,
                code_dependencies=(alternatives,),
            ),
            Specialist(
                "tactics",
                "1",
                ("state",),
                "tactical-findings/v1",
                "cpu",
                "Descriptive findings with abstention",
                tactics_stage,
                libraries=("numpy",),
            ),
            Specialist(
                "brief",
                "1",
                ("state", "tactics"),
                "analyst-brief/v1",
                "cpu",
                "Traceable analyst brief",
                brief_stage,
            ),
        ]
    else:
        specs = specialists()
    specs.append(
        Specialist(
            "checks",
            "1",
            ("state", "evidence"),
            "consistency-report/v1",
            "cpu",
            "Independent structural consistency and explicit skipped coverage",
            run_consistency,
        )
    )
    specs = [
        replace(
            s, requires=("state", "checks"), run=checked_tactics, code_dependencies=(tactics_stage,)
        )
        if s.name == "tactics"
        else replace(
            s,
            requires=("state", "tactics", "checks"),
            run=checked_brief,
            code_dependencies=(brief_stage,),
        )
        if s.name == "brief"
        else s
        for s in specs
    ]
    consumers = {
        s.name: tuple(sorted(c.name for c in specs if s.name in c.requires)) for s in specs
    }
    return ColonyRegistry([replace(s, consumers=consumers[s.name]) for s in specs])


def execute_run(harness, run, revision=None, max_stages=None):
    context = harness.context(run, revision)
    source = harness.get(context["input_id"])
    return harness.execute(run, build_colony(source), revision, max_stages)
