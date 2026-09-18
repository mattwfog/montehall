"""Inspect durable runs and submit evidence-linked contextual reviews."""

import json
import uuid
import zipfile
from pathlib import Path

import gradio as gr
import pandas as pd

from soccerviz.harness.colony_runtime import build_colony, execute_run
from soccerviz.harness.engine import Harness


class HarnessPanel:
    def __init__(self, root):
        self.harness = Harness(Path(root))

    def choices(self):
        choices = []
        for run in self.harness.list_runs():
            context = self.harness.context(run["id"])
            source = self.harness.get(context["input_id"])
            window = source.get("window", {})
            label = (
                f"{'Tracking' if source.get('source_kind') == 'provider' else 'Video'} · "
                f"{run['created_at'][:19]} · {window.get('start_s', '?')}–"
                f"{window.get('end_s', '?')}s · revision {run['revision']} · {run['id'][:8]}"
            )
            choices.append((label, run["id"]))
        return choices

    def sources(self, artifacts):
        candidates = []
        for path in sorted(Path(artifacts).rglob("report.json")):
            folder = path.parent
            if (folder / "frames.parquet").exists() and (folder / "detections.parquet").exists():
                kind = "video"
            elif (folder / "observations.parquet").exists():
                report = json.loads(path.read_text())
                if not (
                    report.get("provider") == "Metrica CSV"
                    or report.get("schema") == "skillcorner-ingestion/v1"
                ):
                    continue
                kind = "provider"
            else:
                continue
            candidates.append(
                (f"{kind} · {folder.relative_to(artifacts)}", f"{kind}|{folder.resolve()}")
            )
        return candidates

    def start(self, selection, start, end):
        if not selection:
            raise ValueError("Select an existing video or provider import")
        kind, folder = selection.split("|", 1)
        if kind == "provider":
            from soccerviz.harness.provider_colony import import_provider

            source = import_provider(Path(folder), float(start), float(end))
        else:
            from soccerviz.vision.specialists import import_video

            source = import_video(Path(folder), float(start), float(end))
        run = self.harness.create(source)
        execute_run(self.harness, run)
        return run

    def view(self, run, revision):
        if not run or revision is None:
            return {
                "summary": "No run selected.",
                "brief": "",
                "stages": [],
                "evidence": [],
                "reviews": [],
                "attachments": [],
                "evidence_choices": [],
                "diagnostics": {},
            }
        status = self.harness.status(run, int(revision))
        source = self.harness.get(status["input_id"])
        outputs = {
            row["stage"]: self.harness.get(row["output_id"])["payload"] for row in status["results"]
        }
        stages = []
        for spec in build_colony(source).execution_order():
            name = spec.name
            attempts = [
                a
                for a in status["attempts"]
                if a["stage"] == name and a["revision"] == int(revision)
            ]
            stages.append(
                {
                    "Specialist": name,
                    "Status": "available"
                    if name in outputs
                    else (attempts[-1]["status"] if attempts else "pending"),
                    "Attempts this revision": len(attempts),
                    "Quality acceptance": "unmeasured",
                }
            )
        evidence_rows = []
        for frame in outputs.get("state", {}).get("frames", []):
            for player in frame["players"]:
                evidence_rows.append(
                    {
                        "Time (s)": frame["timestamp_s"],
                        "Frame": frame["frame_id"],
                        "Track": player["tracklet_id"],
                        "Team": player["team"],
                        "Identity": player["identity"],
                        "Evidence": player["evidence_id"],
                        "Team alternatives": json.dumps(
                            [p["value"] for p in player["team_hypotheses"]]
                        ),
                    }
                )
        review_rows = [
            {
                "Revision": item["revision"],
                "Reviewer": item["review"]["reviewer"],
                "Field": item["review"]["field"],
                "Value": str(item["review"]["value"]),
                "From (s)": item["review"]["start_s"],
                "To (s)": item["review"]["end_s"],
                "Reason": item["review"]["reason"],
            }
            for item in status["reviews"]
        ]
        observations = source.get("tables", {}).get("detections", [])
        if source.get("source_kind") == "provider":
            observations = [
                {**r, "detection_id": r["evidence_id"], "role_hypothesis": r["entity"]}
                for r in source["observations"]
                if r["entity"] == "player"
            ]
        choices = [
            (
                f"{r['timestamp_s']:.2f}s · track {r['tracklet_id']} · {r['detection_id']}",
                r["detection_id"],
            )
            for r in observations
            if r["role_hypothesis"] == "player"
        ]
        return {
            "summary": f"Revision **{revision}** · state reviews recorded through {status['known_at']}. "
            f"**{len(status['reviews'])}** contextual corrections.",
            "brief": outputs.get("brief", {}).get(
                "markdown", "Analysis is pending for this revision."
            ),
            "stages": stages,
            "evidence": evidence_rows,
            "reviews": review_rows,
            "attachments": [
                {
                    "kind": a["kind"],
                    "attached_at": a["created_at"],
                    "attached_revision": a["revision"],
                    "payload": self.harness.get(a["object_id"]),
                }
                for a in status["attachments"]
            ],
            "evidence_choices": choices,
            "diagnostics": {
                "channels": outputs.get("state", {}).get("channels", {}),
                "consistency": outputs.get("checks", {}),
                "manifests": status.get("colony_manifests", []),
            },
        }

    def apply(self, run, revision, field, evidence_id, value, start, end, reviewer, reason, team):
        if not run or not evidence_id:
            raise ValueError("Select a run and supporting player observation")
        context = self.harness.context(run, int(revision))
        source = self.harness.get(context["input_id"])
        candidates = source.get("tables", {}).get("detections", [])
        if source.get("source_kind") == "provider":
            candidates = [{**r, "detection_id": r["evidence_id"]} for r in source["observations"]]
        row = next((r for r in candidates if r["detection_id"] == evidence_id), None)
        if row is None:
            raise ValueError("Observation is not part of this run")
        clean = value.strip()
        parsed = (
            None
            if clean.lower() in {"", "unknown", "null"}
            else (clean if field == "identity" else int(clean))
        )
        review = {
            "field": field,
            "value": parsed,
            "start_s": float(start),
            "end_s": float(end),
            "reviewer": reviewer,
            "reason": reason,
            "evidence_ids": [evidence_id],
        }
        if field in {"identity", "team"}:
            review["tracklet_id"] = row["tracklet_id"]
        if field == "orientation":
            review["team_cluster"] = int(team)
        from soccerviz.harness.engine import digest

        receipt = self.harness.review_batch(
            run,
            [review],
            source_key=digest({"base_revision": revision, "review": review}),
            expected_revision=int(revision),
        )
        result = execute_run(self.harness, run, revision=receipt["revision"])
        return result["revision"]

    def download(self, run, revision):
        folder = self.harness.export(run, int(revision))
        target = folder.with_suffix(".zip")
        temporary = target.with_suffix(f".{uuid.uuid4().hex}.zip.tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in folder.iterdir():
                if file.is_file():
                    archive.write(file, file.name)
        temporary.replace(target)
        return str(target.resolve())


def build_harness_tab(artifacts):
    panel = HarnessPanel(artifacts / "harness")
    choices = panel.choices()
    first = choices[0][1] if choices else None
    latest = panel.harness.context(first)["revision"] if first else None
    initial = panel.view(first, latest)
    with gr.Tab("Runs & review"):
        gr.Markdown(
            "## Soccer colony: evidence to match state\nAnalyze video or provider tracking, inspect specialists, "
            "compare revisions, "
            "and correct contextual labels with supporting observations."
        )
        with gr.Accordion("Start an analysis from existing data", open=False):
            input_source = gr.Dropdown(
                panel.sources(artifacts), label="Imported source", value=None
            )
            with gr.Row():
                input_start = gr.Number(value=0, label="Start (source clock seconds)")
                input_end = gr.Number(value=10, label="End (exclusive)")
            gr.Markdown(
                "SkillCorner uses its frame clock here; the match clock remains separate. "
                "Provider estimates retain their status throughout analysis."
            )
            launch = gr.Button("Run soccer analysis", variant="primary")
            launch_status = gr.Markdown()
        with gr.Row():
            run = gr.Dropdown(choices, value=first, label="Analysis run", scale=4)
            revision = gr.Dropdown(
                list(range(latest + 1)) if latest is not None else [],
                value=latest,
                label="Revision",
                scale=1,
            )
            refresh = gr.Button("Refresh runs", scale=1)
        summary = gr.Markdown(initial["summary"])
        brief = gr.Markdown(initial["brief"])
        with gr.Accordion("Specialists and supporting observations", open=False):
            stages = gr.Dataframe(
                pd.DataFrame(initial["stages"]), interactive=False, label="Execution"
            )
            evidence = gr.Dataframe(
                pd.DataFrame(initial["evidence"]),
                interactive=False,
                label="Player observations and alternatives",
                max_height=350,
            )
            attachments = gr.JSON(initial["attachments"], label="Imported evidence and evaluations")
            diagnostics = gr.JSON(
                initial["diagnostics"], label="Channel coverage, consistency and colony manifest"
            )
        with gr.Accordion("Add a reviewed correction", open=False):
            gr.Markdown(
                "Review the footage first. Corrections create a new revision; "
                "they do not become training or evaluation labels automatically."
            )
            observation = gr.Dropdown(
                initial["evidence_choices"], value=None, label="Supporting player observation"
            )
            with gr.Row():
                field = gr.Dropdown(
                    [
                        ("Team assignment", "team"),
                        ("Player identity", "identity"),
                        ("Possession team", "possession"),
                        ("Attacking direction", "orientation"),
                    ],
                    value="team",
                    label="Correction",
                )
                value = gr.Textbox(
                    label="Corrected value",
                    info="Team: 0 or 1. Direction: -1 or 1. "
                    "Identity: player label. Use unknown to clear a selection.",
                )
                team = gr.Dropdown([0, 1], value=0, label="Team for direction correction")
            with gr.Row():
                start = gr.Number(value=None, label="Effective from (source seconds)")
                end = gr.Number(value=None, label="Effective until (exclusive)")
            reviewer = gr.Textbox(label="Reviewer")
            reason = gr.Textbox(label="Reason and observed evidence", lines=2)
            apply = gr.Button("Save correction and recompute", variant="primary")
            message = gr.Markdown()
        reviews = gr.Dataframe(
            pd.DataFrame(initial["reviews"]), interactive=False, label="Review history"
        )
        export = gr.Button("Export brief and evidence")
        download = gr.File(interactive=False, label="Run bundle")

        def render(selected, version):
            view = panel.view(selected, version)
            return (
                view["summary"],
                view["brief"],
                pd.DataFrame(view["stages"]),
                pd.DataFrame(view["evidence"]),
                pd.DataFrame(view["reviews"]),
                view["attachments"],
                gr.Dropdown(choices=view["evidence_choices"], value=None),
                view["diagnostics"],
            )

        def versions(selected):
            latest = panel.harness.context(selected)["revision"] if selected else None
            return gr.Dropdown(
                choices=list(range(latest + 1)) if latest is not None else [], value=latest
            )

        def refresh_runs():
            choices = panel.choices()
            return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)

        def observation_interval(selected, evidence_id):
            if not selected or not evidence_id:
                return None, None
            context = panel.harness.context(selected)
            source = panel.harness.get(context["input_id"])
            rows = source.get("tables", {}).get("detections", [])
            if source.get("source_kind") == "provider":
                rows = [{**r, "detection_id": r["evidence_id"]} for r in source["observations"]]
            row = next(r for r in rows if r["detection_id"] == evidence_id)
            time = row["timestamp_s"]
            return time, min(source["window"]["end_s"], time + source.get("sample_period_s", 0.2))

        def apply_review(*args):
            try:
                new = panel.apply(*args)
                return gr.Dropdown(
                    choices=list(range(new + 1)), value=new
                ), f"Saved revision {new}."
            except (ValueError, RuntimeError) as error:
                raise gr.Error(str(error)) from error

        def start_run(selection, start, end):
            try:
                new_run = panel.start(selection, start, end)
                return gr.Dropdown(
                    choices=panel.choices(), value=new_run
                ), "Analysis complete; model quality remains unmeasured."
            except (ValueError, RuntimeError) as error:
                raise gr.Error(str(error)) from error

        refresh.click(refresh_runs, outputs=run)
        display_outputs = [
            summary,
            brief,
            stages,
            evidence,
            reviews,
            attachments,
            observation,
            diagnostics,
        ]
        run.change(versions, run, revision).then(render, [run, revision], display_outputs)
        observation.change(observation_interval, [run, observation], [start, end])
        revision.change(
            render,
            [run, revision],
            display_outputs,
        )
        launch.click(
            start_run,
            [input_source, input_start, input_end],
            [run, launch_status],
            concurrency_id="harness-execution",
            concurrency_limit=1,
        )
        apply.click(
            apply_review,
            [run, revision, field, observation, value, start, end, reviewer, reason, team],
            [revision, message],
            concurrency_id="harness-execution",
            concurrency_limit=1,
        )
        export.click(panel.download, [run, revision], download)
