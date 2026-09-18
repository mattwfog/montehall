"""One upload, a revisable replay, and an explicit human tactical review queue."""

import json
import time
import uuid
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from soccerviz.core.assets import sha256
from soccerviz.vision.video_workflow import CALIBRATIONS, PRESETS


def runs(artifacts):
    found = []
    for report in Path(artifacts).rglob("report.json"):
        try:
            data = json.loads(report.read_text())
        except (OSError, ValueError):
            continue
        if data.get("schema") == "video-workflow/v1":
            found.append(
                (
                    f"{data['preset']} · {report.parent.relative_to(artifacts)}",
                    str(report.parent.resolve()),
                )
            )
    for game in (1, 2):
        folder = Path(artifacts) / f"integrations/formation-metrica-game-{game}"
        if (folder / "source-observations.parquet").exists():
            found.append(
                (
                    f"Metrica game {game} · provider shape review",
                    "provider::" + str(folder.resolve()),
                )
            )
    return found


def provider_view(folder, frame):
    root = Path(folder.removeprefix("provider::"))
    source = pd.read_parquet(root / "source-observations.parquet")
    frames = source[["frame_id", "timestamp_s"]].drop_duplicates().sort_values("timestamp_s")
    index = min(max(int(frame), 0), len(frames) - 1)
    current = frames.iloc[index]
    rows = source[(source.frame_id == current.frame_id) & (source.status == "observed")]
    figure = go.Figure()
    for team, color in [("home", "#419bf0"), ("away", "#ef704d"), (None, "#f5d742")]:
        selected = rows[rows.team_id == team] if team else rows[rows.entity == "ball"]
        figure.add_scatter(
            x=selected.x_m,
            y=selected.y_m,
            mode="markers+text",
            text=selected.provider_entity_id,
            name=team or "ball",
            marker_color=color,
        )
    figure.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68)
    figure.add_shape(type="line", x0=52.5, x1=52.5, y0=0, y1=68)
    figure.update_layout(
        height=420,
        xaxis={"range": [-3, 108]},
        yaxis={"range": [71, -3], "scaleanchor": "x"},
        title=f"Provider tracking · {current.timestamp_s:.2f}s",
    )
    text = (
        "**Provider tracking for shape review.** Observed positions only; no video is aligned "
        "with this source. Home is blue, away is orange. Model formation suggestions are hidden "
        "during labeling. IDs are provider IDs, not verified names."
    )
    return (
        None,
        text,
        figure,
        rows[["provider_entity_id", "team_id", "x_m", "y_m", "status"]],
        gr.update(value=index, maximum=max(1, len(frames) - 1)),
    )


def view(folder, frame=0):
    if folder and folder.startswith("provider::"):
        return provider_view(folder, frame)
    if not folder:
        return (
            None,
            "Select a completed analysis.",
            go.Figure(),
            pd.DataFrame(),
            gr.update(value=0, maximum=1),
        )
    root = Path(folder)
    report = json.loads((root / "report.json").read_text())
    frames = pd.read_parquet(root / "frames.parquet")
    fid = min(max(int(frame), 0), len(frames) - 1)
    current = frames.iloc[fid]
    state = pd.read_parquet(root / "state.parquet")
    state = state[state.frame_id == fid].copy()
    detections = pd.read_parquet(root / "detections.parquet")
    detections = detections[detections.frame_id == fid]
    figure = go.Figure()
    for _, row in state.iterrows():
        if pd.isna(row.x_m):
            continue
        ball = row.entity == "ball_candidate"
        color = "#f5d742" if ball else {0: "#419bf0", 1: "#ef704d"}.get(row.team_cluster, "#999")
        figure.add_scatter(
            x=[row.x_m],
            y=[row.y_m],
            mode="markers+text",
            text=["ball" if ball else f"T{row.tracklet_id}"],
            showlegend=False,
            marker={
                "color": color,
                "size": 11 if ball else 8,
                "symbol": "circle-open" if row.status in {"estimated", "tentative"} else "circle",
            },
        )
    figure.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68)
    figure.add_shape(type="line", x0=52.5, x1=52.5, y0=0, y1=68)
    figure.update_layout(
        height=420,
        xaxis={"range": [-3, 108]},
        yaxis={"range": [71, -3], "scaleanchor": "x"},
        title=f"{current.timestamp_s:.2f}s · camera {'usable hypothesis' if current.calibration_accepted else 'rejected'}",
    )
    text = (
        f"**{report['frames']} frames · {report['detections']} person detections.** "
        f"Camera accepted {report['accepted_calibration_frames']}/{report['frames']}. "
        f"Ball states: {report['ball_status_counts']}.\n\n"
        f"**This frame:** {current.calibration_reason}; ball {current.ball_status}. "
        "Blue/red are uniform clusters, gray means unassigned; hollow balls are estimates. "
        "Jersey numbers are hypotheses, not verified names."
    )
    fields = [
        c
        for c in [
            "tracklet_id",
            "role_hypothesis",
            "team_cluster",
            "jersey_hypothesis",
            "jersey_status",
            "confidence",
        ]
        if c in detections
    ]
    return (
        str(root / "preview.mp4"),
        text,
        figure,
        detections[fields],
        gr.update(value=fid, maximum=max(1, len(frames) - 1)),
    )


def save_shape_review(
    artifacts, folder, start, end, team, formation, pressing, confidence, notes, reviewer
):
    if not folder or not 0 <= float(start) < float(end):
        raise ValueError("Select an analysis and an increasing source-time interval")
    if not str(reviewer).strip():
        raise ValueError("Enter your name so this review has an author")
    provider = folder.startswith("provider::")
    root = Path(folder.removeprefix("provider::"))
    if provider:
        frames = pd.read_parquet(root / "source-observations.parquet")[
            ["frame_id", "timestamp_s"]
        ].drop_duplicates()
        frames["shot_id"] = 0
        report = {"source_sha256": sha256(root / "source-observations.parquet"), "sample_hz": 1}
    else:
        report = json.loads((root / "report.json").read_text())
        frames = pd.read_parquet(root / "frames.parquet")
    if team not in (
        {"home", "away", "unknown"} if provider else {"cluster 0", "cluster 1", "unknown"}
    ):
        raise ValueError("Use home/away for provider tracking and clusters for video")
    selected_frames = frames[(frames.timestamp_s >= start) & (frames.timestamp_s < end)]
    if selected_frames.empty or (selected_frames.shot_id.nunique() > 1 and team != "unknown"):
        raise ValueError("Review a single camera shot when assigning a team cluster")
    if start < frames.timestamp_s.min() or end > frames.timestamp_s.max() + 1 / report["sample_hz"]:
        raise ValueError("Review interval is outside this clip")
    row = {
        "schema": "analyst-shape-review/v1",
        "id": uuid.uuid4().hex,
        "source_sha256": report["source_sha256"],
        "analysis": folder,
        "source_kind": "provider_tracking" if provider else "video",
        "start_s": float(start),
        "end_s": float(end),
        "team_cluster": team,
        "formation": formation,
        "pressing": pressing,
        "confidence": confidence,
        "notes": notes,
        "reviewer": str(reviewer).strip(),
        "label_status": "human_review_unadjudicated",
        "training_eligible": False,
        "created_at_unix": time.time(),
    }
    target = Path(artifacts) / "shape-reviews"
    target.mkdir(exist_ok=True)
    (target / f"{row['id']}.json").write_text(json.dumps(row, indent=2) + "\n")
    return "Review saved separately. It will not enter training automatically."


def write_review_queue(artifacts):
    tasks = []
    for label, folder in runs(artifacts):
        provider = folder.startswith("provider::")
        root = Path(folder.removeprefix("provider::"))
        if provider:
            source = root / "source-observations.parquet"
            report = {"source_sha256": sha256(source), "sample_hz": 1}
            frames = pd.read_parquet(root / "frame-eligibility.parquet")
            frames = frames[frames.eligible].copy()
            frames["shot_id"] = 0
        else:
            report = json.loads((root / "report.json").read_text())
            frames = pd.read_parquet(root / "frames.parquet")
        for shot, rows in frames.groupby("shot_id"):
            start = float(rows.timestamp_s.min())
            end = min(start + 5, float(rows.timestamp_s.max()) + 1 / report["sample_hz"])
            tasks.append(
                {
                    "source_sha256": report["source_sha256"],
                    "analysis": folder,
                    "label": label,
                    "shot_id": int(shot),
                    "start_s": start,
                    "end_s": end,
                    "formation": None,
                    "pressing": None,
                    "review_status": "pending_human",
                    "instructions": "Review team shape and defensive behavior; use unknown for incomplete visibility.",
                }
            )
    path = Path(artifacts) / "shape-review-queue.json"
    path.write_text(
        json.dumps(
            {
                "schema": "shape-review-queue/v1",
                "tasks": tasks,
                "model_suggestions_are_ground_truth": False,
            },
            indent=2,
        )
        + "\n"
    )
    return str(path)


def build_video_workflow_tab(artifacts):
    artifacts = Path(artifacts).resolve()
    initial = runs(artifacts)
    first = initial[0][1] if initial else None
    with gr.Tab("Video workflow"):
        gr.Markdown(
            "## Analyze a clip and review the pitch replay\n"
            "Run up to ten minutes on your configured Spark. Rejected geometry and "
            "missing observations remain visible. Model outputs are experimental."
        )
        with gr.Accordion("Analyze a new video", open=not initial):
            source = gr.File(label="Video file · up to 500 MB", type="filepath")
            preset = gr.Dropdown(list(PRESETS), value="rf-soccer", label="Detector")
            camera = gr.Dropdown(
                list(CALIBRATIONS), value="pnl", label="Pitch model (guards always enabled)"
            )
            seconds = gr.Number(value=20, minimum=1, maximum=600, label="Seconds to analyze")
            jersey = gr.Checkbox(value=True, label="Read jersey evidence")
            run = gr.Button(
                "Run analysis on Spark",
                variant="primary",
                interactive=(artifacts / "workflow-runtime.json").exists(),
            )
            job = gr.Textbox(label="Persistent job ID")
            progress = gr.Markdown()
            check = gr.Button("Check / retrieve existing job")
        with gr.Row():
            selected = gr.Dropdown(initial, value=first, label="Completed analysis", scale=5)
            refresh = gr.Button("Refresh analyses")
        preview, text, figure, rows, slider_update = view(first)
        player = gr.Video(preview, label="Video and pitch replay")
        status = gr.Markdown(text)
        slider = gr.Slider(
            minimum=0, maximum=slider_update["maximum"], value=0, step=1, label="Frame to inspect"
        )
        radar = gr.Plot(figure)
        table = gr.Dataframe(rows, interactive=False, label="Track evidence")
        with gr.Accordion("Formation and pressing review", open=False):
            gr.Markdown(
                "Review a short interval, not a single frame. Choose Unknown when the camera "
                "omits players or roles are unclear. Saved labels require adjudication before evaluation."
            )
            with gr.Row():
                start = gr.Number(value=0, label="Start (source seconds)")
                end = gr.Number(value=5, label="End (source seconds)")
                team = gr.Dropdown(
                    ["cluster 0", "cluster 1", "home", "away", "unknown"],
                    value="unknown",
                    label="Team",
                )
            formation = gr.Dropdown(
                ["unknown", "4-3-3", "4-4-2", "4-2-3-1", "3-5-2", "other"],
                value="unknown",
                label="Observed shape",
            )
            pressing = gr.Dropdown(
                ["unknown", "high press", "mid block", "low block", "transition"],
                value="unknown",
                label="Defensive behavior",
            )
            confidence = gr.Dropdown(
                ["low", "medium", "high"], value="low", label="Review confidence"
            )
            notes = gr.Textbox(label="Evidence, visibility limits, and notes")
            reviewer = gr.Textbox(label="Your name")
            save = gr.Button("Save my review")
            saved = gr.Markdown()
            queue_file = gr.File(label="Review queue", interactive=False)
            queue_download = gr.Button("Prepare review queue")
        selected.change(
            lambda p: view(p, 0), selected, [player, status, radar, table, slider], queue=False
        )
        slider.release(
            lambda p, f: view(p, f)[1:4], [selected, slider], [status, radar, table], queue=False
        )
        refresh.click(lambda: gr.Dropdown(choices=runs(artifacts)), outputs=selected, queue=False)

        def launch(file, detector, calibration, duration, jnr):
            from soccerviz.vision.video_jobs import inspect, submit

            yield "", "Staging video and starting a durable job…", gr.skip()
            key, current = submit(file, artifacts, detector, calibration, float(duration), jnr)
            yield (
                key,
                f"Job {current['status']}. You can return and check this ID later.",
                gr.skip(),
            )
            for _ in range(240):
                current = inspect(key, artifacts)
                if current["status"] in {"completed", "failed"}:
                    folder = current.get("video_run")
                    yield (
                        key,
                        json.dumps(current),
                        gr.Dropdown(choices=runs(artifacts), value=folder),
                    )
                    return
                time.sleep(3)
            yield key, "Still running. Use Check / retrieve to resume monitoring.", gr.skip()

        def inspect_job(key):
            from soccerviz.vision.video_jobs import inspect

            current = inspect(key, artifacts)
            return json.dumps(current), gr.Dropdown(
                choices=runs(artifacts), value=current.get("video_run")
            )

        run.click(
            launch,
            [source, preset, camera, seconds, jersey],
            [job, progress, selected],
            concurrency_limit=1,
            concurrency_id="video-analysis",
        )
        check.click(inspect_job, job, [progress, selected], queue=False)
        save.click(
            lambda *a: save_shape_review(artifacts, *a),
            [selected, start, end, team, formation, pressing, confidence, notes, reviewer],
            saved,
            queue=False,
        )
        queue_download.click(lambda: write_review_queue(artifacts), outputs=queue_file, queue=False)
