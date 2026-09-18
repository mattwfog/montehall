"""Research tabs connecting perception, uncertainty, tactics, and manager evidence."""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from soccerviz.modeling.manager import build_brief, positioning_scenario
from soccerviz.ui.workbench import pitch_figure


def read_report(path):
    return json.loads(path.read_text()) if path.exists() else None


def build_research_tabs(work, data: Path, artifacts: Path):
    with gr.Tab("Overview"):
        gr.Markdown(
            "## End-to-end research status\n"
            "Each component has its own evidence, benchmark, and limitations. "
            "A working pipeline is the starting point for elite-level validation."
        )
        inventory = read_report(artifacts / "suite-report.json")
        if inventory:
            table = pd.DataFrame(inventory["components"])[["component", "status", "limitation"]]
            gr.Dataframe(table, interactive=False, label="Component inventory", max_height=620)
            gr.Markdown(
                f"**{inventory['components_with_artifacts']}/{inventory['components_total']} "
                "components have run artifacts.** Real-video accuracy, analyst relevance, "
                "and causal managerial benefit remain separate validation tasks."
            )
        else:
            gr.Markdown("Run `soccerviz research` to generate the extended experiment inventory.")
    video = artifacts / "video/demo-v2"
    if (video / "report.json").exists():
        with gr.Tab("Video & identity"):
            report = read_report(video / "report.json")
            gr.Markdown(
                "## Video perception · public Roboflow demo\n"
                f"**{report['frames']} frames**, **{report['detections']} detections**, "
                f"**{report['anonymous_tracklets']} anonymous tracklets**. "
                f"Ball candidates in **{report['frames_with_ball_candidate']} frames**. "
                "Coverage is not accuracy; jersey and player-name assignments can abstain."
            )
            gr.Video(
                str((video / "preview.mp4").resolve()), label="Annotated 20-second development clip"
            )
            frame_table = pd.read_parquet(video / "frames.parquet")
            video_state = pd.read_parquet(video / "state.parquet")

            def radar(index):
                index = int(index)
                rows = video_state[video_state.frame_id == index]
                figure = go.Figure()
                for label, selected, color in [
                    (
                        "Uniform cluster 0",
                        rows[(rows.entity == "player") & (rows.team_cluster == 0)],
                        "#4c8bf5",
                    ),
                    (
                        "Uniform cluster 1",
                        rows[(rows.entity == "player") & (rows.team_cluster == 1)],
                        "#ee7866",
                    ),
                    (
                        "Team unassigned",
                        rows[(rows.entity == "player") & rows.team_cluster.isna()],
                        "#888888",
                    ),
                    ("Ball hypothesis", rows[rows.entity == "ball_candidate"], "#111111"),
                ]:
                    figure.add_trace(
                        go.Scatter(
                            x=selected.x_m,
                            y=selected.y_m,
                            mode="markers+text",
                            text=selected.tracklet_id.astype(str),
                            textposition="top center",
                            marker={"color": color, "size": 10},
                            name=label,
                        )
                    )
                figure.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68)
                figure.add_shape(type="line", x0=52.5, x1=52.5, y0=0, y1=68)
                figure.update_layout(
                    title=f"{frame_table.iloc[index].timestamp_s:.2f}s · projected hypotheses · assumed 105×68m pitch",
                    height=420,
                    xaxis={"range": [-3, 108], "constrain": "domain"},
                    yaxis={"range": [71, -3], "scaleanchor": "x", "constrain": "domain"},
                    legend={"orientation": "h"},
                )
                return figure, rows[
                    ["tracklet_id", "entity", "team_cluster", "x_m", "y_m", "status"]
                ]

            initial_radar, initial_state = radar(0)
            index = gr.Slider(0, report["frames"] - 1, value=0, step=1, label="Decoded frame index")
            plot = gr.Plot(initial_radar)
            table = gr.Dataframe(initial_state, interactive=False, max_height=220)
            index.release(radar, index, [plot, table], api_name="video_radar")
            with gr.Accordion("Calibration, identity, and annotation evidence", open=False):
                gr.JSON(report, label="Inference report")
                jersey = read_report(artifacts / "jersey/report.json")
                if jersey:
                    gr.Markdown(
                        f"Jersey OCR: **{jersey['crops']} shirt crops**, "
                        f"**{jersey['crops_with_digit_candidate']} digit candidates**, "
                        f"**{jersey['provisional_jersey_hypotheses']} supported jersey hypotheses**. "
                        "Named-player identity requires a roster and stronger evidence."
                    )
                gr.JSON(
                    read_report(video / "bridge-report.json"), label="Video-to-tactics availability"
                )
                if (video / "annotation-task.json").exists():
                    gr.File(
                        str((video / "annotation-task.json").resolve()),
                        label="Annotation task · suggestions are unreviewed",
                        interactive=False,
                    )
    tactics = artifacts / "tactics"
    if (tactics / "report.json").exists():
        samples = work.samples[work.samples.game == 2].reset_index(drop=True)
        choices = [
            (f"{r.sample_id} · {r.time_s:.1f}s · {'Home' if r.team == 0 else 'Away'}", r.sample_id)
            for r in samples.itertuples()
        ]
        initial_key = choices[len(choices) // 2][1]
        snapshots = pd.read_parquet(tactics / "snapshots.parquet")
        options = pd.read_parquet(tactics / "pass-options.parquet")
        neighbors = pd.read_parquet(tactics / "retrieval.parquet")
        next_events = pd.read_parquet(tactics / "next-event-predictions.parquet")

        def tactical_view(key):
            sample = samples[samples.sample_id == key].iloc[0]
            row = snapshots[snapshots.sample_id == key].iloc[0]
            choices_for_sample = options[options.sample_id == key]
            matches = neighbors[neighbors.query_sample == key]
            forecast = next_events[next_events.sample_id == key].iloc[0]
            match = work.matches[2]
            index, team = int(sample["index"]), int(sample.team)
            figure = pitch_figure(match, index, team, float(sample.time_s), False)
            for i, candidate in enumerate(choices_for_sample.itertuples()):
                figure.add_trace(
                    go.Scatter(
                        x=[match.ball[index, 0], candidate.x_m],
                        y=[match.ball[index, 1], candidate.y_m],
                        mode="lines",
                        name=f"Option {i + 1}: {candidate.receiver}",
                        line={"width": 3, "dash": "dash"},
                    )
                )
            text = (
                f"**{row.phase_proxy}** · template **{row.shape_template}** · "
                f"width **{row.width_m:.1f}m**, depth **{row.depth_m:.1f}m**. "
                f"Pressure proxy: **{'opponent within 3m' if row.pressure_proxy else 'opponent farther than 3m'}**.  \n"
                f"Experimental next action: **{forecast.predicted_next_action}**; recorded: **{forecast.observed_next_action}**. "
                f"Shot-within-10s surrogate: **{forecast.shot_probability:.1%}**."
            )
            return (
                text,
                figure,
                choices_for_sample[
                    [
                        "receiver",
                        "distance_m",
                        "forward_gain_m",
                        "choice_score",
                        "simulated_open_lane_fraction",
                    ]
                ],
                matches[
                    [
                        "neighbor_sample",
                        "neighbor_possession",
                        "neighbor_time_s",
                        "standardized_distance",
                    ]
                ],
            )

        with gr.Tab("Tactics & passes"):
            report = read_report(tactics / "report.json")
            gr.Markdown(
                "## Tactical snapshots and pass options\n"
                f"Pass-recipient top-1: **{report['pass_choice']['model']['top1']:.1%}** "
                f"vs nearest teammate **{report['pass_choice']['nearest_teammate']['top1']:.1%}**. "
                "This imitates completed-pass choices; it does not establish the optimal pass. "
                "The next-action model is below the majority baseline in accuracy."
            )
            sample_picker = gr.Dropdown(
                choices, value=initial_key, label="Evaluation tactical snapshot"
            )
            initial = tactical_view(initial_key)
            description = gr.Markdown(initial[0])
            option_plot = gr.Plot(initial[1])
            option_table = gr.Dataframe(initial[2], interactive=False, label="Candidate options")
            gr.Markdown(
                "Open-lane frequency uses assumed ball/defender speeds and stationary receivers. "
                "It is a simulation diagnostic, not calibrated pass success."
            )
            neighbor_table = gr.Dataframe(
                initial[3], interactive=False, label="Similar snapshots from training match 1"
            )
            sample_picker.change(
                tactical_view,
                sample_picker,
                [description, option_plot, option_table, neighbor_table],
                api_name="tactical_snapshot",
            )
            with gr.Accordion("Set pieces and measured model results", open=False):
                gr.Dataframe(
                    pd.read_parquet(tactics / "set-pieces.parquet"),
                    interactive=False,
                    max_height=300,
                )
                gr.JSON(report)
        with gr.Tab("Manager scenarios"):
            gr.Markdown(
                "## Evidence-linked analyst brief\n"
                "Explore static positioning changes. Opponent responses and causal match effects are not modeled. "
                "Lineup or substitution advice abstains until availability and reviewed player-role evidence are supplied."
            )
            manager_sample = gr.Dropdown(choices, value=initial_key, label="Brief snapshot")
            brief = gr.Markdown(build_brief(data, artifacts, initial_key))

            def player_choices(key):
                sample = samples[samples.sample_id == key].iloc[0]
                match = work.matches[2]
                eligible = [
                    p
                    for p, team, xy in zip(
                        match.players, match.teams, match.xy[int(sample["index"])], strict=True
                    )
                    if team == sample.team and np.isfinite(xy).all()
                ]
                return gr.Dropdown(choices=eligible, value=eligible[0])

            player = player_choices(initial_key)
            player.label = "Observed player to reposition"
            with gr.Row():
                dx = gr.Slider(-10, 10, value=0, step=0.5, label="Change in pitch x (metres)")
                dy = gr.Slider(-10, 10, value=0, step=0.5, label="Change in pitch y (metres)")
            apply = gr.Button("Inspect geometric change")
            differences = gr.Dataframe(
                headers=["feature", "observed", "scenario", "change"], interactive=False
            )

            def scenario(key, person, x, y):
                sample = samples[samples.sample_id == key].iloc[0]
                try:
                    return positioning_scenario(
                        work.matches[2], int(sample["index"]), int(sample.team), person, x, y
                    )
                except ValueError as exc:
                    raise gr.Error(str(exc)) from exc

            manager_sample.change(
                lambda key: build_brief(data, artifacts, key), manager_sample, brief
            ).then(player_choices, manager_sample, player)
            apply.click(
                scenario,
                [manager_sample, player, dx, dy],
                differences,
                api_name="positioning_scenario",
            )
            gr.File(
                str((artifacts / "manager/roster-context-template.json").resolve()),
                label="Roster context template",
                interactive=False,
            )
    if (artifacts / "probabilistic/report.json").exists():
        with gr.Tab("Forecast uncertainty"):
            report = read_report(artifacts / "probabilistic/report.json")
            with np.load(artifacts / "probabilistic/evaluation.npz", allow_pickle=False) as z:
                forecast = {key: z[key] for key in z.files}
            gr.Markdown(
                "## Player forecast with empirical uncertainty\n"
                f"Nominal 90% regions cover **{report['calibrated_90_point_coverage']:.1%}** "
                f"of evaluation points and **{report['calibrated_90_endpoint_coverage']:.1%}** "
                "of three-second endpoints. Fit: game 1 first half; calibration: second half; evaluation: game 2."
            )

            def probabilistic_plot(window):
                window = int(window)
                index, player = forecast["origins"][window]
                match = work.matches[2]
                origin = match.xy[index, player]
                mean = origin + forecast["predicted_displacement"][window]
                actual = origin + forecast["target_displacement"][window]
                radius = float(forecast["empirical_90_radius_m"][window, -1])
                figure = pitch_figure(
                    match, int(index), int(match.teams[player]), float(match.times[index]), False
                )
                figure.add_trace(
                    go.Scatter(
                        x=mean[:, 0],
                        y=mean[:, 1],
                        mode="lines",
                        name="Gaussian mean",
                        line={"color": "#ffd166", "width": 3},
                    )
                )
                figure.add_trace(
                    go.Scatter(
                        x=actual[:, 0],
                        y=actual[:, 1],
                        mode="lines",
                        name="Recorded future",
                        line={"color": "#a8f0bc", "dash": "dash"},
                    )
                )
                figure.add_shape(
                    type="circle",
                    x0=mean[-1, 0] - radius,
                    x1=mean[-1, 0] + radius,
                    y0=mean[-1, 1] - radius,
                    y1=mean[-1, 1] + radius,
                    fillcolor="rgba(255,209,102,0.2)",
                    line={"color": "#ffd166"},
                )
                figure.update_layout(
                    title=f"{match.players[player]} · {match.times[index]:.2f}s · endpoint radius {radius:.1f}m"
                )
                return figure

            window = gr.Slider(
                0,
                len(forecast["origins"]) - 1,
                value=1050,
                step=1,
                label="Probabilistic evaluation window",
            )
            plot = gr.Plot(probabilistic_plot(1050))
            window.release(probabilistic_plot, window, plot, api_name="probabilistic_forecast")
            gr.JSON(report, label="Uncertainty calibration report")
    if (artifacts / "ball/report.json").exists():
        with gr.Tab("Ball & actor"):
            report = read_report(artifacts / "ball/report.json")
            actor = read_report(artifacts / "actor/report.json")
            gr.Markdown(
                "## Ball motion and event attribution\n"
                f"One-second ball forecast: GRU average error **{report['gru']['ade_m']:.2f}m**, "
                f"constant velocity **{report['constant_velocity']['ade_m']:.2f}m**. "
                "The learned model does not yet beat constant velocity on average error."
            )
            with np.load(artifacts / "ball/evaluation.npz", allow_pickle=False) as z:
                ball_predictions = {k: z[k] for k in z.files}

            def ball_plot(window):
                window = int(window)
                index = int(ball_predictions["origins"][window])
                match = work.matches[2]
                origin = match.ball[index]
                predicted = origin + ball_predictions["predicted_displacement"][window]
                actual = origin + ball_predictions["target_displacement"][window]
                constant = (
                    origin
                    + ((match.ball[index] - match.ball[index - 2]) / 0.4)[None]
                    * (np.arange(1, 6) / 5)[:, None]
                )
                fig = pitch_figure(match, index, 0, float(match.times[index] - 1), False)
                for label, points, color in [
                    ("Ball GRU", predicted, "#f4cf58"),
                    ("Recorded ball future", actual, "#abefba"),
                    ("Constant velocity", constant, "#deaded"),
                ]:
                    points = np.vstack([origin, points])
                    fig.add_trace(
                        go.Scatter(
                            x=points[:, 0],
                            y=points[:, 1],
                            mode="lines+markers",
                            name=label,
                            line={"color": color, "width": 3},
                        )
                    )
                fig.update_layout(title=f"Match 2 · {match.times[index]:.2f}s · 1s ball forecast")
                return fig

            window = gr.Slider(
                0,
                len(ball_predictions["origins"]) - 1,
                value=50,
                step=1,
                label="Ball evaluation window",
            )
            plot = gr.Plot(ball_plot(50))
            window.release(ball_plot, window, plot, api_name="ball_forecast")
            if actor:
                gr.Markdown(
                    f"Event actor: **{actor['accuracy_when_attributed']:.1%}** accuracy "
                    f"at **{actor['event_coverage']:.1%}** event coverage. "
                    "Measured against provider pass/recovery/shot actors. This is not continuous possession truth."
                )
                gr.JSON(actor, label="Actor attribution report")
            gr.JSON(report, label="Ball forecast report")
    if (artifacts / "uncertainty/report.json").exists():
        with gr.Tab("Missing players"):
            report = read_report(artifacts / "uncertainty/report.json")
            missing_state = pd.read_parquet(artifacts / "uncertainty/reconstructed-state.parquet")
            frames = sorted(missing_state["index"].unique())
            gr.Markdown(
                "## Simulated broadcast visibility\n"
                f"Hidden-player error: velocity **{report['velocity_mae_m']:.2f}m**, "
                f"last seen **{report['last_seen_mae_m']:.2f}m**. "
                f"Only **{report['evaluation_coverage']['hidden_coverage_within_3s_ttl']:.1%}** "
                "of hidden observations are eligible for estimation. Estimates expire after three seconds."
            )

            def missing_plot(value):
                index = int(frames[int(value)])
                rows = missing_state[missing_state["index"] == index]
                match = work.matches[2]
                figure = go.Figure()
                # Truth is shown explicitly as evaluation-only reference.
                figure.add_trace(
                    go.Scatter(
                        x=match.xy[index, :, 0],
                        y=match.xy[index, :, 1],
                        mode="markers",
                        marker={"symbol": "circle-open", "size": 13, "color": "#555555"},
                        name="Provider truth (evaluation only)",
                    )
                )
                for label, marker, color in [
                    ("observed", "circle", "#4788df"),
                    ("estimated", "diamond", "#e89535"),
                ]:
                    points = rows[rows.status == label]
                    figure.add_trace(
                        go.Scatter(
                            x=points.x_m,
                            y=points.y_m,
                            mode="markers",
                            name=label,
                            marker={"symbol": marker, "color": color, "size": 10},
                        )
                    )
                figure.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68)
                figure.update_layout(
                    title=f"{match.times[index]:.2f}s · observed / estimated / unavailable",
                    height=440,
                    xaxis={"range": [-3, 108]},
                    yaxis={"range": [71, -3], "scaleanchor": "x"},
                )
                return figure, rows[["player_id", "status", "age_s", "empirical_90_radius_m"]]

            slider = gr.Slider(
                0, len(frames) - 1, value=10, step=1, label="Visibility benchmark snapshot"
            )
            initial = missing_plot(10)
            plot = gr.Plot(initial[0])
            status = gr.Dataframe(initial[1], interactive=False, max_height=250)
            slider.release(missing_plot, slider, [plot, status], api_name="missing_players")
            gr.JSON(report, label="Occlusion benchmark report")
