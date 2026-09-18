"""Local reference-tracking replay, baseline predictions, and analyst review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from soccerviz.core.analysis import FEATURE_LABELS, FEATURES, TARGET, control_grid
from soccerviz.core.data import load_match
from soccerviz.core.reviews import read_reviews, save_review


def pitch_figure(match, index: int, team: int, start_s: float, show_space: bool):
    fig = go.Figure()
    if show_space:
        xs, ys, z = control_grid(
            match.xy[index, match.teams == team], match.xy[index, match.teams != team]
        )
        fig.add_trace(
            go.Heatmap(
                x=xs,
                y=ys,
                z=z,
                zmin=0,
                zmax=1,
                opacity=0.25,
                colorscale="RdBu",
                reversescale=team == 1,
                showscale=False,
                hoverinfo="skip",
            )
        )
    for side, color in [(0, "#4285ff"), (1, "#ff715b")]:
        mask = match.teams == side
        xy = match.xy[index, mask]
        labels = [
            p.split(":")[-1].replace("Player", "")
            for p, keep in zip(match.players, mask, strict=True)
            if keep
        ]
        fig.add_trace(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker={"size": 12, "color": color},
                name="Home" if side == 0 else "Away",
            )
        )
    start = np.searchsorted(match.times, max(start_s, match.times[index] - 5))
    trail = match.ball[start : index + 1]
    fig.add_trace(
        go.Scatter(
            x=trail[:, 0],
            y=trail[:, 1],
            mode="lines",
            line={"color": "#ffffff", "width": 2},
            name="Ball: past 5s",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[match.ball[index, 0]],
            y=[match.ball[index, 1]],
            mode="markers",
            marker={"color": "white", "size": 9, "line": {"color": "black", "width": 1}},
            name="Ball",
        )
    )
    line = {"color": "#d5e5d8", "width": 1}
    for x0, y0, x1, y1 in [
        (0, 0, 105, 68),
        (0, 13.84, 16.5, 54.16),
        (88.5, 13.84, 105, 54.16),
        (0, 24.84, 5.5, 43.16),
        (99.5, 24.84, 105, 43.16),
    ]:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=line)
    fig.add_shape(type="line", x0=52.5, x1=52.5, y0=0, y1=68, line=line)
    fig.add_shape(type="circle", x0=43.35, x1=61.65, y0=24.85, y1=43.15, line=line)
    arrow = "→" if match.direction[index, team] == 1 else "←"
    fig.update_layout(
        title=f"{match.times[index]:.2f}s · {'Home' if team == 0 else 'Away'} attack {arrow}",
        height=490,
        plot_bgcolor="#173f35",
        paper_bgcolor="#173f35",
        font_color="white",
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        legend={"orientation": "h"},
        uirevision=f"game-{match.game}",
        xaxis={"range": [-3, 108], "visible": False},
        yaxis={"range": [71, -3], "visible": False, "scaleanchor": "x", "scaleratio": 1},
    )
    return fig


def shap_figure(row):
    contributions = sorted(
        [(FEATURE_LABELS[f], float(row[f"shap_{f}"]) * 100) for f in FEATURES],
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            x=["Training background"] + [x[0] for x in contributions] + ["Prediction"],
            y=[float(row.base_probability) * 100] + [x[1] for x in contributions] + [0],
            measure=["absolute"] + ["relative"] * len(contributions) + ["total"],
            increasing={"marker_color": "#3083dc"},
            decreasing={"marker_color": "#de7254"},
        )
    )
    fig.update_layout(
        title="SHAP · contributions in probability percentage points",
        yaxis_title="Probability (%)",
        height=480,
        margin={"b": 180},
    )
    return fig


class Workbench:
    def __init__(self, data: Path, artifacts: Path):
        self.matches = {g: load_match(data, g) for g in (1, 2)}
        self.possessions = {
            g: pd.read_parquet(data / "processed" / f"game_{g}" / "possessions.parquet")
            for g in (1, 2)
        }
        self.samples = pd.concat(
            [pd.read_parquet(data / "processed" / f"game_{g}" / "samples.parquet") for g in (1, 2)]
        )
        self.predictions = pd.read_parquet(artifacts / "predictions.parquet")
        self.report = json.loads((artifacts / "baseline-report.json").read_text())
        for split, game_key in [("train", "train_game"), ("eval", "eval_game")]:
            path = data / "processed" / f"game_{self.report[game_key]}" / "samples.parquet"
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.report["input_sha256"][split]:
                raise ValueError(
                    "Processed samples changed since training. Run soccerviz train again."
                )
        self.review_path = artifacts / "reviews.sqlite"
        self.forecast = None
        if (artifacts / "forecast-evaluation.npz").exists():
            with np.load(artifacts / "forecast-evaluation.npz", allow_pickle=False) as z:
                self.forecast = {key: z[key] for key in z.files}
            self.forecast_report = json.loads((artifacts / "forecast-report.json").read_text())
            for split, g in [("train", 1), ("eval", 2)]:
                path = data / "processed" / f"game_{g}" / "tracking.npz"
                if (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    != self.forecast_report["input_sha256"][split]
                ):
                    raise ValueError(
                        "Tracking changed since forecast training. Retrain the forecast."
                    )

    def forecast_plot(self, window):
        index, player = self.forecast["origins"][int(window)]
        match = self.matches[2]
        team = int(match.teams[player])
        fig = pitch_figure(match, int(index), team, float(match.times[index]), False)
        origin = match.xy[index, player]
        history = match.xy[index - 10 : index + 1, player]
        prediction = np.vstack(
            [origin, origin + self.forecast["predicted_displacement"][int(window)]]
        )
        actual = np.vstack([origin, origin + self.forecast["target_displacement"][int(window)]])
        velocity = match.xy[index, player] - match.xy[index - 5, player]
        constant = origin + velocity[None, :] * (np.arange(16) / 5)[:, None]
        for name, path, color, dash in [
            ("Observed history", history, "#ffffff", "solid"),
            ("GRU forecast", prediction, "#f4d35e", "solid"),
            ("Actual future (revealed)", actual, "#b0f2b4", "dash"),
            ("Constant velocity", constant, "#e9b5fa", "dot"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=path[:, 0],
                    y=path[:, 1],
                    mode="lines",
                    name=name,
                    line={"color": color, "width": 3, "dash": dash},
                )
            )
        fig.update_layout(
            title=f"Match 2 · {match.players[player]} · {match.times[index]:.2f}s · 3s forecast"
        )
        return fig

    def choices(self, game):
        rows = self.possessions[int(game)]
        return [
            (
                (
                    f"{r.possession_id} · {'Home' if r.team == 0 else 'Away'} · "
                    f"{r.start_s:.1f}s · {r.duration_s:.1f}s"
                ),
                r.possession_id,
            )
            for r in rows.itertuples()
        ]

    def segment(self, game, key):
        return self.possessions[int(game)].set_index("possession_id").loc[key]

    def render(self, game, key, elapsed, show_space):
        game = int(game)
        segment = self.segment(game, key)
        match = self.matches[game]
        time = min(segment.start_s + float(elapsed), segment.end_s - 0.01)
        index = max(0, int(np.searchsorted(match.times, time, side="right")) - 1)
        pitch = pitch_figure(match, index, int(segment.team), segment.start_s, show_space)
        candidates = self.predictions[
            (self.predictions.possession_id == key)
            & (self.predictions.time_s <= match.times[index] + 1e-6)
        ]
        if candidates.empty:
            return pitch, "No prediction sample at or before this frame.", go.Figure(), ""
        row = candidates.iloc[-1]
        status = (
            f"**{row.probability:.1%} predicted progression** · {row.split} match  \n"
            f"Snapshot **{row.time_s:.2f}s** ({match.times[index] - row.time_s:.1f}s before replay). "
            f"Training background: {row.base_probability:.1%}.  \n"
            f"Recorded outcome: **{'advanced ≥10m' if row[TARGET] else 'did not advance ≥10m'}** "
            "before five seconds or this possession ended. Outcome is revealed for review."
        )
        return pitch, status, shap_figure(row), row.sample_id

    def details(self, game, key):
        segment = self.segment(game, key)
        match = self.matches[int(game)]
        events = match.events[
            (match.events["Start Time [s]"] >= segment.start_s)
            & (match.events["Start Time [s]"] <= segment.end_s)
        ]
        sample = self.samples[self.samples.possession_id == key]
        fig = go.Figure()
        for feature in ["team_width_m", "team_depth_m", "nearest_opponent_m"]:
            fig.add_trace(
                go.Scatter(
                    x=sample.time_s,
                    y=sample[feature],
                    mode="lines+markers",
                    name=FEATURE_LABELS[feature],
                )
            )
        fig.update_layout(
            title="Possession snapshots",
            xaxis_title="Match time (s)",
            yaxis_title="Metres",
            height=280,
        )
        return events[["Team", "Type", "Subtype", "Start Time [s]", "From", "To"]], fig


def build_app(data: Path, artifacts: Path):
    work = Workbench(data, artifacts)
    first = work.choices(2)[0][1]
    segment = work.segment(2, first)
    first_predictions = work.predictions[work.predictions.possession_id == first]
    initial_elapsed = (
        float(first_predictions.iloc[0].time_s - segment.start_s) if len(first_predictions) else 0
    )
    initial = work.render(2, first, initial_elapsed, False)
    events, trends = work.details(2, first)
    report = work.report
    from soccerviz.ui.datasets_ui import build_datasets_tab
    from soccerviz.ui.harness_ui import build_harness_tab
    from soccerviz.ui.research_ui import build_research_tabs
    from soccerviz.ui.video_workflow_ui import build_video_workflow_tab

    with gr.Blocks(title="SoccerViz · football research", analytics_enabled=False) as app:
        gr.Markdown(
            "# SoccerViz · football research\nVideo evidence, tactical models, and managerial analysis in one lab."
        )
        with gr.Tabs():
            build_video_workflow_tab(artifacts)
            build_harness_tab(artifacts)
            build_datasets_tab(artifacts)
            build_research_tabs(work, data, artifacts)
            with gr.Tab("Possession & SHAP"):
                gr.Markdown(
                    "# SoccerViz · possession lab\n"
                    "Reference tracking → spatial features → progression prediction → analyst review. "
                    "[Metrica Sports sample data](https://github.com/metrica-sports/sample-data), "
                    "downsampled to 5 Hz. Player numbers are anonymous provider IDs."
                )
                gr.Markdown(
                    f"**Evaluation: match {report['eval_game']}** · trained on match "
                    f"{report['train_game']} · Brier **{report['random_forest']['brier']:.3f}** "
                    f"vs prior **{report['training_prior_baseline']['brier']:.3f}** (lower is better). "
                    "One evaluation match is a development check, not evidence of elite performance."
                )
                with gr.Row():
                    game = gr.Dropdown([1, 2], value=2, label="Match")
                    possession = gr.Dropdown(
                        work.choices(2), value=first, label="Event-derived possession"
                    )
                with gr.Row():
                    elapsed = gr.Slider(
                        0,
                        round(max(0.2, float(segment.duration_s) - 0.01), 2),
                        value=initial_elapsed,
                        step=0.01,
                        label="Seconds into possession",
                    )
                    play = gr.Checkbox(False, label="Play replay")
                    space = gr.Checkbox(False, label="Show nearest-distance space proxy")
                timer = gr.Timer(0.5, active=False)
                pitch = gr.Plot(initial[0])
                status = gr.Markdown(initial[1])
                explanation = gr.Plot(initial[2])
                sample_id = gr.Textbox(
                    initial[3], label="Prediction sample being reviewed", interactive=False
                )
                gr.Markdown(
                    "SHAP explains this model's probability relative to training examples. "
                    "It does not establish causal player value. Possessions come from provider events; "
                    "the heatmap is a distance proxy, not calibrated pitch control."
                )
                with gr.Accordion("Possession events and spatial changes", open=False):
                    event_table = gr.Dataframe(events, interactive=False)
                    trend_plot = gr.Plot(trends)
                with gr.Accordion("Review the recorded progression label", open=False):
                    verdict = gr.Radio(
                        ["Correct", "Incorrect", "Uncertain"],
                        value="Uncertain",
                        label="Label review",
                    )
                    note = gr.Textbox(label="Analyst note", lines=3)
                    save = gr.Button("Save review")
                    saved = gr.Markdown()
                    review_table = gr.Dataframe(read_reviews(work.review_path), interactive=False)
                    gr.Markdown(
                        "Reviews are append-only and stay separate from model training and evaluation."
                    )
                if work.forecast is not None:
                    with gr.Accordion(
                        "Spark trajectory experiment · inspect player forecasts", open=False
                    ):
                        fr = work.forecast_report
                        gr.Markdown(
                            f"Two seconds of history → three seconds of player movement. "
                            f"Trained on match 1 using **{fr['hardware']}**. Match 2 average "
                            f"displacement error: GRU **{fr['gru']['ade_m']:.2f} m**, constant "
                            f"velocity **{fr['constant_velocity']['ade_m']:.2f} m**, stationary "
                            f"**{fr['stationary']['ade_m']:.2f} m**. These are deterministic "
                            "forecasts on complete tracks, not counterfactual tactics."
                        )
                        window = gr.Slider(
                            0,
                            len(work.forecast["origins"]) - 1,
                            value=0,
                            step=1,
                            label="Evaluation player window",
                        )
                        forecast_plot = gr.Plot(work.forecast_plot(0))
                        window.release(
                            work.forecast_plot, window, forecast_plot, api_name="forecast"
                        )

                def change_game(g):
                    choices = work.choices(g)
                    return gr.Dropdown(choices=choices, value=choices[0][1])

                def change_possession(g, key):
                    seg = work.segment(g, key)
                    event_rows, feature_plot = work.details(g, key)
                    return (
                        gr.Slider(
                            maximum=round(max(0.2, float(seg.duration_s) - 0.01), 2), value=0
                        ),
                        False,
                        event_rows,
                        feature_plot,
                    )

                def tick(g, key, value):
                    end = float(work.segment(g, key).duration_s) - 0.01
                    nxt = min(float(value) + 0.5, end)
                    return nxt, nxt < end

                def review(sid, label, text):
                    if sid not in set(work.predictions.sample_id):
                        raise gr.Error("Move the replay to a prediction sample first.")
                    message = save_review(work.review_path, sid, label, text)
                    return message, read_reviews(work.review_path)

                game.change(change_game, game, possession).then(
                    change_possession, [game, possession], [elapsed, play, event_table, trend_plot]
                ).then(
                    work.render,
                    [game, possession, elapsed, space],
                    [pitch, status, explanation, sample_id],
                )
                possession.input(
                    change_possession, [game, possession], [elapsed, play, event_table, trend_plot]
                ).then(
                    work.render,
                    [game, possession, elapsed, space],
                    [pitch, status, explanation, sample_id],
                )
                elapsed.change(
                    work.render,
                    [game, possession, elapsed, space],
                    [pitch, status, explanation, sample_id],
                    api_name="replay",
                )
                space.change(
                    work.render,
                    [game, possession, elapsed, space],
                    [pitch, status, explanation, sample_id],
                )
                play.change(lambda active: gr.Timer(active=active), play, timer)
                timer.tick(
                    tick, [game, possession, elapsed], [elapsed, play], show_progress="hidden"
                )
                save.click(
                    review, [sample_id, verdict, note], [saved, review_table], api_name="review"
                )
    return app
