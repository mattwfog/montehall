"""Read-only public dataset and benchmark inspector."""

import math

import gradio as gr

from soccerviz.datasets.dataset_catalog import DatasetCatalog


def _finite_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric(value, digits=1, percent=False):
    number = _finite_number(value)
    if number is None:
        return "unavailable"
    return f"{number * 100:.{digits}f}%" if percent else f"{number:,.{digits}f}"


def _count(value):
    return _metric(value, digits=0)


def _candidate_summary(report):
    """Describe experimental scope and failure coverage alongside headline scores."""
    schema = report.get("schema")
    is_efpi = (
        report.get("schema_version") == 1
        and report.get("method") == "UnravelSports EFPI public API"
    )
    if (
        schema
        not in {
            "vision-benchmark-results/v1",
            "detector-tracking-comparison/v1",
            "calibration-benchmark-results/v1",
            "jersey-evaluation/v1",
        }
        and not is_efpi
    ):
        return None
    if report.get("status") in ("error", "failed", "blocked", "unavailable"):
        reason = report.get("error", report.get("reason", "See the full report for details."))
        return [
            f"**Experimental run: {report['status']}.** No completed result is available.",
            str(reason),
        ]
    lines = ["**Experimental comparison — these results do not establish production readiness.**"]
    overall = report.get("overall") or {}
    sequences = report.get("by_sequence") or {}
    sequence_count = _count(len(sequences)) if sequences else "unavailable"
    if schema == "vision-benchmark-results/v1":
        thresholds = report.get("thresholds") or {}
        backend = report.get("backend") or {}
        lines.append(
            f"{backend.get('name', 'Detector')}: {_count(overall.get('frames'))} frozen frames "
            f"across {sequence_count} sequences; {_count(report.get('submitted_frames'))} submitted, "
            f"{_count(report.get('missing_prediction_frames'))} missing. Missing frames count as empty predictions."
        )
        classes = overall.get("classes") or {}
        if classes:
            parts = []
            for label in ("person", "ball"):
                if label in classes:
                    values = classes[label] or {}
                    parts.append(
                        f"{label} precision {_metric(values.get('precision'), percent=True)}, "
                        f"recall {_metric(values.get('recall'), percent=True)}"
                    )
            lines.append(
                f"Boxes at score ≥{_metric(thresholds.get('score_threshold'), 2)} and "
                f"IoU ≥{_metric(thresholds.get('iou_threshold'), 2)}: {'; '.join(parts)}."
            )
        centers = overall.get("ball_center") or {}
        lines.append(
            f"Ball centers within {_metric(thresholds.get('ball_center_threshold_px'))} px: "
            f"precision {_metric(centers.get('precision'), percent=True)}, "
            f"recall {_metric(centers.get('recall'), percent=True)} "
            f"({_count(centers.get('tp'))}/{_count(centers.get('gt'))} annotated balls)."
        )
        coco = report.get("coco_bbox") or {}
        if coco.get("available"):
            lines.append(
                f"Box mAP (IoU .50:.95): {_metric(coco.get('map'), 3)}; submitted candidate "
                f"score floor {_metric(coco.get('candidate_score_floor'), 2)}. "
                "Compare AP only at matching candidate score floors."
            )
        elif report.get("prediction_geometry") == "center-only":
            lines.append(
                "Center predictions only: box AP is undefined; no person detection score is measured."
            )
        else:
            lines.append(f"Box AP unavailable: {coco.get('reason', 'See the full report.')}")
        if backend.get("training_domain"):
            lines.append(f"Training scope: {backend['training_domain']}.")
        lines.append(
            "Short development clips, not the full official benchmark; public pretraining overlap is unknown."
        )
    elif schema == "detector-tracking-comparison/v1":
        frame_counts = [_finite_number(row.get("frames")) for row in sequences.values()]
        frames = (
            sum(frame_counts) if frame_counts and all(v is not None for v in frame_counts) else None
        )
        lines += [
            (
                f"Person tracking over {_count(frames)} frames across {sequence_count} sequences. "
                f"HOTA **{_metric(overall.get('HOTA'), 3)}** · IDF1 **{_metric(overall.get('IDF1'), 3)}** (0–1)."
            ),
            (
                f"Detection accuracy (DetA) {_metric(overall.get('DetA'), 3)} · "
                f"association accuracy (AssA) {_metric(overall.get('AssA'), 3)}. "
                f"Tracker: {report.get('tracker', 'unavailable')}."
            ),
            (
                "Short sequence comparison with the same tracker. This does not establish team or jersey identity, "
                "or identity continuity across camera cuts."
            ),
        ]
    elif schema == "calibration-benchmark-results/v1":
        errors = overall.get("accepted_point_error") or {}
        lines += [
            (
                f"Calibration only: {_count(overall.get('accepted_frames'))}/{_count(overall.get('frames'))} "
                f"cameras accepted ({_metric(overall.get('frame_coverage'), percent=True)}); "
                f"{_count(report.get('missing_prediction_frames'))} missing frame outputs."
            ),
            (
                f"Projected {_count(overall.get('projected_points'))}/{_count(overall.get('valid_target_points'))} "
                f"target points. **{_metric(overall.get('all_target_fraction_within_2m'), percent=True)}** "
                "of all valid targets fall within 2 m."
            ),
            (
                f"Error on accepted projections: median **{_metric(errors.get('median_m'), 3)} m**, "
                f"p95 **{_metric(errors.get('p95_m'), 2)} m**, mean **{_metric(errors.get('mean_m'), 2)} m**."
            ),
        ]
        tails = [_finite_number(errors.get(key)) for key in ("p95_m", "mean_m")]
        if any(value is not None and value >= 10 for value in tails):
            lines.append(
                "**Large projection errors remain among accepted cameras. A low median does not make calibration reliable.**"
            )
        if report.get("oracle_image_footpoints"):
            lines.append(
                "Ground-truth image footpoints are used for this calibration diagnostic; this is not end-to-end detection or tracking accuracy."
            )
        if report.get("ground_truth_used_for_fitting") is True:
            lines.append("Ground truth was used for fitting; this is an oracle diagnostic.")
        lines.append(
            "Errors are conditional on accepted projections; judge them together with coverage and all-target success."
        )
    elif schema == "jersey-evaluation/v1":
        values = report.get("summary") or {}
        lines += [
            (
                f"Jersey readings from {_count(values.get('expected_crops', values.get('evaluated_crops')))} "
                f"sampled predicted crops: {_count(values.get('submitted_prediction_crops'))} outputs, "
                f"{_count(values.get('missing_prediction_crops'))} missing; missing outputs count as abstentions."
            ),
            (
                f"**{_count(values.get('accepted_crops'))} accepted** "
                f"({_metric(values.get('acceptance_coverage'), percent=True)} of all crops), "
                f"{_count(values.get('abstained_crops'))} abstained. "
                f"{_count(values.get('matched_number_labeled_crops'))} crops have matched numeric jersey labels."
            ),
            (
                f"Accepted labeled precision **{_metric(values.get('accepted_number_precision'), percent=True)}** "
                f"across {_count(values.get('accepted_labeled_crops'))} accepted labeled readings; "
                f"correct accepted coverage of labeled crops **{_metric(values.get('correct_accepted_number_coverage_of_labeled_crops'), percent=True)}**."
            ),
        ]
        if report.get("records"):
            correct = sum(row.get("correct_number") is True for row in report["records"])
            lines.append(f"{correct} correct accepted readings in the saved crop records.")
        lines.append(
            "Ground-truth numbers do not establish current-frame legibility. Confidence is uncalibrated; "
            "these are single-crop readings, not verified player names or persistent identities."
        )
    else:
        sampled, eligible = report.get("sampled_frames"), report.get("eligible_frames")
        total, kept = _finite_number(sampled), _finite_number(eligible)
        coverage = kept / total if total and kept is not None else None
        lines.append(
            f"EFPI on provider tracking, game {report.get('game', 'unavailable')}: "
            f"{_count(eligible)}/{_count(sampled)} sampled frames eligible "
            f"({_metric(coverage, percent=True)}); {_count(report.get('omitted_frames'))} omitted. "
            f"Requested segment {_count(report.get('duration_s_requested'))} s at "
            f"{_metric(report.get('sample_frequency_hz'))} Hz."
        )
        for team, values in (report.get("frame_efpi") or {}).items():
            baseline = (report.get("baseline_three_template") or {}).get(team) or {}
            lines.append(
                f"{team.title()}: dominant template **{values.get('dominant_formation', 'unavailable')}** "
                f"in {_metric(values.get('dominant_share'), percent=True)} of eligible frames; "
                f"adjacent same-phase stability {_metric(values.get('same_phase_adjacent_stability'), percent=True)} "
                f"(existing three-template baseline {_metric(baseline.get('same_phase_adjacent_stability'), percent=True)})."
            )
        lines.append(
            "Descriptive template matches, not analyst-validated formations. Stability is not accuracy. "
            "Goalkeepers and direction are geometry proxies; provider-event possession gaps and incomplete observations are omitted without interpolation."
        )
    return lines


def experiment_summary(record):
    if not record or "report" not in record:
        return "Register a completed dataset or benchmark report to see results here."
    report = record["report"]
    lines = [f"### {record['name']}", f"Split: **{record['split']}**."]
    if not isinstance(report, dict):
        return "\n\n".join(lines + ["Report unavailable or incomplete; inspect the full record."])
    candidate = _candidate_summary(report)
    if candidate is not None:
        return "\n\n".join(lines + candidate)
    if "test_metrics" in report:
        lines += ["", "| Target | Model Brier ↓ | Prevalence baseline ↓ |", "| --- | ---: | ---: |"]
        for target, metric in report["test_metrics"].items():
            lines.append(
                f"| {target} | {metric['model']['brier']:.6f} | {metric['train_prevalence_baseline']['brier']:.6f} |"
            )
        lines += [
            "",
            "Current-action probability baseline. Match-disjoint evaluation does not establish coaching value.",
        ]
    metric = report.get("image", report.get("metrics"))
    if metric and "HOTA" in metric:
        lines += [
            "",
            f"HOTA **{metric['HOTA']:.3f}** · IDF1 **{metric['IDF1']:.3f}** · {metric['frames']} scored frames.",
        ]
        if report.get("oracle_boxes"):
            lines.append(
                "**Association-only diagnostic using ground-truth boxes. This does not measure detector accuracy.**"
            )
        if report.get("gs_hota_unavailable_reason"):
            lines.append(
                f"GS-HOTA unavailable: {report['missing_prediction_pitch']} predictions lack metric pitch geometry."
            )
    if report.get("schema") == "skillcorner-ingestion/v1":
        runs = (
            report.get("descriptive_summaries", {})
            .get("event_type_counts", {})
            .get("off_ball_run", 0)
        )
        lines += [
            "",
            f"**{report['events']:,}** published events · **{runs:,}** off-ball runs · **{report['phases']:,}** phase intervals.",
            f"{report['events_with_complete_tracking_interval']} events have complete coverage in the downloaded tracking sample.",
            "These are provider-derived observations and labels, not independent perception ground truth.",
        ]
    return "\n\n".join(lines) if not "test_metrics" in report else "\n".join(lines)


def build_datasets_tab(artifacts):
    catalog = DatasetCatalog(artifacts / "harness")

    def choices():
        return [(f"{r['name']} · {r['split']} · {r['id'][:8]}", r["id"]) for r in catalog.list()]

    def view(key):
        if not key:
            return {"status": "No dataset experiment registered yet"}
        return catalog.get(key)

    initial = choices()
    first = initial[0][1] if initial else None
    with gr.Tab("Datasets & benchmarks"):
        gr.Markdown(
            "## Public data and measured results\n"
            "Inspect source provenance, dataset splits, benchmark scores, and training reports. "
            "Provider estimates and annotated ground truth retain their original meaning."
        )
        with gr.Row():
            selected = gr.Dropdown(initial, value=first, label="Dataset experiment", scale=5)
            refresh = gr.Button("Refresh experiments", scale=1)
        summary = gr.Markdown(experiment_summary(view(first)))
        with gr.Accordion("Source provenance and full report", open=False):
            report = gr.JSON(view(first), label="Immutable experiment record")
        gr.Markdown(
            "Reports are stored with the shared harness. Large artifacts remain local files; verify their hashes to check they still match the registered experiment."
        )
        verify = gr.Button("Verify local artifacts")
        integrity = gr.JSON(label="File integrity")
        selected.change(view, selected, report, queue=False)
        selected.change(lambda key: experiment_summary(view(key)), selected, summary, queue=False)
        refresh.click(lambda: gr.Dropdown(choices=choices(), value=None), outputs=selected, queue=False)
        verify.click(
            lambda key: catalog.verify(key) if key else {"status": "Select an experiment"},
            selected,
            integrity,
        )
