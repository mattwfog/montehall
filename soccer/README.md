# SoccerViz

A soccer computer vision and modeling project aimed at tactical analysis and
managerial decision support.

The starting reference is the basketball CV work in `../basketball`, especially its distinction
between visual observations, reconstructed game state, and a learned model over
that state. The soccer project extends that ambition toward team behavior and
coaching decisions.

## Durable specialist harness

Start with [STATUS](docs/STATUS.md). The [unified system guide](docs/guides/unified-soccer-system.md) describes the executable
soccer colony: video specialists or provider tracking → shared state → independent
checks → tactical findings → review. It includes the Montehall reuse map and the
remaining model work. The [harness guide](docs/guides/harness.md) covers persistence,
pause/resume, corrections and replay of earlier revisions.

```sh
uv run soccerviz harness start --video-run artifacts/video/demo-v2 --start-s 10 --end-s 20
```

Use the printed run ID with `soccerviz harness inspect` and `soccerviz harness export`.

The [integration guide](docs/guides/integrations.md) covers the new **Runs & review**
workbench and runnable CVAT, TrackEval, TrackLab, Kloppy, socceraction, and Prefect
adapters. Install their isolated workers with
`.venv/bin/python scripts/setup/setup_integrations.py all` after the core environment.

The [public-data guide](docs/guides/public-data.md) covers actual SoccerNet benchmarks,
30-match StatsBomb training, two SkillCorner imports, and the new
**Datasets & benchmarks** tab with shared experiment records.

## Intended questions

The kinds of question the project is built toward:

- How does a team build up against different pressing structures?
- What triggers its press, and where does the opponent escape it?
- How do its defensive shape, spacing, and protection behind the attack change
  across phases and game situations?
- Which off-ball runs create space or expose a defensive weakness?
- Which player roles, lineup changes, and substitutions fit a particular opponent?
- What alternative positioning or movement is worth a coach investigating,
  and which comparable match sequences support that suggestion?

## Current scaffold: version 0.2

The end-to-end research scaffold now has **14 component groups with initial run
artifacts**, including actual video inference on Spark, calibrated geometry,
anonymous tracking/team classification, jersey OCR evidence, missing-player
estimation, player and ball forecasts, event attribution, tactical models,
pass alternatives, similar-sequence retrieval, and managerial scenarios.

Read the [complete results and reproduction guide](docs/experiments/scaffold-and-results.md).
The workbench provides eleven tabs, an explicit component inventory, and the
original possession lab. Core and isolated upstream integration suites exercise
the evidence, review, evaluation, and remote-execution boundaries.

New measured results include pass-recipient top-1 **45.2% vs 33.4%**, simulated
hidden-player MAE **0.95m vs 2.80m**, and event-actor accuracy **91.0% at 77.5%
coverage**. Some baselines are weak: next-action accuracy is below majority,
ball forecasting does not beat constant velocity on mean error, and empirical
90% player forecast regions cover 88.4% of evaluation points. These outcomes are
recorded alongside the stronger results.

## Possession lab foundation

A local possession lab runs on two matches of
[Metrica Sports sample tracking and event data](https://github.com/metrica-sports/sample-data).
It provides pitch replay, an event-derived possession selector, spatial feature
trends, a progression classifier with SHAP explanations, persistent analyst
reviews, and an inspector for player trajectories trained on the DGX Spark.

Video detection, tracklet association, calibration, and reconstruction now have
runnable baselines. Named-player identity, independently validated tactical
recognition, and causal managerial recommendations remain unresolved. Provider
observations, video hypotheses, and off-screen estimates remain distinct.

## Run locally

Python 3.12 and [uv](https://docs.astral.sh/uv/) are used for the locked environment.

```sh
uv sync --locked --python 3.12
uv run soccerviz fetch --games 1 2
uv run soccerviz prepare --games 1 2
uv run soccerviz analyze --games 1 2
uv run soccerviz train
uv run soccerviz workbench --port 7865
```

Open **http://127.0.0.1:7865**. The workbench binds to loopback and does not create
a public Gradio share link. For an already prepared checkout, only the final
command is needed. On restricted environments, fetching data and binding the
local server require network permission.

Choose a match and possession, scrub the replay or press Play, and inspect the
most recent prediction snapshot at or before the replay time. The model asks
whether the ball advances at least 10 metres within five seconds, before the
event-derived possession ends. Recorded outcomes are explicitly revealed for
review. The space overlay is a nearest-distance proxy, not calibrated pitch
control. Reviews of the recorded target are appended to `artifacts/reviews.sqlite`;
they do not overwrite labels or enter training automatically.

## Train trajectories on Spark

This assumes an `ssh spark` alias to a DGX Spark with the NVIDIA PyTorch image cached. After preparing the two matches:

```sh
python3 scripts/spark/spark_forecast.py --epochs 8
```

This transfers only the forecast module and prepared tracking arrays to
`~/soccerviz` on `ssh spark`, runs a separate temporary GPU container using
`nvcr.io/nvidia/pytorch:26.01-py3`, and retrieves the checkpoint, report, and
evaluation trajectories. Existing containers and services are not modified.
Restart the workbench after retrieval to enable the forecast inspector.
GPU training can also run in a suitable local Torch environment with
`soccerviz train-forecast --epochs 8 --device cuda`.

To reproduce the extended scaffold after the reference data is prepared:

```sh
uv run --extra vision soccerviz fetch-demo
python3 scripts/spark/spark_video.py --seconds 20 --run-name demo-v2
python3 scripts/spark/spark_sequence_models.py --model probabilistic
python3 scripts/spark/spark_sequence_models.py --model ball
uv run soccerviz jersey --video-run artifacts/video/demo-v2
uv run soccerviz research
uv run soccerviz workbench --port 7865
```

The video run refuses to overwrite a completed directory. Tesseract is required
for the optional OCR experiment.
The current dashboard targets the `demo-v2` video run. See the results guide for
individual commands, data limits, and the annotation workflow.

## Measured development results

Both models train on game 1 and evaluate on game 2. Parameters and epoch count
were fixed before this evaluation; game 3 remains unused.

| Experiment | Model | Reference baseline |
| --- | ---: | ---: |
| Progression Brier score, lower is better | 0.203 | 0.239, training-set prior |
| Progression ROC AUC | 0.730 | 0.500 |
| Player trajectory mean displacement error, 3s horizon | 1.07 m | 1.57 m, constant velocity |
| Player trajectory error at 3s | 2.53 m | 3.49 m, constant velocity |

The classifier uses 579 training and 602 evaluation snapshots. Probability SHAP
uses 100 training rows as background and verifies additivity for every prediction.
The 48-unit player GRU completed eight epochs on Spark's NVIDIA GB10. Its input
is two seconds of observed player history with team and ball context.
Trajectory metrics include complete player tracks throughout the recording,
including stoppages; they are not an in-possession-only benchmark.

One anonymized development match does not establish performance across leagues,
teams, camera types, or tactical contexts. SHAP explains model behavior, not
causal credit. The GRU produces deterministic motion forecasts, not hypothetical
responses to managerial interventions. See the full
[experiment notes](docs/experiments/first-milestone.md) and checked-in JSON reports there.

## Project layout

- `docs/STATUS.md`: the single status page: defaults, measured numbers, candidates, open work.
- `src/soccerviz/core/`: data ingestion, assets, geometry, identity, possession analysis, baseline model.
- `src/soccerviz/vision/`: the default video pipeline, ball, jersey OCR, video safety, observation channels, specialists.
- `src/soccerviz/candidates/`: experimental detectors, detector training, calibration, temporal ball, jersey and formation models, and the frozen vision benchmark.
- `src/soccerviz/harness/`: durable specialist execution, colonies, consistency checks, experiment gates.
- `src/soccerviz/providers/`: provider tracking, annotation, action value and remote execution adapters.
- `src/soccerviz/datasets/`: dataset catalog, SoccerNet, StatsBomb, tracking evaluation.
- `src/soccerviz/modeling/`: tactics, forecasts, uncertainty, probabilistic models, manager scenarios, action training.
- `src/soccerviz/ui/`, `src/soccerviz/cli/`: workbench tabs and command-line entry points.
- `tests/`: mirrors the package tree.
- `scripts/`: `benchmarks/`, `training/`, `spark/`, `setup/`.
- `envs/<worker>/`: requirements and Dockerfile for each isolated worker.
- `docs/guides/`: how the system works. `docs/experiments/`: per-experiment writeups. `docs/reports/`: dated snapshots.
- `results/`: tracked JSON reports the writeups cite.
- `data/`, `artifacts/`: downloaded sources, checkpoints, predictions, reviews; excluded from version control.

```sh
uv run pytest -q
uv run ruff check src tests scripts
```

Read the [Montehall review](docs/reports/montehall-review.md) and
[recent repository survey](docs/reports/repository-survey-2026-09-07.md) for architecture
references and reuse candidates. The next gate is independent video annotation
and a larger game-disjoint evaluation corpus, followed by analyst tactical labels
and real roster/context evidence. The [architecture and evidence contract](docs/guides/architecture.md)
describes the boundaries between those pieces.

## Executed vision candidate comparisons

The [candidate experiment report](docs/reports/candidate-experiments-2026-09-07.md)
records the expanded 450-frame benchmark, RF-DETR and YOLO26 comparisons,
two completed five-epoch soccer fine-tuning pilots, and actual calibration,
temporal-ball, jersey, and formation experiments. Eighteen records are available
in **Datasets & benchmarks**, including coverage and failure diagnostics.
These are experimental workers and saved models; default video extraction has
not been changed on the strength of this small development corpus.

## Data attribution

Sample data is supplied by **Metrica Sports**. The upstream README requests
source acknowledgement for public use. Download manifests pin the upstream
commit and verify Git blob hashes; the original data stays outside source
control. This project does not assign a new license to third-party data or models.
