# Possession lab: first measured experiment

Historical 0.1 milestone. See [the 0.2 scaffold and results](scaffold-and-results.md)
for the implemented video, tactical, uncertainty, and managerial extensions.

Run date: 2026-09-07. These are initial development results, not a claim of
elite tactical intelligence. Architecture references remain in the
[Montehall review](../reports/montehall-review.md) and [repository survey](../reports/repository-survey-2026-09-07.md).

## Data and evidence

Two anonymized matches from [Metrica Sports](https://github.com/metrica-sports/sample-data),
pinned to commit `e706dd506b360d69d9d123d5b8026e7294b13996`. The six raw CSV files
are verified against their Git blob SHA-1 values before preparation. Source
timestamps, player IDs, and missing coordinates are retained. Each 0.2-second
timestamp bucket keeps its first provider observation; there is no interpolation.
The source coordinate convention is mapped to a 105 × 68 metre pitch.

| Property | Game 1 | Game 2 |
| --- | ---: | ---: |
| Sampled frames | 29,002 | 28,232 |
| Registered players, including substitutions | 28 | 26 |
| Frames with observed ball coordinates | 60.9% | 59.0% |
| Event-derived possession segments | 268 | 247 |
| Usable progression snapshots | 579 | 602 |
| Positive progression labels | 239 | 238 |

The missing-ball figure covers the entire recording, including stoppages.
It is not a measurement of detector accuracy. Tracking is supplied by the
provider; SoccerViz has not inferred it from video.

Each half's direction is inferred from team centers in its first five seconds.
The evidence is in each `census.json` and should be checked in replay before
extending to another data source. Training snapshots exclude those initial five
seconds. Home attacks +x then −x in game 1 and −x then +x in game 2.

## Target and features

A control event (pass, recovery, set piece) begins an interval. A same-team loss,
shot, or a stoppage ends it at the terminal event's start; a team change or new
set piece also closes the interval. Challenges alone do not confer possession.
This is a conservative, retrospective event heuristic, not an online possession
estimator. It can exclude terminal ball flights and needs analyst review.

Snapshots begin at least one second after possession starts, spaced five seconds
apart. The binary outcome is whether signed ball x-position gains ≥10 metres
within five seconds, strictly before the possession ends. An observed possession
ending before five seconds ends the opportunity; a recording/period ending
censors the sample. Five-second endpoints count, but possession endpoints do not.
Future ball coverage must be at least 90%; missing futures are never automatically
labeled failures. Residual label uncertainty remains in allowed short gaps.

Fourteen features use positions at the snapshot and ball speed over the preceding
0.6 seconds only. They include ball location, team width/depth, defenders near or
ahead of the ball, attacking support, and an advanced-zone space proxy. Width
and depth use the 10th–90th percentile range of all observed team players,
including the goalkeeper. At least seven observed players per side and a complete
recent ball history are required. The proxy uses a sigmoid of nearest-team
distance differences over a grid; it is not a fitted control probability.

The target rewards spatial progression only. It does not establish pass success,
retention after the interval, chance quality, EPV, player value, or a good coaching
decision. No future event type, end reason, possession duration, player ID, game
ID, or future coordinates enters the feature matrix.

## Progression model and SHAP

Fixed Random Forest: 160 trees, depth 6, minimum leaf 15, max features 0.8,
seed 20260907. Fit on game 1, evaluate on game 2. No random frame split or
hyperparameter search on the evaluation match.

| Metric | Random Forest | Constant training prior |
| --- | ---: | ---: |
| Brier score | 0.202847 | 0.239352 |
| Log loss | 0.592651 | 0.671710 |
| ROC AUC | 0.729903 | 0.500000 |
| Average precision | 0.619229 | 0.395349 |

The within-match possession-cluster bootstrap 95% interval for model-minus-prior
Brier difference is [−0.04967, −0.02295], from 500 draws. This measures variation
among possessions in this match, not uncertainty across unseen matches.
Probability calibration is not established by these metrics alone.

SHAP TreeExplainer uses interventional probability output and a seeded sample of
100 training rows. Baseline probability plus all feature contributions is checked
against every model prediction with tolerance 1e−5. The workbench displays those
contributions in percentage points. Correlated features can share or shift
attribution; explanations are not causal tactical effects.

Machine-readable report: [baseline-report.json](../../results/experiments/baseline-report.json).
Local artifacts also include the fitted model/background and per-snapshot
probabilities with all SHAP contributions.

## Spark trajectory baseline

Separate temporary container on `ssh spark`, NVIDIA GB10, NVIDIA PyTorch
`26.01-py3`. Exact Torch, NumPy, data hashes and training-code hash are recorded in
[forecast-report.json](../../results/experiments/forecast-report.json).

A shared 48-unit GRU takes 11 observations spanning two seconds and predicts
15 future positions spanning three seconds. Inputs are past position relative
to the latest point, past velocity, observed-ball relative position with a
visibility mask, and relative centers of both teams. The head learns a residual
over a constant-velocity path. No future tracking enters the inputs. Training
minimizes squared displacement error for eight fixed epochs.

| Error | GRU | Constant velocity | Stationary |
| --- | ---: | ---: | ---: |
| Mean displacement over horizon | 1.071 m | 1.574 m | 2.643 m |
| Displacement at three seconds | 2.529 m | 3.489 m | 4.793 m |

There are 63,676 training player windows and 62,008 evaluation player windows.
Windows start every two seconds, do not cross periods, and require a fully
observed target player. Overlapping windows and players are correlated; these
counts are not independent trials. The metrics average all eligible windows,
including stoppages, rather than just tactically active phases. Complete-track
selection may make this easier than broadcast footage with occlusions.
This is deterministic player motion only. There is no ball forecast,
uncertainty distribution, role inference, or counterfactual response model.

The workbench can overlay historical movement, GRU prediction, recorded future,
and constant velocity for each evaluation player window. `forecast.pt` contains
weights and run metadata; `forecast-evaluation.npz` preserves origins and paths.

## Review, verification, and next milestone

Analyst label reviews append to a separate SQLite database. They do not mutate
the original sample tables, model artifact, or evaluation labels. A future label
curation pass must explicitly version its reviewed dataset and keep evaluation
isolated. The current UI is local-only and built for one analyst.

Tests cover provider timestamp/ID preservation, orientation invariance, no future
leakage into snapshot features, missing observations, exact horizon and possession
boundaries, disjoint match enforcement, SHAP additivity, persistent reviews, and
forecast window boundaries. Browser verification covers replay, automatic playback,
space overlay, prediction/SHAP rendering, and forecast inspection.

Next: select a short soccer clip with usable rights, label players/ball/pitch
landmarks, and implement a persisted CV adapter inspired by Roboflow sports.
Evaluate detection, tracking identities, pitch calibration error in metres, and
ball coverage separately before passing video-derived state to these models.
Compare tactical features under tracking error and occlusion. Extend to several
game-disjoint matches and a truly untouched test set before model selection,
probabilistic forecasts, calibrated control/possession value, or tactical advice.
