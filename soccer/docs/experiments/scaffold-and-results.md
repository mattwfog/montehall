# End-to-end scaffold and initial results

SoccerViz 0.2, 2026-09-07. The workbench now connects 14 component groups, with
persisted outputs and a report for each. This is a complete first research
scaffold, not a finished elite tactical or managerial model.

Open `http://127.0.0.1:7865`. Tabs cover the component inventory, video and
identity, tactical snapshots and pass choices, manager scenarios, probabilistic
forecasts, ball motion and actors, missing players, and the original possession
replay with SHAP.

## What ran and what it established

| Component | Initial result | Interpretation / missing evidence |
| --- | --- | --- |
| Video ingestion and player detection | 100 PTS-sampled frames from a 20-second public demo; 2,318 person detections | Actual Spark inference; detection precision/recall and mAP remain unmeasured |
| Ball detection | Candidate in 66/100 video frames using four overlapping tiles | Detection presence, not correct-ball coverage |
| Pitch calibration | 100/100 video fits pass internal gates; median leave-one-landmark-out error about 0.42m | Self-consistency under assumed 105×68m geometry, not real-position accuracy |
| Synthetic calibration stress test | 50/50 fits accepted; median held-out projection error 0.105m, p95 0.250m | Simulated camera, 1.5px noise, 6/32 corrupted landmarks |
| Anonymous tracking and team classification | 44 tracklets; color assignment on 83.9% of person detections, including officials in denominator | Appearance-aware association, not long-term or named identity |
| Jersey OCR | 82 shirt crops, 11 digit candidates, zero supported jersey hypotheses | Conservative abstention; independent OCR labels and roster needed |
| Video → tactical features | 27/100 frames support required features; 108 team/orientation scenarios scored | Direction and possession unresolved; zero validated video tactical predictions |
| Event actor attribution | 91.0% correct when assigned; 77.5% event coverage; 70.6% correct over all events | Reference tracking/IDs, measured against 1,236 provider pass/recovery/shot events |
| Missing-player estimation | Velocity MAE 0.948m vs last-seen 2.799m; 90.0% empirical radius coverage | Simulated crop with perfect observed identities; only 21.5% of hidden observations are estimated within 3s expiry |
| Progression classifier + SHAP | Brier 0.203 vs prior 0.239; AUC 0.730; all SHAP sums verified | Spatial progression surrogate, not causal value |
| Deterministic player motion | Three-second ADE 1.071m vs constant velocity 1.574m | Complete player tracks; includes stoppages |
| Probabilistic player motion | NLL 2.396 vs constant-velocity Gaussian 3.019; empirical 90% regions cover 88.4% of points | Undercoverage remains; one Gaussian per player/horizon, no joint multimodal tactics |
| Ball motion | One-second ADE 2.611m vs constant velocity 2.536m; endpoint 4.739m vs 4.917m | Mixed result; learned model does not replace constant velocity on average error |
| Pass-recipient ranking | Top-1 45.2% vs nearest teammate 33.4%; top-3 81.0% vs 73.3% | Imitates recipients of completed passes, not optimal or successful hypothetical passes |
| Next action | Accuracy 56.3% vs majority 78.1%; macro-F1 0.296 vs majority 0.219 | Weak experimental classifier; rare shot class has one evaluation example |
| Shot-within-10s surrogate | Brier 0.03850 vs prior 0.04284; AUC 0.846 | Only 27 positive training and 27 evaluation snapshots; not xG/EPV |
| Shape, pressing, support, rest defense | Per-snapshot geometry and formation-template assignments exported | Descriptive proxies, no tactical-label accuracy claim |
| Set pieces | 157 restarts indexed, four followed by a same-interval shot within 10s | Descriptive retrieval; no learned corner intervention model |
| Retrieval | 602 evaluation queries with 1,806 neighbors from distinct training possessions | Strictly cross-match; analyst relevance remains unmeasured |
| Manager layer | Evidence-linked brief, bounded positioning sensitivity, constrained lineup assignment | Real lineup/substitution advice abstains on missing availability and role evidence |
| Evaluation and storage | Parquet stage artifacts, SHA-256 manifests, annotation task and evaluator | Model suggestions cannot be counted as reviewed ground truth |

The inventory groups related rows into 14 pipeline components. Its
[machine-readable record](../../results/experiments/v0.2/suite-report.json) links to reports
by their paths under `artifacts/`. Copies of those reports are checked in under
`results/experiments/v0.2/`; data, clips and model weights remain ignored.

## Data splits and boundaries

The pinned Metrica game 1 is the training source, game 2 the development
evaluation source. Game 3 remains untouched. The Gaussian trajectory model
fits game 1's first half and uses its second half for empirical uncertainty
calibration. Its smaller training set differs from the first deterministic
trajectory experiment. The off-screen estimator has no fitted motion model;
its error-radius quantiles come from game 1 and are measured on game 2.

Hyperparameters and training durations were fixed for these initial runs.
No frame-random split is used. Overlapping trajectories, teammates, and adjacent
possessions are correlated. The sample counts do not imply independent trials
or confidence in performance on new competitions, camera systems, or styles.

Snapshot features use current/past observations. Future events and paths are
used only to form labels or evaluate predictions. Pass-option training uses
each candidate's location at release time, never the event's realized endpoint.
The recorded recipient is the label. Completed-pass-only selection is a
material limitation. The next-event and shot labels stop at the possession
boundary; period/recording truncation is censored rather than treated as failure.

The ball forecast uses one second of history and predicts one second ahead,
on 3,283 training and 3,074 evaluation complete-ball windows. The player
Gaussian model uses 31,303 fit, 32,373 calibration, and 62,008 evaluation windows.
Neither is a response model for a coach changing a tactic.

## Video experiment

Clip `2e57b9_0.mp4` and the three YOLO checkpoints come from the pinned
[Roboflow sports setup script](https://github.com/roboflow/sports/blob/42c80c06b6b65a7f89455b89fe31cdf4c38ba227/examples/soccer/setup.sh).
`assets.py` pins the retrieved files' SHA-256 values. The video and weights
are evaluated locally; repository MIT licensing is not assigned to those assets.

The runner decodes source PTS, samples at 5Hz, performs player detection,
tiled ball detection, pitch keypoints, robust homography, anonymous tracking,
and a causal warm-up color classifier. It saves independent frame, detection,
keypoint, calibration, projected-state, and ball-candidate tables, plus a preview.
Color labels are anonymous clusters. Referees and goalkeepers are not silently
assigned to a playing team by shirt color. Off-screen positions are not invented
in the video observation table.

Homography needs at least six usable landmarks, RANSAC inlier support, and a
leave-one-out consistency check. Field geometry uses standard 105×68m dimensions
and penalty/goal-area landmarks, an assumption requiring verification on new
footage. Aerial balls violate the planar projection model. Calibration residuals
and successful fit rates do not establish physical accuracy.

The first image-motion-only association run swapped two team assignments in
an inspected development frame. The appearance gate fixed those two cases.
The [provisional visual spot-check](../../results/research/video-team-spotcheck.json) records
18/20 vs 20/20 uniform agreement on that selected frame. This is assistant visual
review after observing the error, not independent ground truth or a test-set
tracking result. Both `demo-v1` and `demo-v2` are preserved locally.

The OCR stage runs installed Tesseract on sampled shirt crops. Two consistent
readings on different frames above the confidence threshold are required even
for a provisional jersey hypothesis. No track passed the gate. No names are
inferred. This is a measured failure/abstention, not an omitted stage.

The bridge exercises the same tactical feature function on video projections,
scoring both possible directions for each color cluster. Those scores are
explicit domain-transfer diagnostics. They do not enter training, claim known
possession, or drive recommendations.

## Tactical and managerial scope

The nominal formations 4-3-3, 4-4-2, and 4-2-3-1 are geometric templates. A
Hungarian assignment compares normalized positions after removing the deepest
player as a goalkeeper proxy. The result is an instantaneous shape fit, not
evidence of intended roles, responsibilities, or tactical instructions.

The ball's third defines the phase proxy. Distance to the nearest opponent
defines pressure; teammates behind the ball provide a crude rest-defense count.
These outputs are reviewable starting features. They are not automatically
promoted to labeled pressing schemes or analyst-approved descriptions.

Pass alternatives show a learned recipient-imitation score and a toy open-lane
simulation. Ball speed, defender speed/reaction and stationary receiver position
are explicit assumptions. The simulation omits aerial delivery, coordinated
movement, control technique, offside, and opponent adaptation. Its frequencies
are not calibrated success probabilities or a tactical recommendation.

The manager brief cites match, possession, snapshot and retrieved comparison
identifiers. Static scenarios move one observed player at most ten metres and
report feature deltas. There is no estimated causal reward. The lineup solver
optimizes only supplied role scores, enforces one assignment per player and
explicit availability, and refuses missing context. The generated template
contains only the selected team's anonymous provider IDs, with unknown
availability and empty role scores. It therefore abstains as intended.

## Reproduce

After the first-milestone fetch/prepare/analyze/train commands:

```sh
uv sync --locked --python 3.12
uv run --extra vision soccerviz fetch-demo
python3 scripts/spark/spark_video.py --seconds 20 --run-name demo-v2
python3 scripts/spark/spark_forecast.py --epochs 8
python3 scripts/spark/spark_sequence_models.py --model probabilistic
python3 scripts/spark/spark_sequence_models.py --model ball
uv run soccerviz jersey --video-run artifacts/video/demo-v2
uv run soccerviz research
uv run soccerviz workbench --port 7865
```

The video command refuses to overwrite a completed run; choose a fresh run name
for further experiments. The current research dashboard targets `demo-v2`.
Spark jobs run in disposable containers and only use `~/soccerviz` for this
project. The CV image is separate from the existing Spark services. Tesseract
is an optional system dependency for the OCR command and was already installed
on the local machine.

Individual commands include `tactics`, `uncertainty`, `actors`, `manager`,
`inventory`, `train-ball`, `train-probabilistic`, and `video --help`.
`lineup --context artifacts/manager/roster-context-template.json` demonstrates
abstention until reviewed context is supplied. Host neural training needs the
`gpu` extra; the Spark scripts use the cached NVIDIA PyTorch image.

```sh
uv run pytest -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run python scripts/setup/verify_workbench.py
```

The final check uses the prepared artifacts to exercise seven workbench data
flows, including video radar, tactical retrieval, manager scenarios, both
uncertainty views, ball forecasts, and possession replay. The unit tests run
without downloading footage or model weights.

## Next gates

1. Review the exported video annotation task. Measure detection/ball precision,
   tracking identity, team/jersey accuracy, calibration in metres, and failures
   around cuts/occlusions. Model-generated boxes are never evaluation truth.
2. Acquire additional games and verified pitch/camera metadata. Freeze a truly
   untouched test set before selection and calibration iterations.
3. Add analyst labels for pressing, buildup, transitions, rest defense and set
   pieces. Evaluate agreement and cross-team generalization, including abstention.
4. Replace toy ground-pass simulation with multimodal joint player/ball response
   models and validated possession value. Keep offside, ball height, contextual
   state and tactical constraints explicit.
5. Bring real roster, availability, fitness and reviewed role evidence into
   managerial evaluation. Use controlled expert comparisons before issuing
   substitution or tactical-change recommendations.
