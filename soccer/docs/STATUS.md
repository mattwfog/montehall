# Status

One page. What runs by default, what has been measured, what is a candidate,
and what is open. Every number below is copied from the linked source; none was
re-measured when this page was written on 2026-09-07. Update this page in the
same commit as any change to a default or a measured number.

## Default pipeline

- Video: Roboflow soccer YOLOv8x detector at 1280 px plus four ball tiles,
  anonymous tracklet association, homography calibration, jersey OCR evidence.
  Code: `src/soccerviz/vision/`.
- Provider tracking: Metrica sample data (two matches), SkillCorner and StatsBomb
  public imports. Code: `src/soccerviz/providers/`, `src/soccerviz/datasets/`.
- Shared state, checks and review: the specialist harness with immutable
  revisions, pause/resume and replay. Code: `src/soccerviz/harness/`.
- Tactical models: progression classifier, player GRU forecast, ball and actor
  models, formation templates, lineup assignment. Code: `src/soccerviz/modeling/`.
- Video workflow: selectable RF-DETR or YOLO presets with camera guards, causal
  ball confirmation and short-gap estimates, shot-local jersey consensus,
  annotated replay, and durable Spark submission with hash-verified retrieval.
  Code: `src/soccerviz/vision/video_workflow.py`, `video_safety.py`,
  `video_jobs.py`; tab in `src/soccerviz/ui/video_workflow_ui.py`. Default
  detector since 2026-09-08: RF-DETR Medium trained 50 epochs on the CC BY
  Roboflow v10 corpus (`rf-soccer.pth`, SHA256 `e5a740c7…`, image
  `soccerviz-workflow:20260908-e50`). Four SoccerNet validation clips are
  scored on it below.
- Workbench: eleven Gradio tabs. `uv run soccerviz workbench --port 7865`.

The Roboflow-trained RF-DETR is the first candidate promoted into this
default (2026-09-08). No calibrator, ball model, jersey model or formation
method has been promoted.

## Measured, on the default pipeline

Source: [first milestone](experiments/first-milestone.md),
[scaffold and results](experiments/scaffold-and-results.md).

| Metric | Result | Baseline |
| --- | ---: | ---: |
| Progression Brier score (lower is better) | 0.203 | 0.239 prior |
| Progression ROC AUC | 0.730 | 0.500 |
| Player trajectory mean displacement, 3 s | 1.07 m | 1.57 m constant velocity |
| Pass-recipient top-1 | 45.2% | 33.4% |
| Simulated hidden-player MAE | 0.95 m | 2.80 m |
| Event-actor accuracy at 77.5% coverage | 91.0% | |

Weak or failing on the default pipeline: next-action accuracy is below the
majority class, ball forecasting does not beat constant velocity on mean error,
and the 90% player forecast region covers 88.4% of evaluation points.

## Measured, video workflow on the frozen 120-frame corpus

Source: [video workflow](experiments/video-workflow.md), preset `rf-soccer`,
scored 2026-09-08 before and after the detector promotion.

| Set | SoccerNet pilot P / R | Roboflow e50 (default) P / R |
| --- | ---: | ---: |
| Person boxes | 70.2% / 83.5% | 79.5% / 84.9% |
| Raw ball candidates | 100% / 45.8% | 52.9% / 45.8% |
| Ball confirmed by the causal track | 100% / 37.5% | 80.3% / 40.8% |
| Ball estimated by the track | 73.7% / 11.7% | 24.2% / 6.7% |

## Candidates, measured against the frozen 450-frame benchmark

Source: [candidate experiments 2026-09-07](reports/candidate-experiments-2026-09-07.md).
Manifest SHA `75200edcfe9851a5c68a68bfe36ef320755d90d33bd4ece98590c8d31df08795`.

| Candidate | Verdict | Key numbers |
| --- | --- | --- |
| RF-DETR Medium COCO, 960 px | Best detector, not promotable: its single `person` class boxes bench and crowd, so the camera guard rejects close shots (SNGS-022 rerun 2026-09-08: 11 of 30 cameras accepted vs 28 on `rf-soccer`, job `968bb722…`). Needs a role-aware model, hence the training step below | HOTA 0.722 vs 0.522 default; ball recall 38.7% vs 10.0%; 9.0 frames/s |
| RF-DETR Medium COCO, 576 px | Faster alternative | HOTA 0.711; 16.2 frames/s |
| YOLO26 Medium COCO, 640 px | Throughput option | HOTA 0.701; small-person recall 8.9%; 32.9 frames/s |
| Five-epoch soccer fine-tunes (RF-DETR, YOLO26) | Path works, models regress overall | Small-person recall up, HOTA down to 0.622 / 0.637 |
| RF-DETR five-epoch fine-tune on Roboflow CC BY v10 (permissive path) | Superseded by the 50-epoch run below | HOTA 0.661; person precision 85.4%; small-person recall 61.2%; ball-center precision 27.0% |
| RF-DETR 50-epoch fine-tune on Roboflow CC BY v10 (`rf-roboflow-v10-e50`, scored 2026-09-08) | **Promoted to the workflow default 2026-09-08**: best fine-tune and the only role-aware detector cleared for shipping, though still below the COCO control on HOTA. On the four-clip corpus person precision 70.2% to 79.5%; ball false candidates appeared on SNGS-050 | HOTA 0.679; IDF1 81.3%; person precision 88.7%; FP/frame 2.34; small-person recall 59.9%; ball-center precision 26.2%; internal mAP peaked at epoch 9 |
| PnLCalib calibration | Needs physical and temporal rejection first | Median error 0.48 m vs 4.29 m; p95 18.23 m; two catastrophic frames |
| Permissive heatmap calibrator (`pitch-hrnet-w32-v18` → `v18-vis` → `v18-lines`, 2026-09-08) | Candidate, **not promoted**: on the game-disjoint corpus (`artifacts/calibration-benchmark`, 360 frames, 6 unseen GSR games, frozen 2026-09-08) it loses to PnLCalib on every line, so the 42 / 42 on the three development games did not transfer. The fitter fixes from the rejection audit (tangent gauge, 5 m prune, DLT fallback) are in `primitive_calibration` with tests; SoccerNet's camera is wrong on SNGS-089 frames 1, 11, 21 and excluded at the ground-truth index. Failure census 2026-09-09 (`scripts/benchmarks/census_calibration_failures.py`, PnLCalib as reference): 39 of the 48 attributable failures have every visible line detected correctly and fail in the free-homography fit (halfway views with the circle cut by the frame, end views extrapolating past their evidence); more of the same training data cannot fix those 39. Model comparison 2026-09-10 (`scripts/benchmarks/evaluate_fit_variants.py`, same maps and acceptance, only the model changed): a pinhole camera held at a per-game position estimated from its own fits solves 27 of those 39 and lifts the corpus to 84.7% within 2 m, p95 2.72 m, still below PnLCalib's 89.2% and 348 accepted; the free homography is the cause of the 39. Registered shipping-class in `checkpoint_licence` (SHA256 `5f3e7b12…`) | Game-disjoint 360 frames: v18-lines with the free homography 319 / 360 accepted, 74.8% within 2 m, median 0.72 m, p95 5.39 m; with the held pinhole camera 326 / 360, 84.7%, 0.54 m, 2.72 m; PnLCalib 348 / 360, 89.2%, 0.52 m, 2.9 m. Development 42 frames: 42 / 42, 99.8%, 0.45 m, 1.39 m |
| WASB temporal ball | Comparator only | 6.0% recall, 39.4% precision |
| Uncertainty-JNR jersey numbers | Aggregate over tracks before naming players | 89.5% accepted precision, 9.5% acceptance |
| Permissive RF-DETR digit reader on Pusan volleyball digits (one-epoch smoke and 4-epoch `rf-digits-pusan-v1`, 2026-09-08) | Dead end, closed: no transfer to broadcast crops at either training length (digits under 10 px wide against 90 px in training). Replaced by the track-level identity build: whole-crop classifier with abstain, lineup-constrained decoding, track aggregation, track-level evaluation | smoke 2 accepted / 0 correct; 4-epoch 7 accepted / 0 correct; forced top-one 0 / 114 both; valid mAP50 0.954 / 0.957 on its own split |
| Permissive whole-crop jersey classifier with abstain, track-level identity (`jersey-resnet34-v1` then `jersey-resnet34-synth`, 2026-09-08) | Candidate, not deployable. Machinery built and measured on 43 oracle tracks over three sequences. The synthetic route (numbers rendered on real CC BY broadcast torsos, the same torsos as the illegible class) fixed abstention: median legibility 0.91 to 0.11, wrong track decisions 29 to 15. Reading is still weak: open forced accuracy 16% on real crops against 24.6% for SoccerNet-trained JNR, predictions cluster on thin-stroke shapes. Next levers: real digit appearance (raise the real per-class cap, fold the taiseis test split in), rendering variety, backbone last | synth: per crop forced 16.3% open / 29.5% lineup; per track lineup rule 31 decided / 16 correct / 15 wrong / 12 undecided of 43 (v1: 40 / 11 / 29 / 3); 200 frozen crops 11 accepted at 66.7% precision, forced 14.9% |
| UnravelSports EFPI formations | Optional descriptive tool | Less stable than the three-template baseline; no analyst labels |
| ViTPose++ base body pose | Keypoints only; no labels, sanity rates | 9,388 people; 94.4% keypoints in box; 93.0% with confident foot; 400 ms/frame on MPS |
| SAM 3D Body mesh (450 frames, 2026-09-08) | Measured, no consumer: 2D keypoints slightly worse than ViTPose++ on the same boxes at 30 times the cost; the 3D body output has no ground truth to score it | 9,388 people; 91.6% keypoints in box; 99.85% feet below hips; reprojection within 0.5 px; median body height 1.53 m; 11.6 s/frame on Spark |

Per-candidate writeups: [calibration](experiments/calibration-model.md),
[temporal ball](experiments/temporal-ball.md), [jersey](experiments/jersey-model.md),
[formations](experiments/formation-analysis.md),
[RF-DETR training, SoccerNet and Roboflow pilots](experiments/detector-training.md),
[YOLO26 training](experiments/yolo-candidate-training.md),
[body pose](experiments/pose-model.md),
[body mesh](experiments/mesh-model.md).

## Constraints

- Avoid GPL and AGPL data and model code where a permissive alternative
  exists (2026-09-07). The default YOLOv8x detector is Ultralytics,
  AGPL-3.0; RF-DETR is Apache-2.0. SoccerNet is GPL-3.0 and is evaluation only.
  Census and consequences: [licence census](guides/public-data.md#licence-census-and-the-copyleft-rule).
- Resolved 2026-09-08: the workflow default `rf-soccer.pth`
  is now the Roboflow-trained 50-epoch RF-DETR (SHA256 `e5a740c7…`, CC BY
  data), replacing the SoccerNet pilot. The detector half of the shipping
  path is clean; runs stay `evaluation-only` until the camera and jersey
  models below are replaced.
- Found 2026-09-08: the workflow's default calibrator PnLCalib is GPL-2.0 with
  SoccerNet-trained weights, so the camera half of the shipping path is
  copyleft too. It has no permissive replacement yet; see census consequence 4.
- Found 2026-09-08: StatsBomb Open Data forbids commercial exploitation of the
  data or analysis derived from it (agreement clause 1.2.2). The VAEP models
  trained on it are research artefacts. Resolved the same day: the same
  protocol retrained on the CC BY 4.0 Wyscout open corpus (300 matches, seven
  competitions) matches the StatsBomb skill, see
  [Wyscout training](experiments/wyscout-training.md). Metrica sample data
  has no licence; SkillCorner is MIT.
- Found 2026-09-08: the jersey model (Uncertainty-JNR) is CC BY-NC-SA with
  SoccerNet-trained weights. Of the ten checkpoints in the workflow image only
  `rf-detr-medium.pth` is cleared for shipping. Guard: every checkpoint the
  workflow loads must be registered by SHA256 in
  `soccerviz.vision.checkpoint_licence`; runs report their licence class and
  `--shipping` refuses anything not fully cleared.

## Open

Unresolved top-level goals:

1. Named-player identity. Jersey readings are single-crop hypotheses, not
   persistent identities.
2. Independently validated tactical recognition. No analyst formation or
   phase labels exist yet.
3. Causal managerial recommendations. The forecast models are deterministic
   motion predictors, not intervention responses.

Next concrete steps, in order:

1. Train RF-DETR on the larger CC BY Roboflow Universe sets (the 5,761-image
   `soccer-players-xy9vk`, class-remap census first; not yet downloaded). The
   v10 corpus is saturated: the 50-epoch run's internal mAP peaked at epoch 9.
   Give the ball its own data. Do not tune against the 450-frame development set.
2. Permissive calibrator: the line-aware model and the 360-frame game-disjoint
   corpus exist; the 2026-09-09 census shows the corpus failures occur with
   correct detections inside the free-homography fit, and the 2026-09-10 model
   comparison shows a pinhole camera held at a per-game position solves 27 of
   those 39 on the same evidence (84.7% within 2 m against PnLCalib's 89.2%,
   [calibration](experiments/calibration-model.md)). Not built into the product
   path: a held camera needs a position estimate that survives clips where the
   free fits scatter and a camera-plausibility acceptance check; lens distortion
   is untested. Open decision; the promotion bar is to be ratified first. PnLCalib
   still needs rejection rules so catastrophic frames cannot hide behind the median.
3. Jersey identity at track level: whole-crop classifier with abstain trained at
   broadcast scale, lineup-constrained decoding, legibility-weighted track
   aggregation, measured as identity accuracy and coverage over oracle tracks
   first (`run_track_identity_benchmark.py`), predicted tracklets second. Build
   launched 2026-09-08 (`jersey-resnet34-v1`).
4. Open the Video workflow tab in a browser once (upload, submit, retrieve,
   frame slider, provider replay, review save have only been exercised by
   tests). The corpus scoring is done; rerun it on the replacement preset.
5. Collect independent video annotation and a larger game-disjoint corpus.
6. Wyscout is in and the VAEP baseline is retrained on it (2026-09-08). The
   9,310-crop `pusan-national-university-aajlj` jersey-digit set (CC BY 4.0) is
   in; the RF-DETR digit detector's one-epoch smoke reached valid mAP50 0.954
   on its own split but read 0 of 114 labelled SoccerNet crops (digits under
   10 px wide there, 90 px in training). The 4-epoch run `rf-digits-pusan-v1`
   is scored the same way when it finishes; the real next step is
   scale-matched training data (downscale augmentation of the Pusan crops, or
   a football set at broadcast scale). TeamTrack (MIT, full-pitch 4K video) is
   worth a benchmark before training.

## Verification

Full local suite: `uv run pytest -q`. Last run on this layout (2026-09-10): 357 passed,
29 skipped (isolated-worker tests skip without their `.venvs/` environments).
Isolated workers install with `.venv/bin/python scripts/setup/setup_integrations.py all`.
