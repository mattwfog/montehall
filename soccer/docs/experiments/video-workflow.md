# Video workflow on the frozen corpus

The video workflow (`src/soccerviz/vision/video_workflow.py`) is the integrated
path: selectable detector preset, PnLCalib or YOLO calibration behind the
camera guard, causal ball track, shot-local jersey consensus, durable Spark
jobs. This page records its measured behaviour on the frozen workflow corpus.
Detector-only numbers live in [detector training](detector-training.md) and
the [candidate report](../reports/candidate-experiments-2026-09-07.md).

## Corpus

Four SoccerNet GSR validation sequences chosen before any workflow inference
(SNGS-022, 050, 079, 093; games 2, 3, 5, 5), 150 consecutive native frames
each, sampled every fifth frame into lossless 5 Hz clips
(`artifacts/workflow-corpus/videos5hz/<seq>/source.mp4`, frame maps beside
them). Evaluation manifest and ground truth: `artifacts/workflow-corpus/evaluation/`,
120 frames, 2,063 person boxes, 120 ball boxes. Official test and challenge
splits stay unopened (`artifacts/workflow-corpus/holdout-policy.json`).

## Runs scored

Two batches, both preset `rf-soccer`, PnLCalib, 6 s at 5 Hz, jersey on:

- `artifacts/workflow-corpus/jobs-rf-soccer-20260908.json`, image
  `soccerviz-workflow:20260908`, where `rf-soccer.pth` was the SoccerNet-trained
  five-epoch pilot (SHA256 `4e5cb479…`), evaluation-only under the copyleft
  rule. Summary tag `rf-soccer-20260908`.
- `artifacts/workflow-corpus/jobs.json`, image `soccerviz-workflow:20260908-e50`,
  where `rf-soccer.pth` is the Roboflow v10 50-epoch model (SHA256
  `e5a740c7…`, CC BY training data, shipping-cleared), promoted as the default
  on 2026-09-08. Summary tag `rf-soccer-e50-20260908`.

## Workflow image

The image is built by hand on Spark, not by a script in this repo:
`~/soccerviz-workflow-build/` holds a copy of `envs/workflow/Dockerfile`
(identical, checked 2026-09-08), `models/` (the ten checkpoints the presets
load) and `upstreams/` (pinned PnLCalib and Uncertainty-JNR checkouts).
`docker build -t soccerviz-workflow:<tag> .` there produces the image, and
`artifacts/workflow-runtime.json` pins its sha256 image id for the durable
executor. Retired checkpoints move to `models-retired/` so they are not
copied into the next image.

| Tag | Built | `rf-soccer.pth` |
| --- | --- | --- |
| `soccerviz-workflow:20260908` | 2026-09-07 22:07 EDT | SoccerNet five-epoch pilot, SHA256 `4e5cb479…` (evaluation-only) |
| `soccerviz-workflow:20260908-e50` | 2026-09-08 | Roboflow v10 50-epoch model, SHA256 `e5a740c7…` (shipping-cleared) |

## Scoring method

`scripts/benchmarks/score_workflow_corpus.py --tag rf-soccer-20260908` maps
each run's frames to the manifest through the clip frame map and scores three
separate prediction sets with the frozen detector protocol (score 0.25, IoU
0.50, ball centre within 10 source pixels):

- persons and raw ball candidates: every person box with roles collapsed,
  every raw ball candidate box, at the pipeline's own confidences;
- ball confirmed: centres the causal track reported as `detected`;
- ball estimated: centres reported as `tentative` or `estimated`.

The last two are binary pipeline decisions scored at 1.0. The workflow ran at
its operating threshold, so there is no low-score capture floor and no box AP
comparable with the detector benchmark. Outputs:
`artifacts/workflow-corpus/evaluation/rf-soccer-20260908-*-{predictions,results}.json`,
summary `results/experiments/workflow-corpus-rf-soccer-20260908-summary.json`.

## Results, 2026-09-08

Same 120 frames, same protocol, same PnLCalib camera and jersey model; only
the detector checkpoint differs.

| Set | SoccerNet pilot P / R | Roboflow e50 P / R | Counts, Roboflow e50 |
| --- | ---: | ---: | --- |
| Person boxes | 70.2% / 83.5% | **79.5% / 84.9%** | 1,751 matched, 451 false positives (3.8 per frame, from 6.1), 2,063 annotated |
| Raw ball candidates | 100% / 45.8% | 52.9% / 45.8% | 55 matched, 49 false, median centre error 2.0 px |
| Ball confirmed by the track | 100% / 37.5% | 80.3% / 40.8% | 49 matched, 12 false, median centre error 1.9 px |
| Ball estimated by the track | 73.7% / 11.7% | 24.2% / 6.7% | 8 matched, 25 false, median centre error 4.5 px |

Per sequence, person precision/recall, pilot to e50: SNGS-022 71.9/90.9 to
67.7/85.1, SNGS-050 62.4/87.4 to 86.3/93.5, SNGS-079 73.1/61.1 to 73.2/67.2,
SNGS-093 77.7/97.3 to 83.5/92.2. Confirmed ball true/false frames: SNGS-022
8/0 to 7/1, SNGS-050 0/0 to 0/11, SNGS-079 8/0 to 13/0, SNGS-093 29/0 to 29/0.
Camera-guard acceptance is unchanged (28, 30, 30, 30 of 30).

Reading: the person channel improved as the benchmark predicted, false boxes
down by 38% with recall held. The ball channel regressed: the pilot produced no
ball candidate at all on SNGS-050, while the e50 model produces confident wrong
ones there, and the causal track confirms eleven of them. Every ball false
positive in the confirmed set but one comes from that sequence. The ball head
of the e50 model (benchmark ball-centre precision 26%) is the weak part of the
promotion and the reason the ball needs its own training data. These are 120
development frames from three games; not a generalization claim.

## Camera guard and the COCO preset

Rerunning SNGS-022 with preset `rf-general` (RF-DETR Medium COCO, 960 px) on
2026-09-08 accepted 11 of 30 cameras against 28 with `rf-soccer`, from the
same raw PnLCalib homographies. The COCO model's single `person` class boxes
bench, staff and crowd; their feet project off the pitch, the in-pitch
fraction drops to 0.36 to 0.50 on frames 13 to 29, and the guard's 0.7 rule
rejects a correct camera. The guard therefore presumes a role-aware detector.
Job `968bb722…` under `artifacts/workflow-jobs/`.

## Licence class of every run

Since 2026-09-08 each run's `configuration.json` and `report.json` carry a
`licence` block from `soccerviz.vision.checkpoint_licence`: the class
(`shipping`, `evaluation-only` or `no-checkpoints`), the roles that block
shipping, and each loaded checkpoint's registered source and licence. The
e50 batch is the first with real reports carrying the block: class
`evaluation-only`, blocking `camera_keypoints`, `camera_lines`, `jersey`, with
the detector `e5a740c7…` cleared. The detector half is clean; the PnLCalib
camera and the jersey model are still SoccerNet-trained (see the
[licence census](../guides/public-data.md#licence-census-and-the-copyleft-rule)).
`--shipping` on the workflow command refuses such a run before any frame is
read. A checkpoint whose SHA256 is not registered cannot run at all.

## Open

- Give the ball its own training data; the promoted detector's ball head
  produces confident false candidates on SNGS-050.
- The human formation and pressing review queue
  (`artifacts/shape-review-queue.json`, Video workflow tab) has no reviews
  yet; nothing tactical is measured here.
- The tab itself has only been exercised by tests, never opened in a browser.
