# Executed vision candidate experiments

Completed September 7, 2026 (America/New_York), in batches of subagents with
parent-managed sequential Spark GPU jobs. This extends the existing research
system with executable candidate workers, actual comparisons, and two newly
fine-tuned detector checkpoints. No candidate has been promoted to the default
video pipeline.

## What is now in place

- RF-DETR and Ultralytics detector adapters with explicit class mappings,
  source-pixel boxes, checkpoint hashes, timing, and immutable predictions.
- A frozen 450-frame, three-game validation benchmark with independent labels,
  COCO box AP, point-based ball scoring, and fixed-tracker comparisons.
- Actual PnLCalib, WASB, Uncertainty-JNR, and UnravelSports EFPI workers.
- Shared COCO/YOLO training exports and executed five-epoch RF-DETR Medium and
  YOLO26 Medium soccer fine-tuning pilots.
- Eighteen verified experiment records in **Datasets & benchmarks**. The
  read-only selector no longer stalls on its second selection: catalog lookup
  handlers bypass the Gradio queue. Other workbench pipelines retain their queues.

The running workbench is at **http://127.0.0.1:7867**. Ports 7865/7866 were not
stopped. The new workers are research adapters and comparison tools; the original
video extraction defaults remain unchanged.

## Detector results

All rows use the same 450 frames from SNGS-021, SNGS-045, and SNGS-089, containing
8,719 person boxes and 432 ball boxes. Reported precision/recall use score >=0.25;
person matching uses IoU >=0.50, and ball centers must be within 10 source pixels.
All detector AP runs retain candidates down to 0.01. HOTA uses the same existing
anonymous associator with appearance disabled and person roles collapsed.

| Configuration | Person precision | Person recall | Small-person recall | Ball-center precision | Ball-center recall | HOTA | Frames/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing soccer YOLOv8x, 1280 + four ball tiles | 80.1% | 86.3% | 49.0% | 48.3% | 10.0% | 0.522 | 4.1 |
| RF-DETR Medium COCO, 576 | 91.4% | 93.7% | 28.7% | 44.4% | 25.5% | 0.711 | 16.2 |
| RF-DETR Medium COCO, 960 | 90.5% | 94.3% | 36.3% | 43.3% | 38.7% | 0.722 | 9.0 |
| RF-DETR Medium soccer pilot, 576 | 79.2% | 94.6% | 63.1% | 81.9% | 22.0% | 0.622 | 14.6 |
| YOLO26 Medium COCO, 640 | 92.0% | 91.2% | 8.9% | 59.5% | 26.2% | 0.701 | 32.9 |
| YOLO26 Medium soccer pilot, 576 | 88.8% | 89.5% | 38.2% | 81.0% | 7.9% | 0.637 | 33.1 |

The 960-pixel RF experiment was declared after observing small-object misses in
the initial comparison. It is an adaptive development trial, not an untouched
confirmation set. Small means source box area below 32 × 32 pixels; only
157 person boxes fall into that bucket.

These are comparisons of the actual configurations, not proof of architecture
superiority. Resolutions, resizing, training histories, and ontology differ. The
old YOLO row includes a person forward pass plus four dedicated ball passes;
newer detector rows use one model. Throughput includes image loading, integrity
hashing, and model preprocessing/postprocessing, but excludes initialization and
warm-up. COCO models predict generic persons, not soccer roles or named identities.
The clips total roughly 18 seconds and do not establish full-match performance.

Full evidence is in `artifacts/vision-benchmark/*-results.json`, corresponding
`*-predictions.json`, and fixed-tracker output files. The frozen manifest SHA is
`75200edcfe9851a5c68a68bfe36ef320755d90d33bd4ece98590c8d31df08795`.

## Training actually completed

Both pilots use 150 training images from official TRAIN games 4 and 6, with
75 images from official TRAIN game 9 reserved for internal checkpoint selection.
External benchmark games 2, 3, and 5 are excluded from both fitting and selection.
The four classes are player, goalkeeper, referee, and ball.

| Pilot | Epochs | Verified optimizer updates | Elapsed | Selected local checkpoint |
|---|---:|---:|---:|---|
| RF-DETR Medium | 5 | 95 | 207.6 s | `artifacts/models/rf-detr-soccer-pilot/checkpoint_best_total.pth` |
| YOLO26 Medium | 5 | 93 | 131.0 s | `artifacts/models/yolo26m-soccer-pilot/best.pt` |

Both runs verify finite gradients, changed classification-head weights, and a
four-class checkpoint reload. Local checkpoint hashes match the training evidence
and the checkpoints used for external inference. RF selected its native best EMA
checkpoint; YOLO selected native best.pt using only the internal development game.
The methods share data and a bounded budget, but native augmentation, loss,
scheduler, precision, parameter groups, and update timing differ.

The pilots improve small-person recall and conditional ball precision but reduce
overall person/tracking performance versus their generic starting configurations.
They establish a working fine-tuning path and usable saved models, not convergence
or a reason to replace the current pipeline automatically. The training set is
only two short match segments; public pretraining overlap remains unknown.

See [RF training](../experiments/detector-training.md) and
[YOLO training](../experiments/yolo-candidate-training.md).

## Other specialists

**Calibration:** 45 sampled frames contain 876 annotated person footpoints.
PnLCalib projects all 876, with 735 within 2 m (83.9%), versus 97 of all 876
(11.1%) for the existing YOLO calibration. Median error improves from 4.29 m to
0.48 m. However, PnL's p95 is 18.23 m and mean is 37.89 m: two accepted frames
have catastrophic errors. Finite matrices are not reliable calibration. Also,
44/45 PnL frames have no accepted line candidates, so this run does not isolate
a benefit from line refinement. The evaluator uses annotated image footpoints
to isolate calibration; this is not full detector-to-pitch accuracy.
[Details](../experiments/calibration-model.md)

**Ball:** WASB's real three-frame model runs on all 450 frames. At the fixed
operating point it produces 26 correct centers and 40 false positives, for
6.0% recall and 39.4% precision. This is worse than the existing detector on
recall. Its context uses up to two future frames (80 ms), and its peak scores
are uncalibrated heatmap values. Retain as a comparator, not a replacement.
[Details](../experiments/temporal-ball.md)

**Jersey recognition:** Uncertainty-JNR processes 200 crops selected from actual
YOLO detections. Of 114 crops with matched numeric labels, 19 readings are
accepted and 17 are correct: 89.5% accepted precision, 14.9% correct accepted
coverage of labeled crops, and 9.5% overall acceptance. Same-crop Tesseract yields
no correct accepted labeled readings. Ground-truth numbers do not prove that
they are legible in each frame. These are single-crop hypotheses, not persistent
identities; one wrong reading had 99.35% model probability.
[Details](../experiments/jersey-model.md)

**Team shape:** Actual UnravelSports EFPI runs on five minutes from each Metrica
game at 1 Hz. After explicit possession, ball, and coordinate gates, 126/300 and
209/300 frames qualify. Coordinate and source-clock conversion is exact for
7,705 retained observations. Frame-level formation labels are less stable than
the existing three-template baseline, and no analyst formation labels were
available to measure accuracy. Keep as an optional descriptive tool.
[Details](../experiments/formation-analysis.md)

## Decisions and remaining work

1. RF-DETR deserves continued development: generic 960 gives the strongest
   measured HOTA and ball recall here, while generic 576 is faster. Neither
   supplies soccer-role identification without additional modeling.
2. YOLO26 is a useful throughput-oriented alternative. Its generic native-size
   configuration has weak recall on the smallest people in these clips.
3. Expand training across more matches and difficult small-object cases before
   another fine-tuning round. Do not tune repeatedly against this external
   development set and call it final validation.
4. Add physical and temporal rejection to PnL calibration before integration as
   a default. Do not hide catastrophic frames behind the good median.
5. Combine jersey hypotheses over reliable predicted tracks and measure
   legibility, precision, and coverage before naming players.
6. Ball coverage and reliable team-shape interpretation remain unsolved. The
   broader foundation-model and segmentation experiments from the survey were
   intentionally deferred.

## Verification and reproduction

The complete local suite passes **266 tests**, with **23 optional tests skipped**.
Focused integration tests also exercised real EFPI, official TrackEval/COCO
metrics, and the full source annotation census. Browser verification checks
consecutive RF-DETR, PnLCalib, and jersey selections after the queue fix; the
workbench shows coverage and failure warnings alongside headline metrics.

Useful commands, run from this repository:

```sh
PYTHONPATH=src .venvs/evaluation/bin/python -m soccerviz.candidates.vision_benchmark evaluate \
  --manifest artifacts/vision-benchmark/manifest.json \
  --predictions artifacts/vision-benchmark/rf-medium-coco-predictions.json \
  --output /private/tmp/new-rf-evaluation.json
PYTHONPATH=src .venv/bin/python scripts/benchmarks/register_vision_candidate_results.py
.venv/bin/python -m soccerviz.cli.main workbench --port 7867
```

Use a new output filename/directory for each run. Do not launch a second server
on a port that is already occupied. Per-specialist guides contain the native
worker commands, dependency files, source hashes, and limitations. Docker images
are isolated on Spark and preserve its NVIDIA-provided PyTorch wheels. The
existing Spark services were not stopped or replaced.
