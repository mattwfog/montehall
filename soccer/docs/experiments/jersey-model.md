# Uncertainty-JNR jersey experiment

This worker evaluates the released SoccerNet-trained ViT-B from
[Uncertainty-JNR](https://github.com/lukaszgrad/uncertainty-jnr), using its actual
model, configuration, image dataset and validation transforms. Ground truth is
used only in a separate evaluation command after predictions have been saved.
This is a number-reading experiment on predicted boxes, not verified player identity.

## Frozen inputs

The crop source is the completed 450-frame soccer YOLOv8x detector run on SNGS-021,
SNGS-045 and SNGS-089. Its output SHA-256 is
`0a59eeae7d465df02284d9e606ddf8f5799d4b438a64beb2c0e39cf0b6c04473`.
The image manifest SHA-256 is
`75200edcfe9851a5c68a68bfe36ef320755d90d33bd4ece98590c8d31df08795`.

There are 8,885 predicted player/goalkeeper candidates across the three sequences.
The worker chooses 200 crops using equal sequence quotas (67, 67 and 66), then
uniform indices over detections sorted by frame and image x. This sampling does not
use GT boxes, player identities, jersey numbers, or whether a number is visible.
Predicted tracks are unavailable, so no per-identity sampling cap or temporal vote
is claimed; repeated people and adjacent frames remain correlated.

The prepared input is `artifacts/integrations/jersey-model-crops/crop-manifest.json`
with SHA-256 `fec109bdbf67bdd3a3ef253651080f7843ed92d5a254c98bd8595f3985284fc1`.
Each crop is saved losslessly from original image pixels. The manifest records
source image hashes, predicted floating-point boxes, clipped floor/ceil integer
crop bounds, image dimensions, crop hashes, and the model's torso region.
Crop paths are relative to this manifest for portable Spark execution.

The official preprocessing removes the top one-sixth and bottom one-third of each
person crop, resizes the torso to 224 × 224 using cubic interpolation, converts BGR
to RGB, and scales intensities to [-1, 1]. The worker uses the upstream
`SimpleImageDataset` and `get_val_transforms`, rather than recreating that pipeline.
It verifies images before loading: unreadable crops cannot silently become the
upstream loader's zero-image fallback.

## Model and acceptance

The fixed checkpoint is the public **ViT-B, SoccerNet Dataset** release, SHA-256
`43eb17804e94e012883b37618d70fbab3fb190f58eb851843e38589a9cf80923`.
The upstream source revision is `f19d9cb90e1a67d5ffe44acbf69fb348fe6525e7`.
Seven source/configuration files are checked against pinned SHA-256 values before
import, so exported archives do not need runtime Git metadata.
The supplied `base8_reid.yaml` selects `vit_base_patch8_224.augreg2_in21k_ft_in1k`
with the tied digit-aware classifier and Dirichlet uncertainty head.

Inference uses float32 and strict checkpoint loading. Every result retains its full
100-number probability distribution, Dirichlet concentrations, 101 logits,
top number, probability, and uncertainty. Assertions reject inconsistent or
nonfinite outputs. The published Dirichlet head's uncertainty is not a calibrated
probability that a number is unreadable.

The fixed preliminary rule accepts a provisional number only if top-number
probability is at least 0.8 and Dirichlet uncertainty is at most 0.2. Otherwise the
worker abstains. These thresholds were fixed before evaluating JNR outputs; they
are not claimed to be optimal or calibrated. No roster names or persistent player
identities are inferred.

## Evaluation and local comparator

Predicted crops are matched to player/goalkeeper GT boxes in the same original
sequence/frame using one-to-one IoU ≥ 0.5, ordered by detector confidence. The jersey
prediction cannot influence matching. Of the 200 selected crops, 156 match a GT
player/goalkeeper, and 114 have a numeric jersey attribute (22, 45 and 47 by sequence).
Matching always uses all 200 frozen crops, including when inference returns only a
smoke-test subset. Missing outputs count as abstentions, and expected, submitted and
missing counts are reported explicitly. Omitting outputs cannot shrink denominators
or let another detection claim the missing detection's GT match.

GT jersey attributes indicate the player's number, **not its current visibility or
legibility**. Precision and correct-number coverage are therefore conditional on
these matched, labeled crops. Unreadable-class accuracy and full-video jersey recall
cannot be measured from this experiment. The published checkpoint was trained with
SoccerNet JNR and ReID data; match overlap with GSR validation has not been ruled out,
so these are development diagnostics rather than independent generalization results.

The same 200 predicted boxes and identical torso regions were also run through
local Tesseract, adapting the existing PSM 7 digit-only baseline: grayscale, 4× linear
upscale and confidence ≥80. Tesseract accepted 1/200 crops; that crop had no evaluable
matched numeric label. It accepted no correct number among the 114 labeled crops.
Its accepted labeled precision is undefined, rather than zero. Results are saved in
`jersey-model-crops/tesseract-predictions.json` and `tesseract-evaluation-v2.json`.

## Actual Spark results

The full 200-crop JNR run completed on Spark with the NVIDIA 26.01 torch and
torchvision wheels preserved. It loaded every checkpoint key strictly and executed
the upstream model in float32. All 200 probability distributions, Dirichlet
concentrations and logits were checked against the JSON predictions; all source
image, crop, adapter, upstream source and distribution file hashes passed verification.

| Measure | Uncertainty-JNR ViT-B | Same-crop Tesseract |
|---|---:|---:|
| Submitted / expected crops | 200 / 200 | 200 / 200 |
| Accepted number readings | 19 | 1 |
| Acceptance across all sampled crops | 9.5% | 0.5% |
| Accepted readings with evaluable GT number | 19 | 0 |
| Correct accepted readings | 17 | 0 |
| Precision on accepted labeled readings | 17/19 = 89.5% | Undefined |
| Correct accepted coverage of 114 labeled crops | 14.9% | 0% |
| Forced top-one agreement on 114 labeled crops | 28/114 = 24.6% | 0% |

JNR's accepted/correct counts were 10/9 for SNGS-021, 3/2 for SNGS-045, and 6/6
for SNGS-089. The two accepted errors were a predicted 6 against GT 8 and a
predicted 18 against GT 16. The first error had model probability **99.35%**:
high model probability is not verified correctness. Abstention is necessary here,
but these fixed thresholds do not eliminate confidently wrong readings.

The measured inference-and-output phase took **6.64 seconds for 200 crops**
(about 30 crops/s), excluding checkpoint/model initialization and source-image crop
preparation. That is a bounded worker measurement, not end-to-end video throughput.
The runtime used `timm` 1.0.22, Albumentations 1.4.24, NumPy 2.1.0 and
`torch` 2.10.0a0+a36e1d39eb.nv26.1.42222806 on `cuda:0`.

This is a useful improvement over the Tesseract comparator for this sample. Keep
the dedicated model and its abstention in an optional pipeline; next add predicted
track consistency, analyst legibility labels, and separate-match calibration before
using its readings as persistent player identities. Ground-truth tracks must remain
evaluation-only when adding temporal voting. The limited coverage, small correlated
sample, and unverified training-match overlap prevent a broader accuracy claim.

Final artifacts are in `artifacts/integrations/jersey-model-full/`:
`predictions.json`, `distributions.npz`, `evaluation.json`, and `verification.json`.
The evaluation includes all 200 expected crops and reports zero missing outputs.

## Permissive digit reader, first scoring 2026-09-08

`soccerviz.candidates.jersey_digits` reads digit boxes from an RF-DETR Medium
(Apache-2.0) fine-tuned on the CC BY 4.0 Pusan National University
`jersey-number-detection` export (24,315 training images, 640 px, volleyball),
and returns the same prediction contract as JNR. The one-epoch smoke checkpoint
(`rf-digits-smoke`, valid mAP50 0.954 on the export's own split) was scored on
the same 200 frozen crops with `spark_jersey_digits.py score`, which runs the
reader on Spark, pulls the predictions back and evaluates locally against the
same ground truth.

| Measure | RF-DETR digits, Pusan smoke | Uncertainty-JNR ViT-B |
|---|---:|---:|
| Submitted / expected crops | 200 / 200 | 200 / 200 |
| Crops with any digit above 0.5 | 31 | n/a |
| Accepted number readings | 2 | 19 |
| Correct accepted readings | 0 | 17 |
| Forced top-one agreement on 114 labeled crops | 0 / 114 | 28 / 114 |
| Inference and output | 30.8 s | 6.6 s |

No transfer at this scale. The frozen crops are whole-body cutouts, median 40 by
92 px, so a digit is under 10 px wide; the Pusan digit boxes are about 90 px wide
in 640 px images. Of the 63 digit boxes the reader emitted, 27 were class `0`,
and where it fired on a labeled crop it read a different number every time
(6 for 7, 17 for 28, 0 for 24). The ten-times scale gap dominates the
volleyball-to-football gap; the 4-epoch run (`rf-digits-pusan-v1`, 448 px) is
trained on the same export and is scored the same way when it finishes, but a
digit reader for these crops needs training data at this scale: strong
downscale augmentation of the Pusan crops to 30 to 60 px torsos, or a football
set with digits at broadcast scale. Artifacts:
`artifacts/integrations/jersey-digits-rf-digits-smoke/{predictions,evaluation}.json`.

## Permissive digit reader, 4-epoch run scored 2026-09-08

`rf-digits-pusan-v1` (448 px, batch 16, 4 epochs, 3.7 h while sharing the GPU
with the pitch calibrator retrain) finished at valid mAP50 0.957, mAP50:95 0.690
on its own split, best epoch 3. On the same 200 frozen crops: 7 accepted, 0
correct, forced top-one 0 / 114, acceptance coverage 3.5%. Identical to the
smoke: three more epochs on 90 px volleyball digits cannot reach 10 px football
digits. Artifacts:
`artifacts/integrations/jersey-digits-rf-digits-pusan-v1/{predictions,evaluation}.json`,
checkpoints under `artifacts/models/rf-digits-pusan-v1/`.

Decision 2026-09-08: no more digit-detector epochs. Identity is
rebuilt as a track-level inference: a whole-crop number classifier with an
abstain class trained permissively at broadcast scale, decoding restricted to
the team's lineup, evidence aggregated over the track with a legibility gate,
and evaluated as named-identity accuracy and coverage over tracks (oracle
tracks from the SoccerNet labels first, predicted tracklets second).

## Track-level identity build, launched 2026-09-08

Code: `src/soccerviz/candidates/jersey_classifier.py` (model, data, per-crop
contract), `src/soccerviz/vision/track_identity.py` (lineup restriction and
track aggregation), `scripts/training/train_jersey_classifier.py`,
`scripts/benchmarks/run_track_identity_benchmark.py`,
`scripts/spark/spark_jersey_classifier.py`. Run `jersey-resnet34-v1`.

**Classifier.** timm ResNet-34, ImageNet init, 128 px input, 101 classes: 0 to 99
and `illegible`. Every training crop is first downscaled to a torso height drawn
from 16 to 64 px, blurred, motion-blurred, JPEG-compressed, colour-cast and
jittered, then resized to the input, so the network only ever sees broadcast
scale. Untuned choices, stated once: label smoothing 0.05, 12 epochs, taiseis
tiles repeated four times per epoch, Pusan negatives capped at 6,000 per epoch.
`best.pt` is the highest valid number accuracy among epochs whose illegible-class
recall clears 0.8: a model that cannot say "no number" is not deployable.

**Data, all CC BY 4.0.** `taiseis-workspace/jersey-number-ijbaq` v1 (Universe,
classification export; 3,219 train / 436 valid / 432 test tiles at 224 px after
dropping `00`; `no number` and `unreadble` map to `illegible`). Viewed on 48 train
tiles (`universe-previews/taiseis-train-contact.png`): colour broadcast crops,
basketball-heavy by kit and number style (0, 42, 45, 55), some football; the
closest permissive source to the target. Pusan digit boxes read left to right
as whole numbers (23,138 train, 1,152 valid; images with more than two digits or
a leading zero skipped) with a same-sized window below the digits as a
number-free negative (17,591 train). Degraded samples:
`universe-previews/jersey-classifier-degraded-contact.png`.

**Decoding and aggregation.** `restrict_to_lineup` renormalises the number mass
over the team's lineup and keeps the illegible mass. `aggregate_track` sums
legibility-squared-weighted log-likelihoods over a track's frames, skipping
frames under 0.2 legibility, and decides at posterior 0.9 with at least one
weighted frame of evidence. Tested: forty unreadable frames leaning one way
cannot outvote two legible frames; conflicting legible evidence stays undecided.

**Evaluation, oracle tracks.** `run_track_identity_benchmark.py prepare` cut one
torso crop (top sixth and bottom third removed, the JNR rule) for every GT
player and goalkeeper box on the 150 available frames of SNGS-021, SNGS-045 and
SNGS-089: 7,723 crops, 56 tracks, 43 with a jersey label, all internally
consistent. Lineups are the numbers the labels give each team in the sequence, a
stand-in for a provider lineup. Reported unrolled: per-crop open, per-crop
lineup, per-track open, per-track lineup, each with decided / correct / wrong /
undecided. Tracking is held fixed by design; predicted tracklets are the next
stage. SoccerNet stays evaluation-only.

### Result, `jersey-resnet34-v1`, scored 2026-09-08

Training reached 90.5% number accuracy and 0.99 illegible recall on its own
degraded validation tiles (best epoch 11 of 12, 28 min; checkpoint SHA256
`12585310aedc5669…`, registered shipping-class in `checkpoint_licence`, not
deployable). On the real broadcast crops it does not transfer, and the failure
is specific:

| Level | Result |
|---|---|
| 1. Per crop, open, forced top-one on 5,832 labelled oracle crops | 14.8% |
| 2. Per crop, lineup-restricted | 22.4% |
| 3. Per track, open aggregation, 43 labelled tracks | 39 decided, 7 correct, 32 wrong, 4 undecided |
| 4. Per track, lineup + aggregation (the shipping rule) | 40 decided, 11 correct, 29 wrong, 3 undecided |
| Continuity: the 200 frozen crops, forced top-one on 114 labelled | 13.2% (JNR 24.6%, digit reader 0%) |

By sequence, level 4: SNGS-021 5 of 8 decided correct (the largest players,
median box 106 px), SNGS-045 2 of 15, SNGS-089 4 of 17. Coverage is 93% and is
not a result: it is the symptom.

Why, from the saved distributions (`distributions.npz`): the legibility head is
uncalibrated on this domain. Median legibility over all 7,723 crops is 0.91, and
only 1,926 fall under 0.5, although most broadcast torsos show no readable
number (fronts, sides, motion). The number head is diffuse (median top
probability 0.23) and miscalibrated where it is confident (698 crops at
probability ≥ 0.8 are 44% correct; 116 at ≥ 0.95 are 31%). Predictions collapse
onto a few classes, 17 on 1,465 crops and 5 on 871 across all sequences, which
matches the taiseis class frequencies repeated four times per epoch: the prior
leaked. With 150 confident, wrong, "legible" frames per track, the aggregation
does exactly what it should on that input and reaches posterior 1.0 on the
wrong number (SNGS-089: twelve of eighteen tracks decided 27). The aggregation
and lineup steps work as designed (level 2 over level 1, level 4 over level 3);
the input to them is the defect.

What this establishes: the track-level machinery is built and measured, and the
missing piece is a classifier whose illegible class is learned on real
broadcast torsos rather than on number-free windows below volleyball digits,
and whose number prior is flat. No permissive football set with number labels
at broadcast scale exists (census in `guides/public-data.md`). The route that
does not need one: the CC BY Roboflow football player-detection sets already in
the tree (`football-players-detection-3zvbc` v10 / v13 / v20) give thousands of
real broadcast player crops at exactly this scale; nearly all are illegible by
viewpoint and serve as the illegible class as they are, and rendering numbers in
kit fonts onto back-view torsos among them gives in-domain positives with exact
labels and a flat prior. Class-balanced sampling, per-frame likelihood clipping
in the aggregation and a calibration pass on held-out synthetic data complete
it. Built the same day as `jersey-resnet34-synth`, below. Artifacts:
`artifacts/integrations/jersey-classifier-jersey-resnet34-v1-{oracle,frozen}/`,
`artifacts/models/jersey-resnet34-v1/`.

### Result, `jersey-resnet34-synth`, scored 2026-09-08

Same network and degradation; the data changed. Every player and goalkeeper box
of the CC BY `football-players-detection` v10 export is cut to a torso
(`artifacts/integrations/jersey-torso-pool`, 5,174 train, 881 valid) and used
as the illegible class as it is; the same torsos with a number drawn uniformly
from 1..99 in one of eight OFL kit fonts (`artifacts/public-data/fonts`,
provenance in `source.json`) are the positives, rendered at 256 px and pushed
through the same broadcast degradation
(`universe-previews/jersey-synthetic-contact.png`). Real taiseis and Pusan
numbers stay in at 120 per class. Per epoch: 24,000 synthetic, 8,000 illegible,
about 4,500 real. A temperature (0.8) is fitted on the valid set and stored in
the checkpoint. The aggregation clips each frame to 3 nats. Best epoch 10 of 12:
valid number accuracy 80.3%, illegible recall 0.98, precision 0.91. Checkpoint
SHA256 `e83a76b32e875a4f…`, registered shipping-class, not deployable.

| Level | v1 | synth |
|---|---:|---:|
| 1. Per crop, open, forced top-one, 5,832 labelled crops | 14.8% | 16.3% |
| 2. Per crop, lineup-restricted | 22.4% | 29.5% |
| 3. Per track, open: decided / correct / wrong / undecided of 43 | 39 / 7 / 32 / 4 | 28 / 12 / 16 / 15 |
| 4. Per track, lineup + aggregation | 40 / 11 / 29 / 3 | 31 / 16 / 15 / 12 |
| Accuracy over decided tracks, level 4 | 27.5% | 51.6% |
| 200 frozen crops: accepted / precision / forced top-one | 49 / 27.3% / 13.2% | 11 / 66.7% / 14.9% |

By sequence, level 4: SNGS-021 6 of 6 decided correct (0 wrong), SNGS-045 5
correct of 9, SNGS-089 5 correct of 16.

What moved: the illegible class transferred. Median legibility over the 7,723
crops fell from 0.91 to 0.11, and 5,276 crops now sit under 0.5, which is what
broadcast torsos look like; the aggregation therefore abstains where it should
(12 undecided) and wrong decisions halved. What did not move enough: the number
head. Open forced accuracy on real crops is 16%, against 24.6% for JNR trained
on SoccerNet itself, and the top predictions still cluster on thin-stroke shapes
(1 on 853 crops, 7 on 743, then 71, 5, 44, 77): at this resolution a partial
vertical stroke reads as 1 or 7, and the synthetic digits do not yet carry
enough of the real stroke variety to break that. SNGS-089 shows the limit of
the lineup lever too: both teams there field a 27, so seven wrong tracks decide
27 and the lineup cannot reject it.

Where this leaves identity: track-level machinery works and is measured; the
classifier's abstention is now usable and its reading is not. Levers left, in
the order the evidence points: real in-domain digit appearance (raise the real
per-class cap, and pull the taiseis test split into training since the oracle
set is the held-out measure), rendering variety (stroke weight, outline width,
number wear, sponsor clutter behind the number), and a stronger backbone last.
Predicted tracklets instead of oracle tracks stay the stage after a usable
reader. Artifacts:
`artifacts/integrations/jersey-classifier-jersey-resnet34-synth-{oracle,frozen}/`,
`artifacts/models/jersey-resnet34-synth/`.

## Commands

Build with `envs/jersey-model/Dockerfile`. Its base is NVIDIA PyTorch 26.01 and it
pins the image's exact torch/torchvision versions while installing the minimal
inference dependencies; it does not replace the GB10-compatible NVIDIA wheels.

```sh
PYTHONPATH=src .venv/bin/python scripts/benchmarks/run_jersey_benchmark.py prepare \
  --manifest artifacts/vision-benchmark/manifest.json \
  --detections artifacts/vision-benchmark/yolov8x-soccer-predictions.json \
  --out artifacts/integrations/jersey-model-crops --limit 200

# Inside the Spark container, with source at /work and checkpoint at /models:
python scripts/benchmarks/run_jersey_benchmark.py infer \
  --crop-manifest /work/artifacts/integrations/jersey-model-crops/crop-manifest.json \
  --upstream /work/artifacts/upstreams/uncertainty-jnr \
  --checkpoint /models/jersey-vitb.pt --out /results/jersey-full --batch-size 8

PYTHONPATH=src .venv/bin/python scripts/benchmarks/run_jersey_benchmark.py evaluate \
  --crop-manifest artifacts/integrations/jersey-model-crops/crop-manifest.json \
  --predictions PATH_TO_JNR_PREDICTIONS_JSON \
  --truth artifacts/vision-benchmark/ground-truth.json --out PATH_TO_NEW_EVALUATION_JSON

PYTHONPATH=src .venv/bin/python -m pytest tests/test_jersey_model_adapter.py -q
```

Use new output paths for repeat runs; completed files are not overwritten.
The inference smoke option `--limit 8` explicitly marks the output incomplete.
Prepared crops, distributions and predictions contain no GT-derived inference inputs.
The evaluator has a separate `--path-prefix OLD=NEW` option when original annotation
locations need remapping without modifying frozen manifests.

Eighteen focused tests pass. They cover invalid/missing numbers, unreadable or
out-of-bounds crops, inconsistent probabilities/concentrations, abstention rules,
GT-free sampling, source-hash mismatch, exported source integrity, unlabeled evaluation exclusions, and stable
denominators when outputs are missing. Ruff checks also pass.

Upstream attribution and license: Łukasz Grad, *Single-Stage Uncertainty-Aware Jersey
Number Recognition in Soccer*, CVPR Workshops 2025;
[source and checkpoint instructions](https://github.com/lukaszgrad/uncertainty-jnr/blob/main/docs/INFERENCE.md).
The upstream repository is CC-BY-SA-4.0; its source is imported without modification.
