# Soccer detector training pilot

SoccerViz now has a soccer-fine-tuned RF-DETR Medium checkpoint with verified
training and external development results. It completed the fixed five-epoch
pilot on a 225-image official TRAIN corpus. Results are mixed, and this
checkpoint should remain an experiment rather than replace the current control.
Two games supplied optimization images and a third selected the checkpoint.
The existing 450-frame validation benchmark stayed outside both operations.

## Frozen data

| Partition | Sequence | Official game ID | Images |
| --- | --- | --- | --- |
| Training | SNGS-060 | 4 | 75 |
| Training | SNGS-097 | 6 | 75 |
| Internal development | SNGS-170 | 9 | 75 |

All images originate from official `train.zip` at SoccerNet/SN-GSR-2024 revision
`3cc710eb6d53a23350a3d2863311c1c0a2e645d7`. Source frame IDs are 1, 11, ..., 741:
75 sampled images spanning each 30-second clip. Game IDs 2, 3 and 5, which
supply the external development benchmark, were excluded before training.

The first metadata probe encountered many annotations from the same game and
reached its 80 MB ceiling. A subsequent sparse endpoint check identified the
third game. Selection used game metadata only; no model scores informed it.
The retained selection inventory documents this. The reusable discovery code
now reads small JSON metadata prefixes before fetching selected annotations.

Conservative accounted download is **160,599,092 bytes**, including the full
80 MB upper bound for the initial probe, below the 250 MB cap. Entire archives
were not downloaded; member ZIP CRC32 and local SHA256 were verified.

Frozen files:

- `artifacts/detector-training/protocol.json`, SHA256
  `1b8fed76eb7deb2a62aacedac3c8f3320c478ffee4dee2b5660e37d89dbcaced`.
- `artifacts/detector-training/manifest.json`, SHA256
  `828750c5f00fc6e802b87b5af0bfbd3124521e410cb337dc14a296f1b20c7814`.

Both exports use the same pixels and annotations. COCO categories 1–4 are
`player`, `goalkeeper`, `referee`, `ball`; YOLO uses the corresponding 0–3.
RF-DETR's custom COCO loader remaps categories 1–4 to output labels 0–3 and
shares the training mapping with internal validation. No synthetic grouping or
background category is included in the annotations.

Training has 2,178 player, 104 goalkeeper, 208 referee and 147 ball boxes.
Internal development has 1,029 player, 71 goalkeeper, 100 referee and 59 ball
boxes. There is no separate source guarantee of exhaustive ball labeling.

The exporter supports official ignore polygons, `other` objects and explicit
ignore/difficult/crowd flags. It masks those regions consistently in both
formats and records omitted annotations. No selected frame needed masking,
so all exported image bytes match the original public images.

## Fixed experiment

RF-DETR is pinned to 1.10.1, with Medium's 576-pixel input resolution and the
public COCO initialization checkpoint SHA256
`749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8`.

The pilot runs exactly five epochs, batch size 2, gradient accumulation 4
(effective batch 8), seed 20260907, AdamW at learning rate 0.0001 and encoder
learning rate 0.00015, weight decay 0.0001, and EMA. Multiscale training and scale
jitter are disabled for this bounded first experiment. Native torchvision
augmentation and checkpoint callbacks are retained. Early stopping and external
logging are disabled. There is no hyperparameter search.

The training worker preserves the existing NVIDIA PyTorch/Torchvision versions
and pins RF-DETR's training extras separately:

```sh
docker build -f envs/detector-training/Dockerfile \
  -t soccerviz-detector-training:20260907 .
```

First run exactly one optimizer step in a separate output directory:

```sh
python scripts/training/train_detector_candidate.py \
  --corpus artifacts/detector-training \
  --checkpoint /models/rf-detr-medium.pth \
  --checkpoint-sha256 749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8 \
  --out /outputs/rf-soccer-smoke --smoke-step
```

Then run the same command with `--out /outputs/rf-soccer-pilot` and omit
`--smoke-step`. The five-epoch run starts fresh from the public checkpoint, not
from the smoke update. Existing or partial output directories are preserved;
use a new directory for a retry.

The runner uses RF-DETR's native Lightning data/model/trainer components.
The native EMA, COCO metric and checkpoint-selection callbacks remain installed.
Only the internal development game selects `checkpoint_best_total.pth`.
An additional evidence callback records optimizer steps, finite loss/gradients
and actual classification-head weight changes. The saved best model is loaded
again to check that its four class names and class count round-trip correctly.

## Validation and interpretation

Eleven local tests pass, including split/game leakage rejection, fixed training
budget enforcement, coordinate round-trip, difficult flags, ignore masking,
complete shared exports, class order and checksum failure on changed images.
All 225 exported images have been verified locally for hashes and dimensions.
The separate Spark training image has passed native training import checks.
The initial smoke reached native dataset/class validation but failed during
trainer setup because RF-DETR's device-selection helper rejected a list of GPU
indices. The runner now passes `devices=1` for the sole GB10 and rejects other
GPU indices explicitly. Setup failures receive a separate `failure.json` with
no verified optimizer update. The initial failed run is preserved; it is not
training evidence. The corrected one-step smoke passed on Spark: 18.82 seconds,
about 1.92 GB peak allocated GPU memory,
finite gradients, a verified classification-head update and a successful
four-class checkpoint reload.

The separate full pilot completed **five epochs and 95 optimizer updates**.
Its runner elapsed time was 207.63 seconds and peak allocated GPU memory was
2,202,406,912 bytes. The evidence records finite gradients, a changed
classification-head hash, all epoch-end hooks 0–4, and optimizer steps 1–95.
The last logged training loss was 0.2988; this is not a convergence claim.

The selected model is
`artifacts/models/rf-detr-soccer-pilot/checkpoint_best_total.pth`:

- SHA256: `4e5cb479a57e48fc13bc69361bf7321d5f09d6097dcf97c95433b3fe1b176f18`.
- Size: 133,806,681 bytes.
- Native selected source: EMA, chosen on the internal development game's box
  mAP. The stored internal selection mAP is 0.23537 across the four soccer classes.
- Reloaded classes: `player`, `goalkeeper`, `referee`, `ball`; `num_classes=4`.

The local checkpoint hash matches the training evidence, checkpoint class
round-trip report, external predictions and external metric report. Saved
checkpoint metadata names the training corpus and explicitly records that
external development was not used for selection. The audited corpus and
protocol hashes match their frozen originals. The audit is saved at
`artifacts/detector-training/rf-pilot-audit.json`.

## External development comparison

Both models below used the same 450 source frames, native 576-pixel resolution,
candidate capture floor 0.01 and evaluation operating threshold 0.25. Person
matching uses image-box IoU at least 0.50; ball-center matching uses at most
10 source pixels. Box AP uses the same captured score floor. The tracking
comparison has identical tracker and comparison-code hashes, uses the existing
`TrackletAssociator` with appearance disabled, and resets at each sequence.

| Metric | COCO-pretrained RF-DETR Medium | Soccer pilot |
| --- | ---: | ---: |
| Person precision | 91.43% | 79.21% |
| Person recall | 93.69% | 94.64% |
| Recall on 157 small person boxes | 28.66% | 63.06% |
| Ball-center precision | 44.35% | 81.90% |
| Ball-center recall | 25.46% | 21.99% |
| Two-class box mAP, IoU .50:.95 | 28.94% | 28.39% |
| Person tracking HOTA | 0.7114 | 0.6224 |
| Person tracking IDF1 | 86.54% | 75.01% |

The pilot found 99 of 157 small people versus 45 for the pretrained model,
but total person false positives rose from 766 to 2,166. Its ball detections
were more selective: 95 correct centers and 21 false positives, compared with
110 correct and 138 false positives for the pretrained model. The missed-ball
count therefore increased from 322 to 337. The gains do not justify promoting
this pilot with its current precision and tracking regressions.

The soccer pilot has a different class ontology: three person roles plus ball,
while the control uses generic person and sports-ball classes. Evaluation
collapses the soccer roles into person, so these results are not an isolated
architecture comparison. Role-label accuracy was not separately established.

The model/evidence bundle is at `artifacts/models/rf-detr-soccer-pilot/`.
External artifacts are
`artifacts/vision-benchmark/rf-medium-soccer-pilot-predictions.json`,
`rf-medium-soccer-pilot-results.json`, and
`rf-medium-soccer-pilot-tracking-results.json` in that same benchmark directory.
The matching generic controls are `rf-medium-coco-results.json` and
`rf-medium-tracking-verified-results.json`; the latter also matches the pilot's
comparison-code hash exactly.

This is a small pilot with heavily correlated images and only one internal
development game. Five epochs do not demonstrate convergence or broad transfer.
The external corpus has already informed project decisions and is not an
untouched test set. No tuning or checkpoint reselection followed these mixed
external results within this pilot.

The matching YOLO-format dataset is at `artifacts/detector-training/yolo/data.yaml`
for the separately tracked YOLO26 Medium comparison.

## Roboflow corpus pilot, 2026-09-07

Motivation: the copyleft rule in the [licence census](../guides/public-data.md#licence-census-and-the-copyleft-rule)
makes the SoccerNet-trained pilot above an evaluation artefact only. This run
repeats the identical five-epoch RF-DETR Medium protocol on the CC BY 4.0
Roboflow Universe `football-players-detection-3zvbc` v10 export (312
source-resolution 1920x1080 images, 15 source videos).

### Corpus

Roboflow's own train/valid/test split leaks every one of the 15 videos across
splits, so it is discarded. The corpus is re-split by video id with a fixed
rule (video ids sorted descending are held out until at least 15% of images
are held; no model scores used): 11 videos / 252 images train, 4 videos / 60
images internal development. Class order is the fixed player, goalkeeper,
referee, ball; Roboflow's alphabetical ids are remapped. Training boxes:
5,033 player, 189 goalkeeper, 591 referee, 222 ball.

Frozen files, tracked under `results/experiments/detector-training-roboflow-v10/`:
`protocol.json` (schema `detector-training-protocol/roboflow-v1`) and
`manifest.json`, SHA256 `568e29f943e468d5db7f65f83453117c602329eda974708d658c3efdc12f7233`.
The demo clip `2e57b9` used by `artifacts/video/demo-v2` is one of the eleven
training videos; the external 450-frame benchmark is SoccerNet and shares no
source with this corpus.

### Training

Same image, checkpoint, hyperparameters and seed as the SoccerNet pilot. The
run completed **five epochs and 160 optimizer updates** in 208.4 s with peak
allocated GPU memory 2,199,166,976 bytes; finite gradients and a changed
classification head are recorded in `training-evidence.json`. Internal
development box mAP@.50:.95 (EMA) rose 0.255, 0.473, 0.523, 0.528, 0.545 over
the five epochs, still climbing at the end. Selected checkpoint
`artifacts/models/rf-detr-roboflow-v10/checkpoint_best_total.pth`, SHA256
`274b3d1c1829a6a595cb052e5b2b035f6253feeb7e537bd6a47a235977fbbeb6`,
133,806,681 bytes, four classes round-trip.

### External development comparison

Same 450 frames, 576-pixel input, capture floor 0.01, operating threshold 0.25,
same fixed tracker. The comparison script's own hash differs from the earlier
rows only because its repo-root path changed in the restructure; the tracker
code hash is identical.

| Metric | COCO-pretrained | SoccerNet pilot | Roboflow v10 pilot |
| --- | ---: | ---: | ---: |
| Person precision | 91.43% | 79.21% | 85.41% |
| Person recall | 93.69% | 94.64% | 94.78% |
| Recall on 157 small person boxes | 28.66% | 63.06% | 61.15% |
| Person false positives per frame | 1.70 | 4.81 | 3.14 |
| Ball-center precision | 44.35% | 81.90% | 27.00% |
| Ball-center recall | 25.46% | 21.99% | 22.69% |
| Two-class box mAP, IoU .50:.95 | 28.94% | 28.39% | 26.34% |
| Person tracking HOTA | 0.7114 | 0.6224 | 0.6608 |
| Person tracking IDF1 | 86.54% | 75.01% | 78.22% |

Against the SoccerNet pilot, the Roboflow pilot keeps the small-person gain,
cuts person false positives by a third, and recovers four HOTA points. It still
trails the COCO-pretrained control on precision, box mAP and tracking, and its
ball detections are the least selective of the three (264 unmatched ball
centres). Five epochs on 252 images with mAP still rising is not a converged
model; the next round should train longer on the larger Universe sets before
another comparison, not tune against this development set.

Artifacts: `artifacts/vision-benchmark/rf-medium-roboflow-v10-{predictions,results,tracking-results}.json`
and the model bundle in `artifacts/models/rf-detr-roboflow-v10/`.

## Roboflow corpus, 50 epochs, 2026-09-08

Same corpus, split, image, initialisation checkpoint and hyperparameters as
the five-epoch pilot above; only `--epochs 50`. Launched through
`scripts/spark/spark_detector_training.py train --name rf-roboflow-v10-e50
--epochs 50` on Spark (container 00:50 to 02:01 EDT, 1,600 optimizer updates,
peak 2.2 GB CUDA). Bundle in `artifacts/models/rf-roboflow-v10-e50/`
(per-epoch `.ckpt` files stay on Spark); benchmark artefacts
`artifacts/vision-benchmark/rf-medium-roboflow-v10-e50-{predictions,results,tracking-results}.json`.

### Training

Internal-development EMA box mAP (IoU .50:.95) rose from 0.263 at epoch 0 to a
peak of 0.547 at epoch 9, then drifted down to 0.525 by epoch 49: 252 training
images saturate well before 50 epochs. The trainer's own selection kept
`checkpoint_best_total.pth` (SHA256 `e5a740c7…`), and that file is what the
benchmark scored. The chain's first retrieve failed because the container
wrote that file mode 0600 as root; the launcher now chmods its outputs at
container exit (commit 3b8e836).

### External development comparison

| Metric | COCO-pretrained | SoccerNet pilot | Roboflow v10 pilot | Roboflow v10, 50 epochs |
| --- | ---: | ---: | ---: | ---: |
| Person precision | 91.43% | 79.21% | 85.41% | 88.65% |
| Person recall | 93.69% | 94.64% | 94.78% | 94.53% |
| Recall on small person boxes | 28.66% | 63.06% | 61.15% | 59.87% |
| Person false positives per frame | 1.70 | 4.81 | 3.14 | 2.34 |
| Ball-center precision | 44.35% | 81.90% | 27.00% | 26.19% |
| Ball-center recall | 25.46% | 21.99% | 22.69% | 25.46% |
| Two-class box mAP, IoU .50:.95 | 28.94% | 28.39% | 26.34% | 26.91% |
| Person tracking HOTA | n/a | 0.6224 | 0.6608 | 0.6789 |
| Person tracking IDF1 | n/a | 75.01% | 78.22% | 81.27% |

The 50-epoch model is the best fine-tune on every person metric: precision
88.65% (from 85.41%), false positives per frame 2.34 (from 3.14), HOTA 0.679
(from 0.661), IDF1 81.27% (from 78.22%), while keeping twice the COCO
control's small-person recall. It still trails the COCO-pretrained control on
precision, box mAP and HOTA (0.679 vs 0.711 at the same 576 px), and its ball
head is no better than the pilot's: ball-center precision 26.19% with 309
unmatched ball centres. Two-class mAP 26.91% is person 50.34% and ball 3.48%
(COCO control: 54.32% and 3.56%).

Promotion rule set 2026-09-08 (beat the COCO control on HOTA with person
precision above the pilot): not met on HOTA. Against the current workflow
default `rf-soccer` (the SoccerNet pilot, HOTA 0.622, evaluation-only) it is
better on every metric except small-person recall and ball precision, and it
is the only role-aware detector in the tree cleared for shipping.

Next lever, unchanged: train on the larger CC BY Universe sets (the 5,761-image
`soccer-players-xy9vk`, class-remap census first) and give the ball its own
data. Do not tune against the 450-frame development set.
