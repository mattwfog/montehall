# YOLO26 Medium soccer training pilot

`scripts/training/train_yolo_candidate.py` prepares a fixed five-epoch YOLO26 Medium
comparison using the same independently partitioned soccer data as the RF-DETR
pilot. It uses the existing `soccerviz-detectors:20260907` Spark image and native
Ultralytics 8.4.143 training APIs. No new GPU job is launched by creating this
runner; completed training is established only by its saved evidence file.

## Data and initialization

The shared frozen corpus is `artifacts/detector-training`: 150 training images
from source games 4 and 6, and 75 internal-development images from source game 9.
All images come from the official SoccerNet TRAIN split. External development
games 2, 3 and 5 are excluded. Both candidates use identical exported image bytes
and labels, including the same ignore masks. Class order is fixed:
**player, goalkeeper, referee, ball** (YOLO IDs 0, 1, 2, 3).

The runner calls `detector_training.verify_corpus` before and after training.
It passes the absolute path of the existing `yolo/data.yaml` to Ultralytics,
then verifies native path resolution. Train and validation loaders must contain
exactly the frozen 150 and 75 image paths; silently dropped labels/images or a
different validation directory cause failure. Test, minival and dataset-download
entries are forbidden.

Initialization is the public `yolo26m.pt` checkpoint, downloaded through native
Ultralytics from
[its official release asset](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt).
Its size is 44,255,705 bytes and required SHA256 is
`401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`.
The checkpoint retains pretrained visual features; the native model adapts to
four soccer classes. Automatic class-name remapping is disabled explicitly.

## Fixed recipe

| Setting | Value |
|---|---|
| Epochs / image size | 5 / 576 pixels |
| Batch / nominal batch | 2 / 8; native accumulation 4 |
| Seed / workers | 20260907 / 2 |
| Optimizer | AdamW, learning rate 0.0001, weight decay 0.0001, beta1 0.9 |
| Schedule / warmup | Constant LR (`lrf=1`), no warmup |
| Precision | Float32, AMP disabled |
| Early stopping / search | Disabled; no hyperparameter search |
| Validation / checkpoint selection | Internal-development partition only; native best validation fitness |
| Native loss weights | Box 7.5, class 0.5, DFL/L1 1.5 |

Augmentation settings are explicit: HSV hue/saturation/value 0.015/0.7/0.4,
horizontal flip probability 0.5, and translation fraction 0.1. Rotation, vertical
flip, random scale, shear, perspective, BGR swapping, multiscale, mosaic, mixup,
CutMix and copy-paste are disabled. Classification-only automatic augmentation
and erasing are also disabled. Every resolved argument is retained in
`run-config.json` and the native run's `args.yaml`.

The candidates share data, image size, epoch budget, batch/effective batch and
base learning rate. Their native augmentation, loss/head architecture, optimizer
parameter grouping, scheduler/EMA implementation and RF encoder learning rate
still differ. **This is not an architecture-controlled comparison**, and five
epochs on 225 correlated images do not establish convergence or generalization.

## Execution and evidence

Mount only the frozen training corpus at `/data`, the source/scripts at `/work`,
the public initialization checkpoint at `/models`, and a fresh results directory
at `/results`. **Do not mount the external 450-frame development benchmark into
the training container.** It is not used for fitting, early stopping, or selecting
the best checkpoint.

```sh
# Inside the existing GPU container:
python scripts/training/train_yolo_candidate.py \
  --corpus /data --checkpoint /models/yolo26m.pt \
  --out /results/yolo26m-pilot --device cuda:0
```

Only CPU or the single GPU `cuda:0` is accepted; another GPU index is never
silently remapped. The required initialization hash is built in. An optional
`--checkpoint-sha256` argument must equal that recorded digest. Existing completed
or partial output directories are preserved, and the native run must use the
requested `native/` subdirectory without automatic name incrementing.

Ultralytics' `optimizer_step` callback is reserved and is not emitted by the
default trainer in this version. The worker therefore attaches **PyTorch optimizer
pre/post hooks** to count actual updates after gradient accumulation. Every update
checks finite gradients and losses, records learning rates, and hashes the
four-class output weights. Per-batch loss checks and exact five-epoch completion
are required. Native OOM recovery cannot silently reduce the fixed batch size or
replace the audited optimizer: either change fails the run.

Successful artifacts include:

- `run-config.json`: exact arguments, corpus and source-code hashes, initialization
  provenance, class names, package versions, and comparison limitations.
- `optimizer-steps.jsonl`: actual optimizer updates, losses, learning rates and
  classification-head hashes.
- `native/results.csv`, `native/args.yaml`, `native/weights/best.pt`,
  `native/weights/last.pt`, and epoch checkpoints.
- `training-evidence.json`: five completed epochs, actual optimizer count, changed
  classifier hashes, finite-loss/gradient evidence, checkpoint hashes, and native
  validation events. Final revalidation of `best.pt` is labeled separately from
  the five epoch validations.

The best checkpoint is reloaded and its four-class mapping checked. Completion
requires a changed classification head, at least one actual update, all five
epochs, and both best/last checkpoints. `failure.json` records setup/training
failure without claiming a completed model; partial checkpoints and logs may
remain available for diagnosis.

## Verification

Nineteen focused tests pass, covering frozen optimizer/batch/augmentation settings,
class ordering, native train/dev path restrictions, external-test rejection,
partition membership, invalid losses, incomplete training/update evidence, fixed
checkpoint provenance and GPU selection. Training APIs were inspected from the
actual existing Spark image, including the reserved callback behavior and native
data YAML resolution. A CPU-only check in that same image accepted the complete
fixed argument set, loaded the pinned public Medium model, and constructed its
four-class native architecture. All six audited output-weight tensors were found
in the one-to-many and one-to-one classification branches; the base head reports
`Detect`, `reg_max=1`, and scale `m`.

The prepared runner must be judged from its eventual completed GPU artifacts.
It does not by itself establish that a fine-tuned YOLO26 checkpoint exists or
improves the current detector.


## Executed pilot and independent development results

The parent executed this runner on Spark successfully: five epochs, 375
microbatches, 93 verified optimizer updates, finite losses/gradients, and changed
classification-head weights. Runtime was 131.0 seconds. The selected local
checkpoint is `artifacts/models/yolo26m-soccer-pilot/best.pt`, SHA256
`02b42ab6daa21d5e8356b3da9707714f60be558dec8fbb5e865523f31daa2a0b`.
Its hash matches both training evidence and external inference. Four classes
were verified after reload; best.pt selection used only the 75-frame internal
development partition. The final extra validation event is not a sixth epoch.

On the separate 450-frame development benchmark at score >=0.25, the pilot
produces person precision/recall 88.8%/89.5%, HOTA 0.637, and ball-center
precision/recall 81.0%/7.9%. The general COCO checkpoint at native 640 gives
92.0%/91.2%, HOTA 0.701, and 59.5%/26.2%, respectively. Small-person recall
increases from 8.9% to 38.2%, but overall detection/tracking and ball recall
regress. This is not a promotion candidate on the current evidence.

The pilot uses fixed 576-pixel training/inference, whereas the COCO reference
uses native 640. RF and YOLO training recipes differ despite sharing the corpus
and five-epoch budget. Do not attribute these differences solely to architecture.
Full reports are under `artifacts/vision-benchmark/yolo26m-*-results.json`;
local training evidence and optimizer records accompany best.pt. Independent
parent checks are saved in `artifacts/detector-training/yolo-pilot-audit.json`.
