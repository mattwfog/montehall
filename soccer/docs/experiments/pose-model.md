# ViTPose++ body pose experiment

The adapter runs a pinned top-down 2D pose model on every predicted person box
in the frozen 450-frame benchmark and emits 17 COCO keypoints in source pixels.
It exists for the offside question: the law counts any part of the head, body or
feet, never the arms, so the pose layer must name those points before the
calibration layer can project them onto the pitch. This worker stops at
keypoints. It does no projection, no temporal smoothing and no identity.

Learning experiment, by design: backends are a table of pinned Hugging Face
revisions, so a stronger or different model (ViTPose++ huge, a mesh model) is
one more `BACKENDS` entry in `src/soccerviz/candidates/pose_model.py`, not a
new pipeline.

## Backends

| Name | Repo | Revision | Weights SHA-256 | License |
| --- | --- | --- | --- | --- |
| `vitpose-plus-base` | `usyd-community/vitpose-plus-base` | `92be54d7a29e42fad47b6e2ca01dd9e685a61e0d` | `640225e4a9dd544239f1da0ae36af865a6e76e21e5f33a12656aa7b77fa5a5fa` | apache-2.0 |
| `vitpose-plus-huge` | `usyd-community/vitpose-plus-huge` | `9f36d7aec1800d23e97f10c2e74393aee92aa53f` | `0ecb49f1ab0b18cc2f18446b8100442cec88bd99dd53779ad5a7f8c71aa08506` | apache-2.0 |

Both are ViTPose++ mixture-of-experts checkpoints; the adapter always selects
expert 0 (COCO). Loading is offline from a local snapshot and refuses a
weights file whose hash differs from the pin. Only `vitpose-plus-base` has
been run; the huge entry is pinned but unmeasured.

Considered and not yet added: `facebook/sam-3d-body-dinov3` (Meta SAM 3D Body,
full-body mesh on Meta's own rig, SAM license, gated; needs a conda
environment with detectron2 and a 2.1 GB checkpoint plus a 0.7 GB rig file)
and `facebook/sapiens-pose-1b` (308 keypoints, cc-by-nc-4.0). Neither has a
`transformers`-native loader, so each would be its own backend class.

## Inputs

Frozen image manifest SHA-256
`75200edcfe9851a5c68a68bfe36ef320755d90d33bd4ece98590c8d31df08795`.
Boxes come from the completed soccer YOLOv8x detector run on the same manifest,
`artifacts/vision-benchmark/yolov8x-soccer-predictions.json`; every detection
labeled `person` is used, with its predicted role carried through unchanged.
No ground truth is read at inference time. Each source image is hash-checked
before it is read.

## Actual run, ViTPose++ base

Run locally on Apple MPS (torch 2.14.0, transformers 5.1.0, numpy 2.5.3) with
`.venvs/pose`, float32, 256×192 crops, all 450 frames.

| Measure | Value |
| --- | ---: |
| People (predicted boxes) | 9,388 |
| By predicted role | 8,882 player, 503 referee, 3 goalkeeper |
| Keypoints inside their box (10% margin) | 94.4% |
| People with a confident foot (score ≥ 0.3) | 93.0% |
| Ankles below hips when both confident | 99.99% |
| Confident-rate per keypoint (score ≥ 0.3) | 91.6% (left wrist) to 95.7% (right hip) |
| Median wall time per frame, MPS | 400 ms |
| Session wall time | 183 s |

These are sanity rates, not accuracy. The benchmark carries no keypoint labels,
so nothing here says a keypoint is on the right limb; it says the model's output
is placed and ordered like a standing person nearly every time. The 5.6% of
keypoints outside their box is expected for tight detector boxes and occluded
limbs and has not been inspected visually.

Keypoint scores are heatmap peaks. They are stored raw as `heatmap_peaks`
and clipped to [0, 1] as `scores`; the clip is explicit and the value is
uncalibrated. The 0.3 floor was fixed before the run and not tuned.

Artifacts: `artifacts/integrations/pose-vitpose-plus-base/predictions.json`
(SHA-256 `b7f29eb61dc25c7aa7bf9925ed75820e4247dfd0406074a46cc30960c9959ffd`,
29.1 MB, gitignored) and the tracked summary
`results/experiments/pose-vitpose-plus-base-summary.json`.

## Commands

```sh
uv venv --python 3.12 .venvs/pose
uv pip install --python .venvs/pose/bin/python torch -r envs/pose-model/requirements.txt

# One-time snapshot at the pinned revision (about 500 MB); later loads are offline.
.venvs/pose/bin/python -c "from huggingface_hub import snapshot_download as s; \
  s('usyd-community/vitpose-plus-base', revision='92be54d7a29e42fad47b6e2ca01dd9e685a61e0d', cache_dir='artifacts/models/hf')"

PYTHONPATH=src HF_HUB_OFFLINE=1 .venvs/pose/bin/python scripts/benchmarks/run_pose_benchmark.py infer \
  --manifest artifacts/vision-benchmark/manifest.json \
  --detections artifacts/vision-benchmark/yolov8x-soccer-predictions.json \
  --out artifacts/integrations/pose-vitpose-plus-base --device mps

PYTHONPATH=src .venv/bin/python scripts/benchmarks/run_pose_benchmark.py summarize \
  --predictions artifacts/integrations/pose-vitpose-plus-base/predictions.json \
  --out results/experiments/pose-vitpose-plus-base-summary.json

.venv/bin/python -m pytest tests/candidates/test_pose_model_adapter.py -q
```

`infer` appends each finished frame to `frames.jsonl` in the output directory
and writes `predictions.json` only when every requested frame is present. A
rerun with the same `--out` resumes; an existing `predictions.json` is never
overwritten. `--limit N` runs a prefix and marks the output incomplete.
`--device cuda:0` and `envs/pose-model/Dockerfile` cover the Spark worker;
that path has not been run.

## Validation status

Seven local tests pass: box clipping, keypoint validation, arm exclusion and
foot requirement in `body_extremes`, hash-checked box selection, per-frame
persistence with resume and overwrite refusal, summary rates, and refusal of
detector output not bound to the manifest. The runner test uses a fake backend;
the real-model evidence is the run above.

## Next

1. Project `body_extremes` through the calibration homography and compare the
   lowest-foot pitch position against the box-bottom proxy the pipeline uses now.
2. Run `vitpose-plus-huge` on the same boxes and diff keypoints against base;
   disagreement is the only accuracy signal available without labels.
3. Add SAM 3D Body as a backend once the gated download is approved, for
   3D joint positions rather than image-plane points.
