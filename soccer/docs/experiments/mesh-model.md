# SAM 3D Body mesh experiment

Built, not yet run. The adapter wraps Meta's SAM 3D Body (single-image
full-body mesh recovery on the Momentum Human Rig) as a second pose backend
beside [ViTPose++](pose-model.md). It consumes the same frozen manifest and
detector boxes and emits, per predicted person, the 70 MHR keypoints in source
pixels and in camera-frame metres, the camera translation, the focal length and
the rig parameters that regenerate the mesh. Mesh vertices are optional
(`--save-vertices`, float16 `.npz` per frame) because 18k vertices per person
do not belong in JSON.

For offside this adds what 2D pose cannot: heels and toe tips as explicit
joints, and a 3D body extent, so the "furthest forward body part" can be argued
in metres rather than pixels once the calibration layer supplies the pitch.

## Run record

- **2026-09-08, first real run on Spark.** Meta approved the gated weights for
  the project's Hugging Face account; the snapshot downloaded on Spark and its
  hashes were recorded on first load (table below). The six-frame smoke on
  the frozen manifest with the YOLOv8x-soccer boxes produced 114 people,
  every one within 0.5 px on the 2D/3D reprojection check, feet below hips
  in 100% of people, 79.9% of keypoints inside their detector box, camera
  depth 21.8 to 57.0 m (p05 to p95) and body extent 1.46 to 1.80 m. Median
  10.2 s per frame with `inference_type="full"` on the GB10: 62 s for six
  frames. Summary: `results/experiments/mesh-smoke-summary.json`. The full
  450-frame run finished the same night (see Validation status).
- **Three gaps found by the smoke, all fixed in place.** (1) The upstream
  backbone calls `torch.hub.load("facebookresearch/dinov3", ...)` at unpinned
  `main`; the launcher now checks out DINOv3 at a pinned commit into the
  torch hub cache and sets `TORCH_HOME` so torch.hub uses it without a
  download. (2) The DINOv3 `hubconf` import closure needs `termcolor` and
  `ftfy`, neither listed by upstream; added to `envs/mesh-model/requirements.txt`
  (image rebuilt as `soccerviz-mesh:20260908`). `MultiScaleDeformableAttention`
  is also absent but its import is try-guarded upstream. (3) The config's
  `MODEL.IMAGE_SIZE` is a `[height, width]` list, not an int; the adapter
  metadata now records it as a list.
- **Checkpoint load warning.** Upstream reports one missing key,
  `backbone.encoder.mask_token`, which inference does not use, plus the
  `head_pose_hand.mhr.character_torch.parameter_limits.*` buffers reported
  as unexpected when Momentum is disabled (the TorchScript rig path is used).
- Still CUDA only: upstream hard-codes `.cuda()` in six places in
  `sam_3d_body/models/meta_arch/sam3d_body.py` (lines 1034, 1246, 1282,
  1289, 1292 and 1508) plus `recursive_to(batch, "cuda")` in the estimator.
  It is imported unmodified, so nothing here runs on the Mac.

## Pins

| Item | Value |
| --- | --- |
| Upstream | `https://github.com/facebookresearch/sam-3d-body` commit `b5c765a0d89d789985e186d396315e7590887b94` |
| Package fingerprint (46 `.py` files under `sam_3d_body/`) | `dfd0eeaac76c8da4fda281e61b67bc881046b4d904c560b47897e790dfd0f956` |
| Checkout | `artifacts/upstreams/sam-3d-body` (gitignored; clone at that commit) |
| Weights repo | `facebook/sam-3d-body-dinov3`: `model.ckpt` 2,109,129,346 bytes, `assets/mhr_model.pt` 696,110,248 bytes, `model_config.yaml` |
| Weights hashes (recorded on first load, 2026-09-08; Meta publishes none) | `model.ckpt` `b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf`; `assets/mhr_model.pt` `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc`; `model_config.yaml` `1012fc3f39cb5e90e3f8fbadf7bded31604bfafdce0321d17a7c1a2d3f08b88d`. Kept in `artifacts/models/hf/sam-3d-body-dinov3.hashes.json` on Spark; later loads must match. |
| DINOv3 backbone code (loaded by upstream through torch.hub, `pretrained=False`) | `https://github.com/facebookresearch/dinov3` commit `6876159a11b4df116f30f667f8c9888617df0751`, checked out into `artifacts/models/torch-hub/hub/facebookresearch_dinov3_main` |
| Spark image | `soccerviz-mesh:20260908`, id `f4cae82cf75f` |
| Licence | SAM License (Meta, 2025-11-19), read from the upstream `LICENSE`: royalty-free use, modification and redistribution under the same terms; acknowledgement in publications; trade-control clauses. Not copyleft. |

`sam-3d-body-vith` is pinned as a second backend name with no byte sizes
recorded; it has not been listed or downloaded.

## What the adapter checks

- The upstream package fingerprint before import, so a drifted checkout
  cannot run under the pinned name.
- Byte sizes of the two large files against the hub listing, then the
  recorded hashes.
- Every person's 2D keypoints against a re-projection of its 3D keypoints
  through `cam_t` and the focal length with the principal point at the image
  centre, the convention upstream applies at `sam3d_body.py:1635`. Disagreement
  above 0.5 px raises. This is the guard against misreading the coordinate frame.
- Output count equals box count; nonfinite values raise.

Inference uses `inference_type="full"` (body decoder, then both hand decoders),
which is the path that produces full-image 2D keypoints upstream. Camera
intrinsics are the upstream default from image size; no FOV estimator, no mask
conditioning, no detector of its own (boxes are the frozen detector output).

## Commands

Everything runs on Spark under `~/soccerviz-mesh` through
`scripts/spark/spark_mesh.py`, which follows the `spark_video.py` pattern
(tar over ssh, one detached container, tar back):

```sh
# Ship src, the mesh env, the frozen manifest/boxes/frames; clone the pinned
# upstream and DINOv3; download the gated snapshot with Spark's approved
# Hugging Face login; build the image. Idempotent.
uv run python scripts/spark/spark_mesh.py stage

# Detached container; frames.jsonl resumes, so rerunning the same --name continues.
uv run python scripts/spark/spark_mesh.py infer --name mesh-smoke --limit 6
uv run python scripts/spark/spark_mesh.py infer --name mesh-sam-3d-body-dinov3

# Log: ~/soccerviz-mesh/logs/<name>.log on Spark; container name mesh-<name>.
uv run python scripts/spark/spark_mesh.py retrieve --name mesh-sam-3d-body-dinov3

PYTHONPATH=src uv run python scripts/benchmarks/run_mesh_benchmark.py summarize \
  --predictions artifacts/integrations/mesh-sam-3d-body-dinov3/predictions.json \
  --out results/experiments/mesh-sam-3d-body-dinov3-summary.json
```

`infer` appends each frame to `frames.jsonl` and resumes on rerun; an existing
`predictions.json` is never overwritten. `summarize` reports sanity rates only
(keypoints inside box, feet below hips in the image, camera depth and body
extent quantiles); the benchmark has no 3D labels.

## Validation status

Seven local tests pass with a fake backend: projection convention, 2D/3D
consistency rejection, arm exclusion over all 70 joints, package fingerprint,
first-load hash recording and drift refusal, per-frame persistence with resume,
vertex saving and summary rates, and the CUDA-only refusal. The real six-frame
smoke above passed every runtime guard.

### Full 450-frame run, 2026-09-08

Container `mesh-mesh-sam-3d-body-dinov3` on Spark, 00:24 to 02:13 EDT
(6,498 s session, 450 frames, median 11.6 s per frame with
`inference_type="full"`, no vertices saved). Predictions
`artifacts/integrations/mesh-sam-3d-body-dinov3/predictions.json` (247 MB,
SHA256 `f8a56c69…`), summary
`results/experiments/mesh-sam-3d-body-dinov3-summary.json`. Boxes are the
same frozen detector predictions the pose candidate used, so the person
counts match ViTPose++ exactly: 9,388 people (8,882 player, 503 referee,
3 goalkeeper).

| Rate (plausibility, not accuracy) | Six-frame smoke | 450 frames | ViTPose++ base, same boxes |
| --- | ---: | ---: | ---: |
| Keypoints inside their box | 79.9% | 91.6% | 94.4% |
| Feet below hips in the image | 100% | 99.85% | n/a (93.0% confident foot) |
| 2D/3D reprojection consistency | passed | every person within 0.5 px | n/a |
| Camera depth, median (p05 to p95) | | 38.5 m (24.8 to 63.8) | n/a |
| Keypoint vertical extent, median (p05 to p95) | | 1.53 m (1.26 to 1.65) | n/a |

Reading: the mesh model's 2D keypoints sit inside the detector box slightly
less often than ViTPose++'s on identical boxes, at about 30 times the
per-frame cost (11.6 s versus 0.4 s on MPS). Its extra output, a metric-scale
body in a model-scaled camera frame, has a median standing height of 1.53 m
with a tight spread, which is plausible for players, but nothing in the
benchmark can score it. Until there is a consumer for 3D body pose (contact,
offside body-part geometry) or keypoint ground truth, SAM 3D Body stays a
built and measured candidate, not a pipeline component. The limitations list
in the summary is the authority: uncalibrated default intrinsics, detector
box errors folded in, no temporal smoothing, identity or pitch projection.
