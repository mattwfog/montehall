# WASB temporal ball experiment

The adapter runs the public soccer WASB checkpoint on consecutive video frames.
It emits ball centers in source pixels, with no fabricated bounding boxes.
The independent evaluation uses the frozen benchmark's 10-pixel center criterion.

## Reproduction

Upstream: [WASB-SBDT](https://github.com/nttcom/WASB-SBDT), commit
`923462cacdeb3353b84ddebdedb3f4b7a8553b0f`.
The aggregate hash of all 98 Python/YAML source files is
`5c8e692d79d39244c60d17a4e7372227411d64514f7532690efe7330ca7b464f`.
The adapter checks this fingerprint before importing upstream code. macOS
AppleDouble sidecars are omitted only when their binary magic and corresponding
real file identify them as resource metadata; actual source files remain pinned.

The public soccer checkpoint is linked in the upstream
[model zoo](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/MODEL_ZOO.md).
Its SHA256 is
`d0369572807c2baf751880d6cdf3cce9fc6283fa8d153f18af6baf4e64d2646c`.
Loading checks that hash and uses `torch.load(..., weights_only=True)` plus strict
state-dictionary loading. No fine-tuning is performed by this runner.

Build the separate worker from the project root:

```sh
docker build -f envs/temporal-ball/Dockerfile -t soccerviz-ball:20260907 .
```

The image uses NVIDIA PyTorch 26.01. Exact constraints protect the base image's
Torch/Torchvision builds from replacement. WASB's native trackers still use
`np.Inf`; this worker uses NumPy 1.26.4 and OpenCV 4.11 rather than changing the
upstream source or other project environments.

Inside the worker, with the project mounted at `/work` and checkpoint at
`/models/wasb-soccer.pt`:

```sh
python scripts/benchmarks/run_temporal_ball_benchmark.py \
  --manifest artifacts/vision-benchmark/manifest.json \
  --upstream artifacts/upstreams/wasb \
  --checkpoint /models/wasb-soccer.pt \
  --device cuda:0 --tracker peak \
  --path-prefix "$PWD"=/work \
  --out /outputs/wasb-peak-predictions.json
```

For a six-frame execution check, add `--limit 6`. Such output is explicitly
marked incomplete and must not be presented as a full benchmark result.
Use `--tracker online` for the upstream 300-pixel displacement gate, with a
separate output file. Existing prediction files cannot be overwritten.

After retrieving the prediction file, evaluate it in the local evaluation
environment:

```sh
.venvs/evaluation/bin/python -m soccerviz.candidates.vision_benchmark evaluate \
  --manifest artifacts/vision-benchmark/manifest.json \
  --predictions artifacts/vision-benchmark/wasb-peak-predictions.json \
  --output artifacts/vision-benchmark/wasb-peak-metrics.json
```

## Native model behavior and deliberate choices

The adapter imports the upstream HRNet, `get_transform`,
`build_img_transforms`, and `TracknetV2Postprocessor` directly. Three RGB images
are warped to 512x288 with the upstream affine transform, normalized using
ImageNet statistics, and concatenated into nine input channels. The network
returns three sigmoid heatmaps. Native connected components at threshold 0.5
produce weighted centroids, transformed back to source pixels.

The upstream blob score is summed heatmap activation, which can exceed 1.
It remains available as `native_blob_score`. The bounded `score` is the peak
sigmoid activation within that component, explicitly labeled uncalibrated.
The shared evaluator's 0.25 threshold does not equate these heatmap activations
with YOLO confidence. All native candidates already pass the fixed 0.5 heatmap
threshold. The native tracker selects using the original blob score.

`--tracker peak` uses upstream `IntraFramePeakTracker` to select one ball per
frame, matching the existing pipeline's single-candidate policy. It isolates
the detector comparison and does not reproduce the paper's default online
tracking behavior. `--tracker online` uses upstream `OnlineTracker` unchanged
and resets it at segment boundaries. Neither mode estimates a bounding box.

The default native Step 3 policy consumes nonoverlapping triples and assigns
heatmap slots 0, 1, 2 to those same source frames. Their future context is 2, 1, 0
frames respectively: at 25 fps, up to 80 ms lookahead. This is an offline/delayed
comparison, not a causal real-time result. Original fps/timestamps, context
frame IDs, hashes, output slots, and affine transforms are retained.

Context never crosses known sequence boundaries, frame/time gaps, missing
images, image-size or shot-ID changes, explicit cuts, or detected histogram
cuts. Remaining one/two-frame tails receive explicit empty predictions and
count in the full evaluation. Frames are never repeated to fill a triple.
An optional `--cut-boundaries` file contains `{sequence: [frame_ids]}`, where
each ID is the first frame after a cut. The histogram guard uses a fixed 0.65
threshold; it is a heuristic, so cuts it fails to identify remain a limitation.

## Validation status

Twelve local tests pass, covering frame gaps, timestamp gaps, sequence/cut
boundaries, unavailable images, tail coverage, heatmap/output alignment,
lookahead, source fingerprints, native center/score handling, and output
preservation. A fake model is used for the runner's alignment test; this is
not evidence of pretrained model accuracy or GPU compatibility.

Local preflight of the frozen 450-frame corpus produced 150 complete triples.
The histogram guard identified no cuts. Maximum adjacent histogram distances
were 0.068 (SNGS-021), 0.082 (SNGS-045), and 0.113 (SNGS-089); all were below the
fixed 0.65 threshold. This establishes available context, not correctness.

The public checkpoint subsequently completed the actual NVIDIA GB10 run on all
450 frames. It emitted 66 ball centers; 26 matched an annotated ball within
10 source pixels, with 40 false positives. Against 432 annotated balls this is
6.02% recall and 39.39% precision. Mean error among the 26 matched centers was
4.05 pixels; this conditional error excludes the many misses.

Measured inference/read/postprocess time was 15.49 seconds (29.06 frames/s),
excluding 2.89 seconds preflight, 2.26 seconds initialization and 0.46 seconds
warmup. The measured median forward pass was 32.00 ms per three-frame window.
PyTorch peak allocated memory was 275,631,104 bytes. All 450 frames had complete
context, so low recall reflects weak detector transfer rather than missing
triples. This candidate should not replace the current ball detector.

Artifacts: `artifacts/vision-benchmark/wasb-peak-predictions.json` and
`artifacts/vision-benchmark/wasb-peak-results.json`. Center-only results have no
box AP; the evaluator records that metric as unavailable. These results use
the detector-only peak policy, not the upstream online displacement gate.
