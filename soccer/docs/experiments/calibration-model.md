# PnLCalib versus the existing pitch model

The experiment compares the released PnLCalib point/line networks and official
camera-refinement pipeline against SoccerViz's existing soccer YOLO pose model
and `geometry.calibrate`. It isolates camera calibration by projecting **oracle
image footpoints** onto independently supplied SoccerNet pitch coordinates.
This does not measure player detection, identity, or end-to-end game-state accuracy.

## Executed comparison

Both models completed the same frozen 45-frame Spark experiment, with all 45
frame outcomes retained. The evaluator used 876 finite source footpoint/pitch
pairs and independently reproduced the saved aggregate and per-sequence results.

| Measure | Existing YOLO calibration | PnLCalib |
|---|---:|---:|
| Accepted finite frame homographies | 38 / 45 (84.4%) | 45 / 45 (100%) |
| Projected target footpoints | 755 / 876 (86.2%) | 876 / 876 (100%) |
| Median error among projected points | 4.29 m | 0.48 m |
| 95th-percentile error among projected points | 10.35 m | 18.23 m |
| Mean error among projected points | 5.14 m | 37.89 m |
| All target footpoints within 2 m | 97 / 876 (11.1%) | 735 / 876 (83.9%) |
| Timed 45-frame processing loop | 3.90 s | 17.70 s |

The processing loop includes image reading/hash checks, inference and geometry;
model initialization is outside it, while the first frame includes cold-start
execution. These are short experiment timings, not full video-pipeline throughput.

**PnLCalib is promising for further integration, but its accepted output needs
additional validation before use in the product.** Its typical projection error
and all-target 2 m success rate improve substantially here, while its error tail
is worse. In SNGS-089, source frames 111 and 121 were accepted despite median
footpoint errors of approximately **490 m and 588 m**. Their own model-derived
reprojection residuals were only 0.86 px and 0.34 px, so a small fitting residual
alone would not catch these failures. Numerical coverage is not evidence that all
returned pitch positions are reliable. These failures remain in every reported
metric; no evaluation-label-based rejection or threshold change was applied.

YOLO rejected seven frames as `inconsistent_landmarks`; those frames contribute
121 unavailable footpoints. The models retain different declared acceptance
policies, so their conditional median errors have different denominators. The
all-target success rate uses the same 876 targets for both models.

The official line network and refinement code both executed, but the frozen
0.7867 line threshold yielded **zero accepted line candidates in 44/45 frames**
and one candidate in the remaining frame. This experiment primarily demonstrates
the keypoint-driven pipeline; it does not isolate or establish a benefit from
line detections or line refinement. Any later threshold experiment needs a new
named development protocol, preserving these results.

| Sequence | YOLO targets within 2 m | PnL targets within 2 m | PnL median / p95 error |
|---|---:|---:|---:|
| SNGS-021 | 37 / 255 | 237 / 255 | 0.43 / 2.13 m |
| SNGS-045 | 48 / 327 | 322 / 327 | 0.32 / 1.45 m |
| SNGS-089 | 12 / 294 | 176 / 294 | 0.92 / 585.46 m |

Completed artifacts are `artifacts/vision-benchmark/calibration/pnl-results.json`
and `yolo-results.json`, with their corresponding `*-predictions.json` files.
All store the unchanged source-manifest and experiment-protocol hashes.
These are validation-development measurements; public checkpoint pretraining
overlap with the SoccerNet GSR source games has not been ruled out.

## Keypoint index order, checked 2026-09-08

The Roboflow field-keypoint exports name their 32 keypoints
`01..13, 15..18, 20..32, 14, 19`, which suggests indices 13 onward are shifted
against the landmark table. They are not: index j is vertex j+1, as the table
assumes. Evidence: all 222 v12 training annotations are homography-consistent
under the index order (median residual 0.27 m) and only 74 under the name
order; the YOLO control re-run under the name order accepted 2 of 45 frames
instead of 38 (`yolo-name-order-hypothesis-{predictions,results}.json`, kept as
the record of the rejected hypothesis). `tests/core/test_geometry.py` pins
this against the export.

## Permissive heatmap calibrator, first run 2026-09-08

`pitch-hrnet-w32-v18` (timm HRNet-W32, ImageNet init, 32 Gaussian heatmaps at
960 px, spatial softmax cross-entropy, 60 epochs on the CC BY Roboflow v18
export: 255 train / 34 valid images). Best epoch 56: valid mean keypoint error
1.52 px, PCK@10px 1.00, in `artifacts/models/pitch-hrnet-w32-v18/`.

On the same frozen 45 frames (`heatmap-hrnet-w32-v18-{predictions,results}.json`)
it accepted 18 of 45 and projected 367 of 876 footpoints, median 1.17 m, p95
15.75 m; all 27 rejections are `inconsistent_landmarks`. Two causes, both
established from the saved diagnostics against PnLCalib's accepted homographies
(which place each of the 32 landmarks in or out of frame):

1. **Phantom landmarks.** The export marks about 17 of 32 keypoints per image
   invisible, and the trainer gave those weight 0, so their heatmaps were never
   supervised and came out as sharp as the visible ones. Per frame the count of
   landmarks above the 0.5 confidence cut was the in-frame count plus roughly
   half of the out-of-frame ones (SNGS-045 frame 1: 13 in frame, 22 confident,
   12 inliers). The RANSAC inlier count tracks the in-frame count almost
   exactly, so localisation is sound and the visibility signal was absent.
   Fix in place: invisible keypoints are now trained against the uniform
   distribution (`heatmap_loss`, `INVISIBLE_WEIGHT`), the trainer reports
   visible/invisible confidence separation per epoch and `best.pt` must clear a
   0.95 visibility-accuracy floor. Relaunched as `pitch-hrnet-w32-v18-vis`.
2. **Too few landmarks in a halfway-line view.** SNGS-021 has five of the 32
   landmarks in frame (the halfway line's top and both circle crossings, plus the
   circle's left and right extremes); three are collinear and the sixth, the
   halfway line's bottom, sits 424 px below the frame. `geometry.calibrate`
   needs six, so a keypoint-only model cannot accept any SNGS-021 frame under
   the current guard. PnLCalib solves those frames from line extremities. This
   bounds the heatmap approach at 30 of 45 on this sample until it also predicts
   line points or the guard admits five consistent landmarks (a change to the
   guard's contract, not made here).

## Permissive heatmap calibrator, visibility-supervised run 2026-09-08

`pitch-hrnet-w32-v18-vis`: same recipe with invisible keypoints trained toward
the uniform distribution (`heatmap_loss`, `INVISIBLE_WEIGHT` 1.0) and a 0.95
visibility-accuracy floor on `best.pt`. 60 epochs on the same 255 / 34 split,
3,780 optimizer steps, 3.3 h on the GB10 while sharing the GPU with the digit run.
Best epoch 50: valid mean keypoint error 1.61 px, median 1.35 px, PCK@10px 0.998;
visible-keypoint confidence median 0.84 against 0.001 for invisible ones, none of
the 594 invisible valid keypoints above the 0.5 cut, visibility accuracy 0.996.
Checkpoint `artifacts/models/pitch-hrnet-w32-v18-vis/best.pt`, SHA256
`528e1b26048a03ce51678f464f1092034c7126064d9ec16042b4ea6bf8a4f869`, registered as
a shipping-class candidate in `soccerviz.vision.checkpoint_licence`; not the
workflow default.

On the frozen 45 frames (`heatmap-hrnet-w32-v18-vis-{predictions,results}.json`,
predictions SHA256 `741eeb14…`): **30 of 45 accepted**, 621 of 876 footpoints
projected, accepted median 0.94 m, p95 12.02 m, 56.3% of all valid targets within
2 m. Every rejection is `fewer_than_six_landmarks` on SNGS-021, the ceiling stated
above; the phantom-landmark failure is gone (no `inconsistent_landmarks`). Per
sequence against PnLCalib:

| Sequence | PnLCalib accepted / median / p95 | v18-vis accepted / median / p95 |
| --- | --- | --- |
| SNGS-021 (halfway-line view) | 15 / 15, 0.43 m, 2.13 m | 0 / 15 |
| SNGS-045 | 15 / 15, 0.32 m, 1.45 m | 15 / 15, 0.36 m, 1.75 m |
| SNGS-089 | 15 / 15, 0.92 m, 585 m | 15 / 15, 1.62 m, 16.9 m |

Where enough vertices are in frame the model is within a few centimetres of
PnLCalib. The remaining gap is structural, not a training defect: the network
predicts 32 vertices and nothing else, and 255 training images is a small set.
Not promoted (2026-09-08): a calibrator that solves 30 of 45 frames is not a
replacement for one that solves 45, whatever its licence. Next build, in order:

1. Line and conic supervision without new labels. For each labelled training
   image fit the ground-truth homography from its own vertices, project the full
   pitch template (line segments, centre circle, penalty arcs) into the image and
   render those as extra heatmap channels; calibrate from point-on-line and
   point-on-conic constraints so a halfway-line view is solvable and the
   six-vertex guard can be retired for a residual-based one.
2. Synthetic expansion by the same route: perturb the fitted homographies, warp
   the frames, re-render targets; add the other CC BY pitch-keypoint sets on
   Roboflow Universe. Never pseudo-label with PnLCalib outputs (GPL weights).
3. A game-disjoint calibration evaluation an order of magnitude larger than 45
   frames before any promotion call; PnLCalib's own p95 on SNGS-089 (585 m) shows
   the incumbent is not clean either.
4. Promotion bar, to be ratified before the next run: accepted frames and
   median error no worse than PnLCalib on the enlarged evaluation, with p95
   bounded.

## Line-aware calibrator `pitch-hrnet-w32-v18-lines`, run and rejection audit 2026-09-08

Same recipe as v18-vis plus 20 primitive channels (17 line segments, 3 conics)
rendered from each training image's own fitted homography, fitted with
`primitive_calibration.calibrate_primitives` (commit 9edcfd6). 60 epochs, best
epoch 34: valid keypoint median 1.59 px, PCK@10px 1.0, visibility accuracy 0.996,
primitive precision 0.80 / recall 0.86, validation calibration 33 of 34 accepted at
0.27 m. Checkpoint `artifacts/models/pitch-hrnet-w32-v18-lines/best.pt`, SHA256
`5f3e7b12715b948c7…`, registered as a shipping-class candidate in
`soccerviz.vision.checkpoint_licence`; not the workflow default, not promoted.

On the frozen 45 frames (`heatmap-hrnet-w32-v18-lines-{predictions,results}.json`,
predictions SHA256 `cc64c750660124ae7…`): **37 of 45 accepted**, 735 of 876
footpoints projected, accepted median 0.46 m, p95 9.86 m, **76.0% of all valid
targets within 2 m** against PnLCalib's 83.9%. The accepted-only medians below are
not comparable across columns when the accepted counts differ; the within-2 m
column counts every target and is the like-for-like number.

| Sequence | PnLCalib accepted, within 2 m of all targets, median | v18-lines accepted, within 2 m, median |
| --- | --- | --- |
| SNGS-021 (halfway-line view) | 15 / 15, 0.93, 0.43 m | 8 / 15, 0.53, 0.60 m |
| SNGS-045 | 15 / 15, 0.98, 0.32 m | 15 / 15, 1.00, 0.33 m |
| SNGS-089 | 15 / 15, 0.60, 0.92 m | 14 / 15, 0.69, 0.93 m |

### Where the eight rejections come from (replayed locally, same verdicts)

The network's maps for all 45 frames were dumped on the Mac
(`scripts/benchmarks/dump_calibration_maps.py`, fp16, MPS) and the fit replayed
in-process (`replay_calibration_fit.py`, `evaluate_fit_variants.py`); the replayed
baseline reproduces the Spark run (37 / 45, 76.5% within 2 m). Three defects, none
of them the network:

1. **The `h33 = 1` gauge in `_refine` is degenerate whenever the image origin lies
   near the pitch horizon.** Every `underdetermined` verdict is a refined Jacobian
   whose eighth normalised singular value is ~1e-8 against ~3e-3 on the accepted
   neighbour frame with identical detections (SNGS-021 f21 vs f31; SNGS-089 f31,
   f41, f51 have the horizon at image y = +68, ~0, -73 px). Re-parameterising the
   refine with a unit-Frobenius 9-vector removes all seven `underdetermined`
   rejections on its own (41 / 45).
2. **Noise ridges above `PRIMITIVE_MIN_PIXELS` (8) enter the fit as lines.** On
   SNGS-021 f1, f41, f51, f81, f91, f101 an 8 to 37 point `side_bottom` or
   `penalty_left_bottom` blob sits 28 m from where the 80-point real lines put it
   (the near touchline is not in frame). The RANSAC seed already identifies them as
   gross outliers; pruning primitives further than 5 m from the seed before the
   refine, with the DLT seed as fallback, clears them.
3. **The vertex-only RANSAC seed can lose the fifth vertex.** With v13, v14, v15
   collinear on the halfway line, `findHomography` fits the four circle points
   exactly and leaves the sideline vertex 2.5 m out; the DLT seed over all
   primitives solves those frames at 0.39 to 0.49 m.

Fitter variants over the same maps, official evaluator
(`artifacts/vision-benchmark/calibration/heatmap-hrnet-w32-v18-lines-maps/variants/`):

| Fitter | Accepted | Within 2 m, all targets | p95 |
| --- | --- | --- | --- |
| baseline (as shipped) | 37 / 45 | 0.765 | 10.1 m |
| unit-norm gauge | 41 / 45 | 0.844 | 8.3 m |
| prune 5 m + DLT fallback | 44 / 45 | 0.901 | 7.9 m |
| unit-norm gauge + prune 5 m + DLT fallback | 45 / 45 | 0.926 | 7.4 m |

With the combined fitter every SNGS-021 frame solves at 0.39 to 0.79 m; no
accepted frame outside the three below exceeds 1.41 m.

**Ported into the library 2026-09-08.** `primitive_calibration`
now refines in a tangent-space chart around the seed (`_gauge`: unit-norm seed plus
an orthonormal 8-basis of its complement, so only the projective scale is removed),
prunes primitives the seed places more than `PRUNE_M` = 5 m out (`_prune`), and
retries from the DLT seed when the RANSAC-seeded refine is rank-deficient.
`RANK_TOLERANCE` moved from 1e-7 to 3e-6: under the new chart the 45 frames sit at
6e-5 to 2.6e-4 and a fully degenerate input (three collinear vertices on their own
line) at 1.2e-7. Tests: `test_image_origin_on_the_horizon_is_still_solvable`,
`test_noise_ridge_far_from_the_seed_is_pruned_not_refined`,
`test_stranded_fifth_vertex_falls_back_to_the_dlt_seed`. The official benchmark was
re-run with the library fitter on the Mac (MPS, same checkpoint, same protocol):
`heatmap-hrnet-w32-v18-lines-tangent-gauge-{predictions,results}.json`, predictions
SHA256 `968c6efb338ace2b…`. Under the ground-truth exclusion below:

| Sequence | PnLCalib accepted, within 2 m, median / p95 | v18-lines + library fitter accepted, within 2 m, median / p95 |
| --- | --- | --- |
| SNGS-021 | 15 / 15, 0.929, 0.43 / 2.13 m | 15 / 15, 1.000, 0.54 / 1.07 m |
| SNGS-045 | 15 / 15, 0.985, 0.32 / 1.45 m | 15 / 15, 1.000, 0.32 / 0.71 m |
| SNGS-089 (12 scored) | 12 / 12, 0.752, 0.58 / 699 m | 12 / 12, 0.991, 0.68 / 1.58 m |
| all 42 | 42 / 42, 0.902, 0.43 / 3.67 m | **42 / 42, 0.998, 0.45 / 1.39 m** |

Still not promoted: 42 frames of three games is the development sample, and the
promotion bar (item 4 above) waits on the game-disjoint evaluation.

### The SoccerNet ground truth is wrong on SNGS-089 frames 1, 11 and 21

PnLCalib, v18-vis and v18-lines all place the footpoints 12 to 17 m (f1, f11) and
2.6 m (f21) from the annotation, with the same mean offset (dx +8 / dy +9 m on f1
for both PnLCalib and v18-lines, std 3 / 5 m) decaying to zero by f31. The overlay
`…-maps/overlays/SNGS-089-000001-gt.png` shows the v18-lines pitch projection lying
on the painted lines while the annotation puts players standing right of the
halfway line at x = -11 to 0 m and a player well below the centre circle at
(-4.3, 4.4). SoccerNet's `bbox_pitch` is its own per-frame camera output and lags
in this shot. Excluding those three frames: PnLCalib 42 / 42, 0.902 within 2 m
(its two 500 m frames f111, f121 remain); v18-lines baseline 34 / 42, 0.816;
combined fitter 42 / 42, **0.996**. The exclusion is a benchmark-protocol decision,
not a model result. Approved 2026-09-08 and applied at the ground-truth
index, not the protocol, so every predictions file keeps its provenance:
`artifacts/vision-benchmark/ground-truth.json` carries `excluded_frames` with the
reason, `evaluate_calibrations` still scores those frames and reports them under
`excluded_frames` in every results file, and `overall` / `by_sequence` run over the
42 remaining. The four earlier results files were re-evaluated in place; their
45-frame versions are kept as `<name>-results.pre-exclusion.json`. The
"Executed comparison" section above is the 45-frame record and is left as written.

## Game-disjoint calibration evaluation, 2026-09-08: the bar is not met

The GSR valid split holds only games 2, 3 and 5,
the three already in the development sample (`artifacts/calibration-benchmark/
gsr-{train,valid,test}-inventory.json`, label headers read through the bounded range
reader, 4.7 MB a split). The other six GSR games are in train (4, 6, 9) and test
(7, 8, 11); they are used here for evaluation only, never training, under the
2026-09-07 SoccerNet eval-only ruling. Corpus `artifacts/calibration-benchmark/`:
two sequences per game, the first two by id with distinct action classes, one frame
a second over each 30 s clip (stride 25), **360 frames, 12 sequences, 6 games**,
frozen before any model output (`calibration/protocol.json`, SHA256 `ed2d1a32…`;
transfer 190 MB). Overlap of these games with the Roboflow v18 training images and
with PnLCalib's SoccerNet-Calibration pretraining is unknown; PnLCalib was trained
on SoccerNet calibration data, so this corpus is disjoint for our model only.

Same checkpoint, same library fitter, three runs: PnLCalib on Spark
(`soccerviz-calibration:20260907`), v18-lines on Spark (cuda) and on the Mac (mps).
The cuda and mps runs agree (318 vs 321 accepted, 0.742 vs 0.749 within 2 m), which
closes the device question from the development sample.

| Corpus, 360 frames | Accepted | Within 2 m, all targets | Median | p95 | Accepted frames over 2 m |
| --- | --- | --- | --- | --- | --- |
| PnLCalib | 348 / 360 | **0.892** | 0.52 m | 2.9 m | 28 (worst 81 m, 17 m, 11 m) |
| v18-lines, library fitter (cuda) | 318 / 360 | 0.742 | 0.72 m | 5.8 m | 57 (worst 26 m, 21 m, 15 m) |

Per sequence the model matches or beats PnLCalib on SNGS-060, 132, 187, 188 and
loses everywhere else, worst on SNGS-117 (12 / 30 accepted, 0.20 within 2 m against
PnLCalib's 24 / 30, 0.72), SNGS-152 (0.60 vs 0.92) and SNGS-116 (0.64 vs 0.93).
Rejections: 17 `underdetermined`, 17 `inconsistent_primitives`, 7 `fit_failed`, 1
`too_few_primitives`; PnLCalib refuses 12 as `upstream_no_camera`.

Ground truth checked the same way as on SNGS-089: over the 317 frames both models
accept, 8 have both more than 2 m from the annotation while agreeing with each other
within 1 m (SNGS-098 frames 1 to 176 and SNGS-151 f401, all about 3 m), so the
annotation is suspect there but mild. 26 frames have v18-lines over 2 m with PnLCalib
under 1 m; 5 the other way. The gap is the model's, not the labels'.

**Verdict: not promoted.** The promotion bar (accepted frames and median no worse
than PnLCalib, p95 bounded) fails on every line on unseen games. The development
sample's 42 / 42 was the three games the fitter was tuned on and does not transfer.
PnLCalib, GPL, stays the workflow default. Next build, in order: a failure census of
the 42 rejections and 57 bad frames on this corpus by the replay tooling (maps dumped
per frame, seed conditioning, primitive census), then the data route the earlier
sections name (synthetic expansion from fitted homographies, the other CC BY pitch
keypoint sets), retrained and re-scored on this same frozen corpus. The corpus is the
promotion gate from now on; nothing is judged on the 45-frame sample again.

## Failure census on the game-disjoint corpus, 2026-09-09: the fitter, not the data

This follows the recorded next step. The v18-lines maps for all 360
corpus frames were dumped on the Mac (`scripts/benchmarks/dump_calibration_maps.py`,
fp16, MPS, 3 min) and every frame scored by
`scripts/benchmarks/census_calibration_failures.py` against a **reference
calibration**: PnLCalib's homography on frames where PnLCalib lands under 1 m of the
ground truth (274 of 360). Under the reference each frame's decoded evidence is
graded (vertices: hit / offset / false / miss / phantom; lines and conics: correct /
offset / mislabelled / noise / miss / phantom, with the template line a wrong ridge
actually lies on), an **oracle fit** is run over only the correctly labelled
evidence, and the failure is attributed one dimension at a time with the rule
recorded per row (`…-maps/census-rows.jsonl`, `census-report.json`). Frames without a
reference are `no_reference`, never inferred. The maps are MPS output, so the
classification is the MPS results file (263 good, 39 rejected, 58 bad; the cuda run
is 42 / 57); the replayed library fit reproduces the MPS verdict on 355 of 360.

| Failures (97) | Frames | What the census says |
| --- | --- | --- |
| no reference (PnLCalib also over 1 m or refused) | 49 | Unattributed. Includes the 12 SNGS-098 frames where both models agree ~3 m from the label and 9 SNGS-117 rejections. |
| geometry: every visible line delivered, oracle fit still fails or drifts | 39 | 20 rejected, 19 bad. Not a detection problem. |
| fitter: only prunable contaminants, oracle solves | 5 | |
| network: a phantom line inside the 5 m prune | 4 | SNGS-132 f676, f726 and SNGS-151 f676, f701, a phantom `penalty_*_bottom`. |
| network: too little correct evidence | 0 | |

The 39 geometry frames are two views:

1. **Halfway views with the centre circle cut by the frame edge.** 52 referenced
   frames show only `side_top`, `halfway` and `centre_circle`; 18 fail (15 rejected,
   3 bad). In all 34 good ones both circle extremes v30 and v31 are in frame and fired.
   In 16 of the 18 failures one of them is **outside the frame** (reference
   `abstain`, network confidence ≤ 0.007: the network is right to stay silent); the
   other two are a v30 miss (SNGS-061 f226) and a 1.8 m v30 offset (SNGS-116 f626).
   Every one of the 18 delivers the halfway line (80 ridge points), the arc (63 to 80)
   and the far touchline. The oracle fit over that correct evidence is
   `underdetermined` or `inconsistent_primitives` in 15, accepted 5.1 and 9.5 m off
   the reference in 2, and solves 1 (SNGS-117 f276, 0.5 m). With every vertex on the
   halfway line and only a partial arc, the free homography has no well-conditioned
   scale along the pitch; PnLCalib solves the same frames.
2. **End views without the near touchline, players in the extrapolation zone.** 128
   referenced frames show a goal end and no `side_bottom`; 22 fail (6 rejected, 16
   bad) against 106 good, and 11 of the 16 bad deliver every visible line correctly.
   The share of ground-truth footpoints lying *below the lowest usable evidence pixel*
   is 0.33 (median) on the bad frames, 0.21 on the rejected, 0.07 on the good; with
   the near touchline in frame it is 0.00 for every class (94 frames, 8 fail). The fit
   agrees with the reference on the lines and drifts where there are none: SNGS-152
   f51 has all seven visible lines `correct`, PnLCalib median 0.45 m, v18-lines
   3.27 m, and the overlay `…-maps/overlays/SNGS-152-000051-ref-vs-fit.png` shows
   the fit placing the near touchline inside the frame where the grass is unmarked.

What the network does wrong, for the record: it hallucinates the near touchline
(`side_bottom` phantom on 89 of 274 referenced frames, 28 of them failures), a
`penalty_left_bottom` on 54 and `penalty_right_bottom` on 29, all end-view mirror
confusions; the 5 m prune removes them on every frame but the four above. Nine
mislabelled ridges in 274 frames (six `side_top` on `goal_right`). Missed vertices on
failures concentrate on the penalty-box corners (v25 11, v21 9, v20 and v23 5). None
of that is what fails the 39 frames.

**What is observed:** on 39 of the 48 attributable failures our detections are
complete and correct under the reference, and `primitive_calibration`'s free 8-DOF
homography still fails or drifts on them, so more of the same training data cannot
be what fixes those 39. **What is not observed:** why PnLCalib solves the same
frames. It differs from us in both its detections (57 keypoints including line
extremities, plus lines) and its model (a calibrated pinhole camera); this census
does not separate the two, and the earlier text in this section that credited the
camera model was an unverified inference. No next step is established by this
census; the recommendation built on that inference was withdrawn (2026-09-09).

## Model comparison on the game-disjoint corpus, 2026-09-10: the camera model is the lever

This is the measurement the census left open. Same dumped maps,
same decode, same seed, prune, inlier and leave-one-out acceptance; only the model
fitted changes. `calibrate_primitives` gained a pluggable `refine` step (the free
homography stays the default) and `pinhole_calibration` supplies a pinhole camera:
square pixels, zero skew, principal point at the image centre, so `pinhole` fits focal,
rotation and position (7 unknowns) and the `pinhole-held` variants fit focal and
rotation (4) with the camera held at a position estimated without ground truth. The
decomposition behind it reproduces PnLCalib's own recorded focal and position from
its homographies on SNGS-060 to 0.01 m. Held positions, per sequence: the median
implicit camera of the frames the library fitter accepted (`pinhole-held`; those
positions scatter with a median absolute deviation of up to 25 m on SNGS-116 and 152),
the median of the free pinhole fitter's accepted frames (`pinhole-held-self`; MAD under
2 m on 10 of 12 sequences), and that median pooled over both sequences of a game
(`pinhole-held-game`; one main-camera tripod per broadcast). `pinhole-held-good`
selects frames by ground truth and is a labelled diagnostic only. Two evidence tracks:
`raw`, the decoded maps as the product sees them, and `oracle`, only the evidence the
census graded correct under the PnLCalib reference (311 frames have one). Harness
`scripts/benchmarks/evaluate_fit_variants.py`; every fit in
`…-maps/variants/frame-fits.jsonl`, official results per variant beside it, the
census-conditioned table in `variants/census-by-variant.json`. Two harness launches: the
second added the pinhole-derived positions after the first's library-derived positions
were read and found to scatter; its rows were reused.

Official evaluator, 360 frames, raw evidence; "solved" is accepted with median footpoint
error within 2 m, the census's own rule:

| Fitter | Accepted | Within 2 m, all targets | Median | p95 | Geometry frames solved, of 39 | Good frames kept, of 263 | Halfway views solved, of 50 | End views solved, of 209 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| free homography (library) | 319 / 360 | 0.748 | 0.72 m | 5.39 m | 0 | 262 | 32 | 179 |
| pinhole, free position | 330 / 360 | 0.801 | 0.64 m | 4.29 m | 11 | 257 | 39 | 185 |
| pinhole held, library-derived position | 320 / 360 | 0.800 | 0.72 m | 2.85 m | 18 | 241 | 36 | 187 |
| pinhole held, own position per sequence | 324 / 360 | 0.826 | 0.59 m | 3.22 m | 23 | 250 | 38 | 194 |
| pinhole held, own position per game | 326 / 360 | **0.847** | 0.54 m | **2.72 m** | **27** | 251 | 42 | 195 |
| PnLCalib (GPL, reference) | 348 / 360 | 0.892 | 0.52 m | 2.90 m | | | | |

On the oracle track (correct evidence only) the ordering is the same and the gap
closes further: the free homography solves 1 of the 39 geometry frames, the free
pinhole 14, held per game 33; halfway views 33 to 47 of 50, end views 185 to 197 of
209. So the model, not the evidence, is what those frames were missing.

What is observed, per view:

1. **Halfway views are the free camera's win.** Where the library was `underdetermined`
   the 7-parameter pinhole solves SNGS-061 f251, 116 f701 and f726, 117 f201, f226 and
   f526, 132 f226 at 0.18 to 0.70 m. Holding the position also solves them, at 0.4 to
   1.8 m depending on how good the position is.
2. **End views are the held camera's win.** The free pinhole barely moves them
   (SNGS-133 f551 11.5 to 6.8 m, SNGS-152 f651 to f726 stay at 2.1 to 2.9 m); holding
   the position takes SNGS-133 f226 to f551 to 0.3 to 0.7 m, SNGS-152 to 0.7 to 1.8 m,
   SNGS-097 f326 from 3.79 to 0.49 m. With the same lines a free camera trades distance
   for focal length and the extrapolation below the evidence drifts; a held position
   removes that trade. PnLCalib's own record shows the same ambiguity: on SNGS-060 its
   position moves from y = 94.0 to 87.7 m and its focal from 3460 to 2536 px between
   frames 1 and 126 (`pnl-predictions.json`, `camera_parameters`).
3. **The held camera needs a position it does not yet get reliably.** The per-game
   median of the free pinhole fits lands within 0.2 to 2.9 m of PnLCalib's on games 4,
   6, 8 and 11 and 6 to 12 m away on game 7 (SNGS-116, 117) and 8 m on SNGS-152. SNGS-117
   stays the worst sequence (0.32 within 2 m against PnLCalib's 0.72) and three of its
   halfway frames (f151, f176, f501) are *accepted* by the held fit at 12 to 32 m: the
   leave-one-out guard passes a wrong camera. Per sequence the held-per-game fit matches
   or beats PnLCalib on SNGS-060, 098, 132, 133, 187 and 188 and loses on 061, 097, 116,
   117, 151 and 152.
4. **Cost.** The held-per-game fit loses 11 frames the library solved (8 refused as
   `inconsistent_primitives` in SNGS-097, 133, 151 and 152, three drifting to 2.2 to
   2.3 m) and gains 53. The free pinhole loses 5 and gains 25. Fit time per frame falls
   from 458 ms (library) to 208 ms (pinhole) and about 100 ms (held).
5. **Still unsolved by every model, on both tracks:** SNGS-116 f251 (thin, mislabelled
   evidence), SNGS-117 f151, f176 and f501 (the wrong accepted cameras above), SNGS-132
   f576 and SNGS-151 f451 (held fits refused as inconsistent with a position 0.3 and
   2.5 m from PnLCalib's; not explained here). Six more of the 39 solve only on the
   oracle track (SNGS-061 f226 and f376, SNGS-117 f76, f126, f326 and f351): there the
   network's contaminants and the model fail together.

**Verdict:** the promotion bar is still not met (0.847 within 2 m and 326 accepted
against PnLCalib's 0.892 and 348), so nothing is promoted and the workflow default is
unchanged. What is established: on identical evidence, replacing the free homography
by a pinhole camera with a held position solves 27 of the 39 census failures and lifts
the corpus from 0.748 to 0.847 within 2 m with a better tail than PnLCalib's; the free
homography is the cause of those failures. What is not established: whether the
remaining gap to PnLCalib is the model (its lens distortion term, untested here: the
fitter's prune, inlier and leave-one-out steps all consume a plain matrix, so a
distortion term needs a mapping abstraction first) or its detections (57 keypoints
including line extremities, 22,816 training images). A product version of the held
camera needs two things this measurement did not build: a position estimate that
survives clips like SNGS-116 and 152, where the free fits scatter (a joint tripod fit
over the clip, as BroadTrack does, or falling back to the free camera until the
estimate tightens), and a camera-plausibility check in the acceptance guard so a
12 m fit cannot pass. Both are open decisions.

## Frozen sample and coordinates

The sample is **45 images across three different games**, drawn from the frozen
450-image development corpus. Each sequence contributes source frames
1, 11, 21, …, 141 at a source rate of 25 Hz. The selection predates calibration
outputs. No candidate-dependent sample filtering is allowed.

The full manifest remains SHA256
`75200edcfe9851a5c68a68bfe36ef320755d90d33bd4ece98590c8d31df08795`.
The separate calibration protocol is
`artifacts/vision-benchmark/calibration/protocol.json`, SHA256
`baaf2c5e39d7692f9e5848b3bc7fc4ef150cbded1dceb40e60aa2d9822fff3d6`.

Both adapters return a 3×3 homography from **original image pixels to centered
105×68 m pitch coordinates**: x from −52.5 to 52.5, y from −34 to 34. Existing
YOLO coordinates are translated by [−52.5, −34]. PnLCalib already uses centered
coordinates; its official camera projection is restricted to z=0 and inverted.
No reflection, sign correction, or geometric fitting uses ground-truth labels.

## Actual implementations

**PnLCalib** imports the pinned upstream `inference.py`, including its real HRNet
models, heatmap extraction/completion, `FramebyFrameCalib`, `heuristic_voting`
with line refinement enabled, and camera projection construction. It resizes
RGB tensors to 960×540 and uses upstream demo thresholds 0.3434 for keypoints and
0.7867 for lines. Normalized network coordinates are denormalized using the
original image dimensions. Each frame gets a fresh camera object. A non-null
upstream result is accepted if its pitch-plane homography is finite and
invertible; no additional error cutoff is selected from evaluation results.

**YOLO control** retains the current 640-pixel inference resolution, detection
confidence 0.25, landmark confidence ≥0.5, and unmodified `geometry.calibrate`
acceptance rules. Its six-landmark/inlier requirements, RANSAC threshold, inlier
fraction, and leave-one-out error gates remain unchanged.

Those acceptance policies differ; compare their error **and coverage together**.
All sampled frames retain an explicit accepted/rejected/error record. Runtime
includes neural inference and official geometric optimization, with CUDA
synchronization. The first frame explicitly includes cold-start overhead.

## Provenance

Upstream: [PnLCalib](https://github.com/mguti97/PnLCalib), commit
`8c87391d6f4ea40c5e4d65e61529916c7a49ce62`.
The adapter verifies eight exact source/configuration file hashes before loading
models and records them with all package versions. It imports upstream source
without patching its geometry or networks.

| Checkpoint | SHA256 |
|---|---|
| `pnl-keypoints.pt` | `7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113` |
| `pnl-lines.pt` | `d72f4ed71734a2e3df9fa084f666e9b8adaef21bf69bac8952d6d3f970ff7455` |
| Existing `football-pitch-detection.pt` | `28f68f7c4056d6d9b137efd2e7ab5f3c494039380c63831649126ced25628b36` |

YOLO's checkpoint and `geometry.py` hashes are recorded directly in each run.
Inference inputs contain only image paths and the frozen protocol. Source
annotations enter only the separate evaluation command.

## Run on Spark

`envs/calibration-model/Dockerfile` starts from NGC PyTorch 26.01. Exact
constraints preserve the base Torch/Torchvision packages; installation fails if
their replacement would be required. Dependencies are isolated from the local
core environment, and the image verifies their imports after installation.

```sh
docker build -f envs/calibration-model/Dockerfile -t soccerviz-calibration .
```

Inside the worker with the repository mounted at `/work` and weights at `/models`:

```sh
python scripts/benchmarks/run_calibration_benchmark.py run \
  --manifest artifacts/vision-benchmark/manifest.json \
  --protocol artifacts/vision-benchmark/calibration/protocol.json \
  --backend pnlcalib --upstream artifacts/upstreams/pnlcalib \
  --weights-keypoints /models/pnl-keypoints.pt \
  --weights-lines /models/pnl-lines.pt --device cuda:0 \
  --path-prefix "$PWD"=/work \
  --out artifacts/vision-benchmark/calibration/pnl-predictions.json

python scripts/benchmarks/run_calibration_benchmark.py run \
  --manifest artifacts/vision-benchmark/manifest.json \
  --protocol artifacts/vision-benchmark/calibration/protocol.json \
  --backend yolo --checkpoint /models/football-pitch-detection.pt --device cuda:0 \
  --path-prefix "$PWD"=/work \
  --out artifacts/vision-benchmark/calibration/yolo-predictions.json
```

Each command refuses to overwrite an existing result. Run GPU jobs sequentially.
Image hashes and dimensions are checked before inference. Model exceptions
become explicit frame failures; source-integrity failures abort the run.

After returning predictions to the original local workspace, evaluate each model:

```sh
PYTHONPATH=src .venv/bin/python scripts/benchmarks/run_calibration_benchmark.py evaluate \
  --manifest artifacts/vision-benchmark/manifest.json \
  --protocol artifacts/vision-benchmark/calibration/protocol.json \
  --predictions artifacts/vision-benchmark/calibration/pnl-predictions.json \
  --out artifacts/vision-benchmark/calibration/pnl-results.json
```

Use the same command with the YOLO prediction/output names. Evaluation checks
the manifest, protocol, original image and annotation SHA256 checksums. It
reports frame coverage, projection coverage, failure reasons, median/p95 meter
error for available projections, and the fraction of **all valid target
footpoints** that land within 2 m. Large errors and off-pitch projections remain
scored. Missing/rejected frames retain all their valid targets in the denominator.

## Verification status

Twelve local tests pass, including known camera-projection inversion, centered
pitch corners/orientation, source corruption, missing/rejected frames, and large
errors remaining in the metrics. The real-source empty-prediction diagnostic is
saved at `artifacts/vision-benchmark/calibration/evaluator-empty-sanity.json` and
explicitly marked as an evaluator sanity check, not model performance.

Both checkpoints and geometric pipelines have now executed on NVIDIA GB10 using
NGC Torch `2.10.0a0+a36e1d39eb.nv26.1.42222806` and Torchvision
`0.25.0a0+6b56de1c.nv26.1.42222806`; the vendor packages were preserved. Each run
contains 45 unique sampled frame records, and every accepted homography is finite.
The evaluator rechecked original image/annotation hashes and reproduced both
saved overall and per-sequence result dictionaries exactly.

Prediction SHA256 values:

- PnL: `29e33bafe249741c15692bbb9dc7daa4abdf54884cd5b207300d0b9393e9744f`
- YOLO: `a84aa5f69ddfddde44bd82783e44a60b8aeb16a3c6e4cef5a4f6252e8c7a8165`
