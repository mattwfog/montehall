# SoccerViz vision model recommendations

Research date: September 7, 2026 (America/New_York). Sources checked live, with
project code, checkpoint metadata, and existing annotation files inspected.
This is an investigation and proposed experiment program. No candidate model
was installed, trained, benchmarked, or promoted during this investigation.

## Recommended direction

Make RF-DETR Small/Medium the first alternative player-detector experiment,
with the current soccer-trained YOLOv8x as the control and a soccer-fine-tuned
YOLO26 Small/Medium as the next comparison. In parallel, evaluate dedicated
calibration, temporal ball detection, and jersey recognition. Select winners
per component using the same annotated clips and measured Spark runtime.

The most material additions to the previous survey are soccer-trained RF-DETR,
WASB-SBDT, Uncertainty-JNR, SoccerMaster/SoccerFactory, and UnravelSports EFPI.
Their release availability establishes candidates, not improvements on SoccerViz.

## What the project actually uses

- Static inspection of the three checkpoint pickle streams, without loading
  executable pickle objects, identifies `yolov8x.pt` for players and ball and
  `yolov8x-pose.pt` for pitch landmarks. The installed Ultralytics version does
  not upgrade those model weights to a newer architecture.
- `src/soccerviz/vision/pipeline.py` loads three models directly through Ultralytics.
  Person inference requests 1280 pixels; pitch inference requests 640; ball
  inference requests 640 on four overlapping tiles. It picks the highest-score
  ball candidate independently at each sampled frame. A detector backend
  interface and architecture-neutral class mapping are not implemented.
- Team grouping uses median shirt-color features. Tracking uses our appearance
  associator; jersey recognition uses Tesseract. RF-DETR, dedicated jersey
  networks, and temporal heatmap ball models are not wired into this path.
- The independent SNGS-021 run covers 50 frames over 9.8 seconds. Its image HOTA
  is 0.50457 and IDF1 0.59246. Accepted pitch coordinates are absent for 250
  predictions. This remains a small development benchmark.
- The 50 sampled SNGS-021 frames have 50 ball annotations. Median annotated
  width/height are 11/13 pixels at 1920x1080. Width quartiles are 11 and 19
  pixels. Resizing a 1920-wide frame to 576 reduces an 11-pixel width to 3.3
  pixels; this arithmetic illustrates why model size alone is insufficient.
- The local GSR annotation file contains ball labels, although the official
  GS-HOTA evaluation excludes balls. Ball accuracy needs its own metric and
  an explicit annotation census; do not assume all dataset subsets are alike.
- The existing GPU image is based on NVIDIA PyTorch 26.01 and has verified YOLO
  execution on GB10. That does not verify new dependencies or export runtimes.

Local evidence: `artifacts/video/demo-v2/report.json`,
`artifacts/benchmarks/sngs-021-original-final-evaluation.json`,
`artifacts/public-data/soccernet/gsr/valid/SNGS-021/Labels-GameState.json`,
`envs/cv/Dockerfile`, and `src/soccerviz/vision/pipeline.py`.

## Detector candidates

| Candidate | Available material | Recommended use |
|---|---|---|
| RF-DETR Small/Medium | General pretrained detectors, custom training, ONNX/TensorRT export; separate segmentation models and preview keypoints. | First new detector family to test. Benchmark both native resolution and an appropriate larger/tiled setting. |
| Soccer-trained RF-DETR Medium | Roboflow Universe model `football-players-detection-3zvbc/20`; four classes: player, goalkeeper, referee, ball. | Relevant immediate inference candidate if accessible; local checkpoint download remains unverified. |
| YOLO26 Small/Medium | Current Ultralytics pretrained family and training/export API. | Lower integration effort for an equally soccer-fine-tuned comparator. |
| RT-DETRv2 / RT-DETRv4 | Released weights, custom-data training and deployment tools. v4 is now available; v2 is a mature reference. | Second wave if the first two candidates leave a meaningful accuracy/runtime gap. |
| D-FINE / DEIM / DEIMv2 | Released models and custom COCO training. DEIM improves matching/training; v2 uses DINOv3 features. | Additional research comparison after establishing the shared benchmark. Avoid spending the first iteration maintaining every detector family. |

[RF-DETR](https://github.com/roboflow/rf-detr) is a DINOv2-based transformer
detector. Its COCO/RF100-VL results motivate testing, but establish neither
soccer-ball superiority nor GB10 speed. Published latency is T4/TensorRT/FP16,
batch one. Nano through Large detection models are Apache-designated; XL/2XL
Plus components use different terms. Prefer Small/Medium for the first test.

Stable [RF-DETR 1.10.1](https://github.com/roboflow/rf-detr/releases/tag/1.10.1)
was released September 7. Pin the version and checkpoint: 1.10 changed training
defaults. [Custom training](https://rfdetr.roboflow.com/latest/learn/train/)
supports COCO JSON and YOLO data formats. Generic pretrained classes do not
distinguish soccer playing roles. Initially collapse person roles for a general
person comparison; evaluate role recognition only for suitably trained outputs.

The [Roboflow soccer model page](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
advertises RF-DETR Medium on 372 source images and documents API-key inference.
This is more task-relevant than comparing generic RF-DETR with soccer-trained
YOLO. The displayed dataset metrics are not independent SoccerViz results.
Hosted access does not prove freely downloadable standalone weights.

[YOLO26 documentation](https://docs.ultralytics.com/models/yolo26) describes
small-target-aware training and end-to-end inference. Its default-head and
end-to-end-head accuracy figures differ; record the head used with the timing.
P2/P6 configurations are available, but their pretrained weights are not, so
they are later training experiments rather than immediate checkpoint swaps.

[RT-DETR](https://github.com/lyuwenyu/RT-DETR),
[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4),
[D-FINE](https://github.com/Peterande/D-FINE), and
[DEIM](https://github.com/Intellindust-AI-Lab/DEIM) offer credible alternatives.
[DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) has model downloads and
custom training/export instructions; its README specifies TensorRT >=10.6 for
the FP16 inference fix. Cross-repository headline AP/FPS tables use differing
protocols; they are not our model-selection result.

## Complementary components

| Component | Candidate | Experiment and limitation |
|---|---|---|
| Ball motion evidence | [WASB-SBDT model zoo](https://github.com/nttcom/WASB-SBDT/blob/main/MODEL_ZOO.md) | Soccer-specific WASB and TrackNetV2-family checkpoints are listed. Test temporal heatmaps against tiled YOLO ball detection. Use consecutive source frames and document look-ahead. |
| Pitch calibration | [PnLCalib](https://github.com/mguti97/PnLCalib), [NBJW](https://github.com/mguti97/No-Bells-Just-Whistles) | Released point/line models and geometric calibration. Compare coordinate error and coverage, especially where the current fit fails. |
| Jersey numbers | [Uncertainty-JNR](https://github.com/lukaszgrad/uncertainty-jnr) | Public SoccerNet ViT-S/ViT-B checkpoints take player crops and return number/uncertainty outputs. Compare correct-number precision and coverage with Tesseract, including unreadable cases. |
| Identity and team | [SoccerNet GameState](https://github.com/SoccerNet/sn-gamestate) | Full PRTReID/tracking/team/jersey/calibration baseline supplies a useful independent comparison. Our installed OC-SORT plugin is a small subset. |
| Team appearance | [Roboflow TeamClassifier](https://github.com/roboflow/sports/blob/main/sports/common/team.py) | SigLIP crop features + UMAP/KMeans versus median color. Labels remain anonymous team clusters until mapped with evidence. |
| Shape and pressing | [UnravelSports](https://github.com/UnravelSports/unravelsports) | EFPI offers 65 formation templates and temporal aggregation; pressing and graph tooling also exist. Start with complete provider tracking. |
| Missing players | [offscreen-impute](https://github.com/nowayfootball/offscreen-impute) | Compare additional estimation policies and downstream pitch-control error. Public benchmark simulates visibility; estimates remain distinct from observations. |

WASB's [setup guide](https://github.com/nttcom/WASB-SBDT/blob/main/GET_STARTED.md)
uses an older Python/CUDA environment and leaves training documentation
unfinished. Treat it first as an inference reproduction task in a modern,
isolated environment. Published results on a narrow soccer dataset do not
establish transfer to our broadcast clips.

Uncertainty-JNR's public ViT-B reports 86.37% test and 83.52% challenge accuracy;
the headline 85.62% challenge model uses an unreleased proprietary-data
checkpoint. Evaluate the available checkpoint, not that headline. Its crop
interface is an attractive first experiment; number recognition still cannot
recover numbers never shown on screen.

EFPI remains geometric formation assignment, not a learned visual tactical
understanding model. Its [documentation](https://unravelsports.readthedocs.io/_/downloads/en/stable/pdf/)
requires ten outfield players and drops incomplete frames. Directly feeding
partial broadcast detections would therefore limit coverage. Use Metrica or
SkillCorner observed tracking first, and assess analyst agreement separately.

## SoccerMaster and annotation tools

[SoccerMaster/SoccerFactory](https://github.com/haolinyang-hlyang/SoccerMaster)
is a substantial new research lead: released model files, a data-generation
pipeline, and 7,000 video sequences with per-frame annotations. Its assembled
pipeline uses specialist models, including tracking, segmentation, calibration,
and a vision-language jersey component. The best setup is heavyweight; the
repository also provides a smaller-language-model path.

Use it first to inspect available soccer-trained heads and additional training
data, then as an offline comparison. Its automatically generated annotations
must stay separate from independently reviewed test labels. The paper's
reported 64.1 GS-HOTA belongs to the annotation pipeline; it is not a measured
SoccerViz result or a score for the foundation model alone.
[Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_SoccerMaster_A_Vision_Foundation_Model_for_Soccer_Understanding_CVPR_2026_paper.pdf),
[released model files](https://huggingface.co/xleprime/SoccerMaster).

[FiftyOne](https://docs.voxel51.com/user_guide/evaluation.html) can compare
predictions on identical images and expose misses, false positives, and class
confusion. [CVAT track tools](https://docs.cvat.ai/docs/annotation/manual-annotation/modes/track-mode-basics/)
fit the existing annotation exchange. These are more immediately useful than
adding another all-purpose annotation model.

[SAM 3](https://github.com/facebookresearch/sam3) and
[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) are useful
optional tools for mask/box suggestions. Model suggestions require review.
Instance masks describe individual object outlines; person keypoints describe
body pose. Neither output alone identifies a team's formation or pressing plan.

## Proposed experiments and decision criteria

1. **Freeze the data and baseline.** Expand beyond SNGS-021 to a bounded set of
   complete sequences with different matches, cameras, kit colors, crowding,
   scale and ball visibility. Reserve train/development/test by match where
   source metadata permits, otherwise document the weaker sequence split.
   Keep SNGS-021 as development because it has already informed choices.
   Preserve source timestamps and include sufficient consecutive frames for
   temporal models. Record any unknown pretraining overlap.
2. **Measure detectors without changing tracking.** Run existing YOLOv8x,
   soccer RF-DETR Medium if accessible, and a pinned general RF-DETR as an
   explicitly separate initialization check. Fine-tune RF-DETR and YOLO26 on
   identical soccer labels, with comparable tuning budgets, before drawing
   an architecture conclusion. Store all boxes/confidences, original image
   dimensions, resize/tile transforms and exact class mappings.
3. **Give the ball its own experiment.** Compare current tiles, configurable
   slicing, and WASB. Score annotated-visible ball recall, false positives,
   center error in source pixels, box AP where supported, and gap duration.
   Report small-object results by source pixel size. A heatmap center method
   need not manufacture boxes just to fit a box-only metric.
4. **Compare calibration and jersey specialists independently.** Retain
   identical detections/crops where possible; measure meter error plus coverage,
   and jersey accuracy plus abstention. Then recompute tracking and game state.
5. **Evaluate downstream usefulness.** Report HOTA/IDF1, identity switches,
   fragmentation, valid pitch-position coverage, ball coverage, team/jersey
   precision, and usable tactical-sequence duration. Where full GS-HOTA cannot
   be computed, preserve its unavailability; any conditional score must show
   the excluded fraction and selection rule.
6. **Benchmark real deployment cost.** On Spark, measure warm forward latency,
   full pipeline throughput, memory, preprocessing, tiled passes and rendering
   separately. Give both accuracy and end-to-end cost. A winning model must
   improve an intended metric without an unacceptable regression elsewhere;
   set numeric thresholds before evaluating the reserved test data.

Implement only a narrow backend adapter first, preserving current Parquet and
provenance contracts. Generic detector classes must not silently enter the
existing soccer-role channel. Keep player, ball, pitch, jersey and shape outputs
separate so each component can improve independently.

For RF-DETR use an isolated Spark-compatible worker.
[Versioned dependencies](https://github.com/roboflow/rf-detr/blob/1.10.1/pyproject.toml)
include Transformers >=5.1 and Supervision >=0.29; training adds Lightning and
a GPU Hungarian matcher. Verify imports, one forward pass, one training step,
and exported-output parity before a larger job. Build TensorRT engines on the
target machine using the [export guidance](https://rfdetr.roboflow.com/latest/learn/export/).
No RF-DETR-on-GB10 execution was verified in this research.

**First work package:** a broader frozen benchmark, RF-DETR versus the existing
soccer detector, PnLCalib, a temporal ball baseline, and public Uncertainty-JNR.
EFPI can proceed on existing complete provider tracking. Defer large-scale
SoccerMaster reproduction and further detector families until those experiments
identify the remaining bottleneck.
