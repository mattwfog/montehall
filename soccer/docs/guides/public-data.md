# Public datasets and measured results

Public benchmarks now supply the first independent annotations. SoccerViz has
SoccerNet GSR and Tracking importers/evaluators, a 30-match StatsBomb corpus with
trained action-value baselines, and two SkillCorner event/tracking imports.
The **Datasets & benchmarks** tab shows results, provenance, and file integrity.
Its experiment records use the same immutable object store as the video harness.

## Actual results

| Experiment | Scope | Result |
| --- | --- | --- |
| Original video pipeline | SoccerNet validation SNGS-021, 50 frames spanning 9.8 seconds | HOTA 0.50457; IDF1 0.59246 |
| Baseline association replay | Same 843 detections, no appearance features | HOTA 0.51184; IDF1 0.60097 |
| TrackLab OC-SORT replay | Same 843 detections, no appearance features | HOTA 0.45054; IDF1 0.52311 |
| Conditional game-state metric | 35/50 frames with complete predicted pitch geometry | GS-HOTA 0.01396; biased toward calibration success |
| Tracking association diagnostic | SoccerNet train SNMOT-060, 750 frames, 13,540 oracle boxes | HOTA 0.83735; IDF1 0.79350; not detector accuracy |
| VAEP scoring probability | 20 train / 5 dev / 5 test StatsBomb matches | Test Brier 0.013189 vs 0.014876 prevalence baseline |
| VAEP conceding probability | Same fixed split | Test Brier 0.002137 vs 0.002149 prevalence baseline |
| SkillCorner ingestion | Two event files and bounded tracking prefixes | 9,045 events, 1,036 off-ball runs, 883 phase intervals |

The video model produced 843 person detections against 801 annotated persons.
Accepted pitch positions were unavailable for 250 detections, so **full-sample
GS-HOTA is unavailable**. The separate 35-frame diagnostic is not a full-clip score.
Unknown team and jersey attributes remain null; target labels never fill model
predictions. Referee/other objects can match without player identity attributes.

The original tracker uses appearance. The fair replay removes it from both
backends because frozen artifacts lack those vectors. This sample supports
retaining our current tracker, but does not establish cross-broadcast performance.
External detector pretraining overlap with SoccerNet is unknown.

StatsBomb conversion produced 67,869 actions: 45,403 train, 10,854 development,
11,612 test. The new probability models use the current completed action, not the
full three-action VAEP architecture. Five test games contain only 175 positive
scoring and 25 positive conceding action labels, with correlated outcomes. Test
match 3788742 previously appeared in the xT smoke test: this split is match-disjoint
but not a pristine project-wide holdout. See the [exposure audit](../../results/experiments/statsbomb-expanded-exposure-audit.json).

SkillCorner data remains provider/model-derived. Detection, extrapolation and
missingness flags are distinct. Full-match events have broader coverage than the
tracking prefixes, and each event carries an explicit coverage flag. Native
tracking axes and attack-normalized event coordinates remain separate.

## Setup and command surface

```sh
uv sync --locked --python 3.12
.venv/bin/python scripts/setup/setup_integrations.py all
.venv/bin/soccerviz workbench --port 7866
```

The setup now includes four worker environments: `analysis`, standard TrackEval
`evaluation`, the separate official SoccerNet fork `soccernet`, and Prefect
`execution`. The SoccerNet fork does not replace standard TrackEval.

All dataset workers take a JSON request and response path:

```sh
.venv/bin/soccerviz datasets run skillcorner --request request.json --response response.json
.venv/bin/soccerviz datasets run statsbomb --request request.json --response response.json --python .venvs/analysis/bin/python
.venv/bin/soccerviz datasets run train-actions --request request.json --response response.json --python .venvs/analysis/bin/python
.venv/bin/soccerviz datasets run soccernet --request request.json --response response.json
.venv/bin/soccerviz datasets run soccernet-evaluate --request request.json --response response.json --python .venvs/soccernet/bin/python
.venv/bin/soccerviz datasets list
.venv/bin/soccerviz datasets inspect RECORD_ID
.venv/bin/soccerviz datasets verify RECORD_ID
```

Add `--name NAME --split SPLIT` to register a successful worker report. Repeat
`--artifact FILE` for produced models and Parquets. Without explicit artifacts,
registration verifies the report only; verification scope is recorded. Existing
reports use `datasets register --name NAME --kind KIND --split SPLIT --report FILE
--artifact FILE`. Large files stay outside SQLite and retain checked hashes.
Registration does not automatically align a provider match with a video run.

Exact requests, pinned sources, and reproduction commands:

- [SoccerNet downloads and evaluation](../experiments/soccernet-benchmarks.md)
- [StatsBomb download, training and portable inference](../experiments/statsbomb-training.md)
- [SkillCorner native ingestion and coverage](../experiments/skillcorner-data.md)

## Actual video benchmark

The downloader extracted source frames 1, 6, …, 246 from SNGS-021 without fetching
the entire multi-gigabyte archive. The helper losslessly encodes those images at
5 Hz and preserves an exact frame map; it adds no interpolation or annotation
overlays. For a new reproduction directory:

```sh
.venv/bin/python -m soccerviz.datasets.soccernet_predictions prepare --sequence artifacts/public-data/soccernet/gsr/valid/SNGS-021 --out artifacts/benchmarks/new-input --frames 50 --stride 5
PYTHONPATH=src .venvs/execution/bin/python scripts/spark/prefect_worker.py video --image sha256:950bb67171f6985147f3f83d3407676587efeb754e187637e8ef148b80f63363 --source artifacts/benchmarks/new-input/source.mp4 --seconds 10 --hz 5
.venv/bin/python -m soccerviz.datasets.soccernet_predictions export --video-run artifacts/execution/JOB_ID/artifacts --frame-map artifacts/benchmarks/new-input/frame-map.json --out artifacts/benchmarks/new-predictions.json
```

Use the prediction file with a `gsr` evaluator request. Requested frame IDs are
original 1-based SoccerNet ordinals. Provenance includes model, frame map,
inference frames, detections, state and calibration hashes.

Actual GPU job: `2738f851cb3f985f8167d95e2e7a753189b4c2c81de5ae1baf25e46443ccce83`.
Completed harness run: `621ddf62541d4676a7d5723f28a9ad5d`; its evaluation is attached
after matching the inference source hash.

## Next measured priorities

Improve calibration coverage and jersey/team outputs; benchmark more official
validation sequences before using the final test split. Expand source samples as
bandwidth and compute permit. Public annotations are the default evaluation path;
CVAT covers specific missing footage and deployment-transfer checks.

Evidence: [tracker comparison](../../results/experiments/soccernet-tracker-comparison.json),
[StatsBomb training](../../results/experiments/statsbomb-expanded-training.json),
[verification record](../../results/experiments/public-data-verification.json).

## Roboflow Universe datasets

`soccerviz.datasets.roboflow_universe` downloads pinned public dataset versions
with the API key read from `ROBOFLOW_API_KEY` only. Each archive is hashed, the
export link is redacted, and the version's preprocessing is recorded because
Roboflow versions can be stretched away from source resolution. Downloaded on
2026-09-07 (request and response in `results/experiments/roboflow-universe-*.json`),
all CC BY 4.0:

| Dataset version | Images | Resolution | Content |
| --- | ---: | --- | --- |
| `football-players-detection-3zvbc` v10 | 312 | 1920x1080 source | player, goalkeeper, referee, ball boxes |
| `football-players-detection-3zvbc` v13 | 372 | stretched 1280x1280 | same, 60 more images |
| `football-players-detection-3zvbc` v20 | 372 | stretched 576x576 | the version the public RF-DETR-M model trained on |
| `football-field-detection-f07vi` v12 | 276 | 1920x1080 source | 32 pitch keypoints per image |
| `football-field-detection-f07vi` v18 | 317 | stretched 960x960 | same, 41 more images |

Use the source-resolution versions (v10, v12) for benchmark work where boxes must
stay in source pixels; the stretched versions only add images.

```sh
ROBOFLOW_API_KEY=... PYTHONPATH=src .venv/bin/python -m soccerviz.datasets.roboflow_universe \
  --request artifacts/public-data/roboflow/download-request.json \
  --response artifacts/public-data/roboflow/download-response-$(date +%Y%m%d).json
```

## Licence census and the copyleft rule

Ruling (2026-09-07): avoid GPL and AGPL licensed data and model code
wherever a permissive alternative exists. Copyleft material may still be used
for internal evaluation, which is not distribution, but nothing trained on it
or built with it is a shipping candidate. Licences marked "observed" were read
from the source's own metadata or terms file on the date given (2026-09-07
unless stated); the rest are unverified until their terms files are read.

| Source | Role today | Licence | Status |
| --- | --- | --- | --- |
| Ultralytics (YOLOv8x default detector, YOLO26 candidate, calibration env) | default video pipeline | AGPL-3.0 | observed on PyPI; **copyleft, network use counts as distribution** |
| RF-DETR (`rfdetr` 1.10.1) | candidate detector | Apache-2.0 | observed on PyPI; permissive |
| SoccerNet SN-GSR-2024 / SN-Tracking-2023 | 450-frame benchmark, 150-image fine-tuning pilots, jersey evaluation | GPL-3.0 plus SoccerNet non-commercial terms | observed on Hugging Face; evaluation only from here |
| Roboflow Universe player and pitch-keypoint sets | training corpus | CC BY 4.0 | observed via the API; attribution only |
| PnLCalib (`mguti97/PnLCalib` at `8c87391`, source and `pnl-keypoints.pt` / `pnl-lines.pt`) | **default calibrator** of the video workflow (`calibration="pnl"`) | GPL-2.0 code; weights pre-trained on SoccerNet-Calibration, fine-tuned on WC14 and TSWC | observed 2026-09-08 in the pinned upstream `LICENSE` and README; **copyleft on both the code and the training data; evaluation only** |
| Roboflow demo checkpoints and clip (`data/demo`) | `legacy-soccer` preset, `yolo` calibration, demo video run | repository code MIT; weights and clip unstated | observed 2026-09-08 in `roboflow/sports` `LICENSE` at `42c80c0`; the weights are Ultralytics YOLOv8 (AGPL runtime) with no licence of their own |
| StatsBomb Open Data (`hudl/open-data`) | 30-match action-value (VAEP) training | StatsBomb Public Data User Agreement, 8 September 2023 (`LICENSE.pdf`) | observed 2026-09-08: research tool only; the User may not distribute the data (1.2.1) or **commercially exploit the data or any analysis derived from it** (1.2.2); publications carry the StatsBomb logo (1.4) |
| SkillCorner open data (`SkillCorner/opendata` at `c1e17a0`) | two tracking imports | MIT | observed 2026-09-08 in the pinned upstream `LICENSE`; permissive |
| Metrica sample data (`metrica-sports/sample-data` at `e706dd5`) | possession lab, formation analysis | none stated | observed 2026-09-08: no `LICENSE` file at the pinned revision (404) and no terms in its README; treat as all-rights-reserved until Metrica states otherwise |
| TrackEval | tracking metrics | MIT | observed in the evaluation venv |
| ViTPose++ base and huge (`usyd-community/vitpose-plus-*`) via `transformers` | pose candidate | Apache-2.0 | observed on the Hugging Face API 2026-09-07; permissive |
| Uncertainty-JNR (`lukaszgrad/uncertainty-jnr` at `f19d9cb`, code and `jersey-vitb.pt`) | jersey reading in the video workflow | CC BY-NC-SA 4.0 code; weights trained on SoccerNet JNR and ReID | observed 2026-09-08 in the pinned upstream `LICENSE` (its README badge says CC BY-SA, the file says NC-SA); **non-commercial on both code and training data; evaluation only** |
| WASB-SBDT (`nttcom/WASB-SBDT` at `923462c`, code and soccer checkpoint) | temporal ball comparator | MIT code; soccer checkpoint training data not read | observed 2026-09-08 in the pinned upstream `LICENSE`; the model-zoo checkpoint's training set is unverified |
| SAM 3D Body (`facebook/sam-3d-body-dinov3`, upstream code and weights) | mesh candidate, run on the frozen benchmark 2026-09-08 | SAM License (Meta, 2025-11-19) | observed in the upstream `LICENSE`; redistribution under the same terms, not copyleft; weights gated |

Consequences for the plan:

1. The two detector fine-tuning pilots (RF-DETR and YOLO26) trained on SoccerNet
   frames are evaluation artefacts, not shipping candidates. Retrain on the
   Roboflow CC BY sets before any promotion.
2. RF-DETR is the only permissive detector in the tree, which strengthens the
   existing recommendation to continue with it. Replacing the Ultralytics
   default is now a licence task as well as an accuracy task.
3. New datasets are checked against this table before download; prefer
   CC BY, MIT, Apache and CC0. Two earlier assumptions here were wrong and
   are corrected in the candidate table below: SoccerTrack's Kaggle release
   is LGPL-3.0, not CC BY, and Voxel51's MIT card on SoccerNet-V3 covers the
   FiftyOne packaging of 1,799 SoccerNet samples, not the SoccerNet images,
   which stay under SoccerNet's own terms.
4. Found 2026-09-08: the video workflow's default calibrator, PnLCalib, is
   GPL-2.0 and its weights are trained on SoccerNet. Both halves of the
   shipping path (detector and camera) are therefore copyleft today. The
   permissive replacement is a keypoint model trained with RF-DETR or
   ViTPose++ (Apache-2.0) on the CC BY Roboflow pitch-keypoint set, feeding
   the repo's own `geometry.py` homography. Until it exists PnLCalib stays
   the default and is labelled evaluation-only, like `rf-soccer.pth`.
5. Found 2026-09-08: the StatsBomb agreement forbids commercial exploitation
   of the data or any analysis derived from it. The VAEP scoring and
   conceding models trained on the 30-match corpus are research artefacts
   and cannot ship in a paid product. Resolved the same day: the Wyscout open
   corpus (CC BY 4.0, `soccerviz.datasets.wyscout_dataset`) is downloaded and
   the same VAEP protocol retrained on 300 of its 1,941 games reaches the same
   Brier skill ([Wyscout training](../experiments/wyscout-training.md)).
6. Metrica sample data carries no licence at all, so it is a lab input only.
   SkillCorner (MIT) is the one event/tracking source in the tree that is
   clean for commercial use.
7. The jersey model is non-commercial on both code and training data, so the
   third model in the workflow (detector, camera, jersey) is evaluation-only
   as well. Of the ten checkpoints in the workflow image only
   `rf-detr-medium.pth` (Apache-2.0 COCO weights) is cleared for shipping.
8. Mechanical guard, 2026-09-08: every checkpoint the video workflow loads
   must be registered by SHA256 in `soccerviz.vision.checkpoint_licence` with
   its source and class. An unregistered checkpoint refuses to run, so a
   replacement copied into the image under an old filename cannot inherit the
   old file's class. Every run's `configuration.json` and `report.json` carry
   a `licence` block naming the class and the blocking checkpoints, and
   `--shipping` refuses any run that is not fully cleared.

### Candidate sources checked 2026-09-08

Each row was read from the source's own terms file, API record or page
metadata on 2026-09-08 (curl, figshare API, Kaggle and Universe page data).
Sport was confirmed by viewing two sample images per Universe set
(`artifacts/public-data/roboflow/universe-previews/`). Nothing here is
downloaded yet.

| Gap | Source | Licence | What it is | Verdict |
| --- | --- | --- | --- | --- |
| Detection scale | `soccer-players/soccer-players-xy9vk` (Universe) | CC BY 4.0 | 5,761 broadcast screenshots (SCOUTINGFEED watermark, scoreboards blacked out), player / ball / goalkeeper / referee, but 19 class names with inconsistent naming across images (`player_team1`, `player`, bare digits); last modified 2024-02-17 | **Usable with a class-remap census first.** The RF-DETR trainer already remaps Roboflow ids; the 19-name merge is the cost |
| Detection scale | TeamTrack (`AtomScott/TeamTrack`, Kaggle) | MIT (repo and Kaggle) | 4K to 8K full-pitch static-camera video, soccer side and top views, over 4M boxes with track ids across soccer, basketball, handball | **Worth a look for small-player detection and tracking**, not broadcast footage, so evaluate on the 450-frame benchmark before training on it |
| Detection scale | SoccerTrack (`AtomScott/SoccerTrack`, Kaggle) | LGPL-3.0 data (Kaggle), GPL-3.0 code | Top-view and wide-view fish-eye plus drone footage with GNSS | **Drop.** Copyleft and not broadcast |
| Detection scale | Voxel51 `SoccerNet-V3` (Hugging Face) | MIT card on the FiftyOne packaging; images are SoccerNet | 1,799 SoccerNet samples | **Drop.** The MIT label does not relicense SoccerNet |
| Detection scale | SportsMOT (`MCG-NJU/SportsMOT`) | none stated | 240 clips cut from YouTube Olympic / NCAA / NBA broadcasts, soccer subset averages 674 frames | **Drop.** No licence and third-party broadcast video |
| Segmentation | Football-Player-Segmentation (Kaggle `ihelon`) | CC0 | Pixel masks of players in football matches; image count not shown on the page | Clean but small and segmentation-only; no consumer in the plan today |
| Event data | Wyscout open match-event dataset (figshare collection 4415000) | CC BY 4.0 on every article read (events, matches, players) | Events (77 MB zip), matches, players, teams, competitions for the 2017/18 Serie A, Premier League, La Liga, Ligue 1, Bundesliga plus Euro 2016 and World Cup 2018; Pappalardo et al. 2019 | **Brought in 2026-09-08** (`artifacts/public-data/wyscout`, MD5-verified); VAEP retrained on it |
| Tracking data | SkillCorner opendata | MIT | 10 A-League matches 2024-11-30 to 2025-05-17, tracking plus dynamic events and phases of play | Already in; small |
| Jersey numbers | `pusan-national-university-aajlj/jersey-number-detection-8a55j` (Universe) | CC BY 4.0 | 9,310 source images (v1 export: 24,315 train + 1,205 valid at 640 px, augmented), digit boxes in 10 classes. **Correction after download:** the crops are grayscale indoor volleyball, not soccer (eight validation samples viewed, `universe-previews/pusan-valid-contact.png`) | **Brought in 2026-09-08** as digit-detector training data; transfer to soccer is measured on the 200 labelled SoccerNet crops, never assumed |
| Jersey numbers | `taiseis-workspace/jersey-number-ijbaq` (Universe) | CC BY 4.0 | 4,114 tiles at 224 px, 103 number classes plus `no number` and `unreadble`; viewed on 48 train tiles 2026-09-08: colour broadcast crops, basketball-heavy with some football | **Brought in 2026-09-08** (v1, `folder` export) as the primary classifier source; see [jersey](../experiments/jersey-model.md) |
| Jersey numbers | `rematch/jersey-number-detection-y4neo`, `rematch/jersey-number-region` (Universe) | CC BY 4.0 | 2,000 and 674 soccer crops, one class: number region | Not downloadable: neither project has an exported version (API checked 2026-09-08) |
| Jersey numbers | `soccer-27vki/jersey-qyuun` (Universe) | Public Domain | 3,613 images, but the samples are ice-hockey product photos despite the name | **Drop** |
| Jersey numbers | `videoanalysis-ncvpt/…`, `data-labeling-cd6r4/…` (Universe) | CC BY 4.0 | Baseball (MLB) crops | **Drop.** Wrong sport |

Broadcast screenshots carry the broadcaster's copyright whatever licence the
Universe uploader attached; that risk is the same for the v10 corpus already
in use and is not resolved by any of the rows above.
