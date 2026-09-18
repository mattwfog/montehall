# SoccerNet public benchmarks

The project can now import SoccerNet GSR 2024 and SoccerNet Tracking 2023, preserve their official split and source frame IDs, and run official tracking metrics. Labels remain benchmark targets; they are not fed into detector, identity, team, or pitch estimation.

## What has actually run

- Downloaded **SNGS-021 from the official GSR validation split**, annotation version **1.3**, plus 50 images at source frames 1, 6, …, 246. Source images are 1920×1080 at 25 Hz; this sample spans 9.8 seconds at 5 Hz. Downloaded **19,922,034 bytes** using HTTP ranges, rather than the entire 11.17 GB validation archive.
- Downloaded **SNMOT-060 from the official Tracking train split**, all 750 frames' ground truth, provided detections, and sequence metadata. Downloaded **4,838,654 bytes**, rather than the entire 9.58 GB train archive.
- Ran our anonymous tracklet associator on the 13,540 official provided detections. It produced 176 tracklets. Official TrackEval reported **HOTA 0.83735**, **IDF1 0.79350**.
- **That tracking run is an association diagnostic using oracle boxes.** The adapter compared every detection's frame and coordinates with ground truth and found exact equality. It does not measure detection accuracy, does not show held-out accuracy, and must not be presented as the accuracy of our video pipeline.
- Ran the actual Spark video model on all 50 sampled SNGS-021 validation frames: **image HOTA 0.50457, IDF1 0.59246**, with 843 predicted people and 801 labeled people (ball excluded by the evaluator). The full sampled-clip GS-HOTA is unavailable because 250 predictions lack accepted pitch geometry.
- Ran the official **GS-HOTA** on a separately labeled conditional sample: 35/50 frames (70%) for which all predictions have finite model-derived pitch geometry. It scores **0.0139608** (1.40%), with 550 ground-truth people and 579 predicted people. Frames were selected using model predictions only. This is conditional on calibration success and biased toward easier frames; it is not a full-clip or official-split score. Null team/jersey predictions cannot match player identities; non-player roles can still match. The complete result is `artifacts/benchmarks/sngs-021-conditional-gsr-evaluation.json`, with the exact request and selection coverage alongside it.
- Model pretraining overlap with SoccerNet is unknown. These are development measurements, not independent held-out generalization claims.
- The isolated official SoccerNet evaluator passes tests covering attribute gating, missing pitch geometry, original source image mapping, annotation versions, and preserved clocks.

Artifacts live in `artifacts/public-data/soccernet/`. Download manifests record the pinned Hugging Face revision, archive size/expected hash, each extracted member's SHA256 and ZIP CRC32, and actual transferred bytes. The whole archive hash is explicitly marked **not verified**, because the whole archive was not downloaded.

## Set up the official GSR evaluator

SoccerNet uses a fork of TrackEval which installs the same `trackeval` module. Keep it separate from `.venvs/evaluation` (standard TrackEval):

```sh
uv venv --python 3.12 .venvs/soccernet
uv pip install --python .venvs/soccernet/bin/python -r envs/soccernet/requirements.txt
```

The official fork is pinned to commit `9c25232f6f2b56c9f203f1eb55784ff1e97df683`. The adapter checks its installed commit before running GS-HOTA.

## Request / response workers

Both modules use fixed JSON requests and write JSON responses:

```sh
PYTHONPATH=src .venv/bin/python -m soccerviz.datasets.soccernet_adapter \
  --request request.json --response response.json

PYTHONPATH=src .venvs/soccernet/bin/python -m soccerviz.datasets.soccernet_evaluation \
  --request request.json --response report.json
```

The root `soccerviz datasets run soccernet` and `soccerviz datasets run soccernet-evaluate` commands dispatch these same workers and register immutable results.

Download one official sequence:

```json
{
  "operation": "download",
  "kind": "gsr",
  "split": "valid",
  "sequence": "SNGS-021",
  "out": "artifacts/public-data/soccernet/gsr",
  "image_count": 50,
  "image_stride": 5,
  "max_bytes": 50000000
}
```

Use `kind: "tracking"`, `split: "train"`, and `sequence: "SNMOT-060"` for tracking. `image_count` defaults to zero so annotation inspection is cheap. The downloader verifies sequence membership in the selected split, pins the repository revision before fetching, rejects servers which ignore HTTP ranges, checks a byte budget, and verifies member CRCs. Test/challenge annotations may not be available publicly; no labels are fabricated and local tracking evaluation fails explicitly if ground truth is absent.

Import annotations without model inference:

```json
{
  "operation": "import-gsr",
  "labels": "artifacts/public-data/soccernet/gsr/valid/SNGS-021/Labels-GameState.json",
  "split": "valid",
  "sequence": "SNGS-021"
}
```

GSR import requires annotation version ≥1.3, reads `info.frame_rate`, retains the complete source `info` including clip and match clock metadata, and rejects inconsistent frame-rate overrides. `frames[].timestamp_s` is clip-relative `(source_frame-1)/source_fps`. Source pixel `xywh` boxes and centered metric `bbox_pitch` coordinates are preserved. No rescaling or team relabeling occurs.

Tracking import request:

```json
{
  "operation": "import-tracking",
  "sequence_dir": "artifacts/public-data/soccernet/tracking/train/SNMOT-060",
  "split": "train"
}
```

## Evaluate genuine GSR predictions

Supply the **official JSON prediction format** with original source `image_id` values. Each object needs its predicted `track_id`, `category_id`, `supercategory`, `bbox_image`, and `attributes` (`role`, `team`, `jersey`). GS-HOTA additionally needs finite predicted `bbox_pitch` bottom-left/middle/right positions. Unknown attributes remain null; never look up ground-truth team or jersey labels to improve a prediction.

```json
{
  "operation": "gsr",
  "labels": "artifacts/public-data/soccernet/gsr/valid/SNGS-021/Labels-GameState.json",
  "predictions": "artifacts/public-data/soccernet/sngs021-predictions.json",
  "split": "valid",
  "sequence": "SNGS-021",
  "frame_ids": [1, 6, 11]
}
```

`frame_ids` must list every sampled source frame to score, including frames with no detections. Omitting it evaluates the full sequence. Original image IDs are checked rather than guessed or shifted. Official annotation coverage flags determine scored frames, and ball is excluded by the official GSR evaluator.

- `image` reports official HOTA/IDF1 in image coordinates, with semantic attributes disabled.
- `gs_hota` reports official GS-HOTA with the default 5 m Gaussian tolerance and role/team/jersey matching enabled.
- If any prediction lacks metric pitch geometry, `gs_hota` is null with an explicit reason. Pixel metrics still run. Unprojectable detections are never silently removed to inflate a pitch score.
- Every score is 0–1. Subset runs are not full official split leaderboard results.

## Association-only tracking diagnostic

First run `operation: "track-public-detections"` with `sequence_dir`, `split`, and `out` (prediction JSON path). Then run `operation: "tracking"` with `sequence_dir`, `split`, and `predictions` using `.venvs/evaluation/bin/python`. The prediction file must carry the matching sequence and split. `oracle_boxes` is machine-readable in both prediction and evaluation artifacts.

## Next scale-up

Run additional clips chosen before seeing their results, retain the official split, and aggregate complete per-sequence metric outputs with the official evaluator before making accuracy claims. The present sample is a real integration check. It does not validate coaching quality, ball possession, generalization across games, or an entire reconstruction pipeline.

Sources: [SoccerNet GSR](https://github.com/SoccerNet/sn-gamestate), [official GS-HOTA evaluator](https://github.com/SoccerNet/sn-trackeval), [SoccerNet Tracking](https://github.com/SoccerNet/sn-tracking), [official downloader](https://github.com/SoccerNet/SoccerNet/blob/master/SoccerNet/Downloader.py), [GSR dataset](https://huggingface.co/datasets/SoccerNet/SN-GSR-2024), [Tracking dataset](https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023). Follow each upstream project's data access and usage terms.
