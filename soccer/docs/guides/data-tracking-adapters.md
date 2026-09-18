# Data, tracking, and action-value specialists

These adapters call upstream libraries through a narrow Parquet/JSON boundary. The
core harness remains the owner of evidence and revisions. The experimental tracker
has **not** been promoted: on the frozen 5 Hz demo it produced substantially more
fragmented tracks than the existing associator, and reviewed identity labels are
still needed for an accuracy comparison.

## Runtime and compatibility

Use the persistent analysis worker created by the repository setup script. This
keeps the incompatible optional dependencies out of the core `.venv`:

```sh
.venv/bin/python scripts/setup/setup_integrations.py analysis
PYTHONPATH=src .venvs/analysis/bin/python -m pytest tests/test_tracking_adapter.py tests/test_provider_adapter.py tests/test_action_value_adapter.py
```

The setup script installs the tested requirements into `.venvs/analysis` and
installs TrackLab's packaged plugin separately with `--no-deps`. Tested on Python
3.12.12 and macOS arm64. To dispatch from the main application:

```sh
.venv/bin/soccerviz integrations provider \
  --home data/raw/game_1/Sample_Game_1_RawTrackingData_Home_Team.csv \
  --away data/raw/game_1/Sample_Game_1_RawTrackingData_Away_Team.csv \
  --out artifacts/integrations/provider-new --limit 500 \
  --python .venvs/analysis/bin/python
```

`envs/integrations/requirements.txt` contains tested direct pins; the complete observed
installation is recorded in `results/experiments/integrations-runtime.txt`. Do not
install this runtime into the core environment: socceraction 1.5.3 requires NumPy
below 2, while SoccerViz requires NumPy 2.1 or newer. Pandera 0.17.2 also needs
`multimethod<2`; the tested pin is 1.12. The tracker imports the `oc_sort` plugin
packaged in TrackLab 1.3.24, with Torch, FilterPy and LAP installed explicitly.
Installing TrackLab with `--no-deps` is intentional: its full application pulls in
many unrelated detector, model-hub, visualization and training packages. We do not
claim the full TrackLab CLI works in this minimal worker. GPU execution has not
been tested by these adapters; the current OC-SORT path is CPU association.

The public socceraction Kloppy bridge failed with Kloppy 3.19.0:
`_SoccerActionCoordinateSystem() takes no arguments`. Supported StatsBomb action
conversion therefore uses socceraction's own local StatsBomb loader and converter.
Kloppy independently handles provider tracking. No upstream implementation is copied.

## Worker protocol

Each module accepts `--request /absolute/request.json --response /absolute/report.json`.
Request and response schema version is 1. All artifact output destinations must be
new; existing outputs are rejected. Relative paths resolve from the worker's current
directory, so orchestration should use absolute paths. Run with `PYTHONPATH=src` to
load this checkout without installing the incompatible core dependency bundle.

### Frozen-detection tracker comparison

```json
{
  "schema_version": 1,
  "video_dir": "/absolute/artifacts/video/demo-v2",
  "out": "/absolute/artifacts/integrations/comparison-new",
  "max_age_s": 0.8
}
```

Invoke `python -m soccerviz.providers.tracking`. Python API:
`compare_trackers(video_dir: Path, out: Path, max_age_s=0.8) -> dict`.

The adapter runs both the existing `TrackletAssociator` and the actual
`oc_sort.ocsort.OCSort` supplied by TrackLab on exactly the same detections. Neither
re-runs detection or changes boxes, confidences, timestamps or detection IDs.
Association resets at camera cuts and runs separately for each role. Source
frames containing no detections still advance association state. OC-SORT's motion
model assumes one regular time step per update; irregular within-shot sampling is
rejected explicitly. The chosen max age is converted from seconds using observed
sample intervals. OC-SORT is configured with zero detection threshold and zero
minimum hits so the comparison retains every frozen detection; this configuration
is a baseline, not a tuned broadcast-soccer model.

`out/soccerviz/` and `out/tracklab-ocsort/` contain `detections.parquet`, remapped
`state.parquet`, and immutable upstream frame/calibration/landmark/ball-candidate
artifacts plus `report.json`. These directories can be imported with the existing
video harness. Original source video and model hashes survive in the report;
frozen detection/frame hashes and backend versions are recorded as lineage.
Track-voted team assignments are cleared because their votes depended on the old
association. Ball hypotheses and their evidence remain unchanged. Derived tactical
artifacts are not copied. The comparison report has no HOTA or IDF1 until an
independent reviewed ground-truth set is supplied to TrackEval.

Actual final output: `artifacts/integrations/tracker-comparison-final/comparison.json`:
2,318 detections per backend; existing association 43 tracklets (14 singletons),
OC-SORT 457 (323 singletons). Track count is a continuity diagnostic, not identity
accuracy. Frozen artifacts lack shirt-color vectors, so both comparison backends
run without appearance features. The saved original tracker used appearance and
is therefore not identical to this fair replay baseline.

Each backend also exports `predictions.json`, a normalized evaluation envelope
containing the original video `source_sha256`, the frozen detection hash and
`predictions: [{frame_id, track_id, label, bbox}]`. The prediction rows use the
same conversion function as the evaluator's original-run baseline. Both envelopes
preserve the identical frozen frame/label/box set; only association IDs differ.
Absolute prediction paths and SHA-256 hashes are included in `comparison.json`
and each backend's `report.json`. The original reviewed input can therefore score
both candidates directly:

```sh
.venv/bin/soccerviz integrations evaluate RUN_ID \
  --reviewed /absolute/reviewed-tracking.json \
  --predictions artifacts/integrations/tracker-comparison-final/soccerviz/predictions.json \
  --out /absolute/soccerviz-evaluation.json --python .venvs/evaluation/bin/python
.venv/bin/soccerviz integrations evaluate RUN_ID \
  --reviewed /absolute/reviewed-tracking.json \
  --predictions artifacts/integrations/tracker-comparison-final/tracklab-ocsort/predictions.json \
  --out /absolute/ocsort-evaluation.json --python .venvs/evaluation/bin/python
```

`RUN_ID` must be the original harness video run whose source was reviewed. Use
`.venv/bin/python scripts/setup/setup_integrations.py evaluation` to install TrackEval's
isolated runtime. Only explicitly reviewed exhaustive frame/class coverage is
scored. Exporting an envelope alone does not produce accuracy metrics.


### Kloppy provider tracking

```json
{
  "schema_version": 1,
  "home": "/absolute/data/raw/game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
  "away": "/absolute/data/raw/game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
  "out": "/absolute/artifacts/integrations/provider-new",
  "sample_rate": 0.2,
  "limit": 500
}
```

Invoke `python -m soccerviz.providers.provider`. Python API:
`export_metrica(home: Path, away: Path, out: Path, *, sample_rate=.2, limit=None)`.
`load_metrica(...)` returns `(observations_dataframe, report)` without writing.

`sample_rate` is a fraction of source frames; 0.2 gives 5 Hz for the 25 Hz sample.
`limit` is the maximum number of output frames. Every registered player and the
ball get an explicit row in every selected frame, including unavailable positions.
The adapter never fills absent positions with zero or an interpolated trajectory.
Kloppy's IDs remain `provider_entity_id`; they do not claim named-player identity.

Kloppy's Metrica CSV parser synthesizes its internal time from frame ID / 25 and
flips the raw Y axis. The adapter checks home/away source clock agreement, rejoins
**actual CSV Time [s]** by frame ID, retains Kloppy's timestamp separately, and
returns the existing SoccerViz top-left 105 × 68 metre convention. A test uses
irregular source timestamps to establish that the library's synthesized clock
cannot replace source evidence. Local game 1 verification preserved 500 frames,
28 registered players, 11,318 observed positions and 3,182 unavailable positions.
Output is `observations.parquet` plus hash-bearing `report.json`.

### SPADL and action value

For supported raw StatsBomb data:

```json
{
  "schema_version": 1,
  "operation": "statsbomb-to-spadl",
  "events": "/absolute/events.json",
  "lineup": "/absolute/lineup.json",
  "game_id": 3788741,
  "home_team_id": 909,
  "out": "/absolute/actions.parquet"
}
```

Invoke `python -m soccerviz.providers.action_value`. Conversion uses the library's
public local loader and SPADL converter, followed by its `play_left_to_right`.
Raw files are not modified. Actual official StatsBomb open-data match 3788741
converted to 2,206 validated SPADL actions; the source hashes and conversion result
are recorded in `results/experiments/statsbomb-spadl-report.json`. The downloaded
fixture was temporary and is not redistributed here. Observe StatsBomb's open-data
attribution terms when using or publishing results from those data.

For xT/VAEP preparation from reviewed SPADL files:

```json
{
  "schema_version": 1,
  "train": "/absolute/training-actions.parquet",
  "evaluate": "/absolute/evaluation-actions.parquet",
  "orientation": "attacking_left_to_right",
  "data_kind": "observed",
  "grid": [16, 12],
  "out": "/absolute/action-baseline-new"
}
```

The adapter validates SPADL with socceraction's schema, rejects shared train/eval
match IDs, checks source action order, requires shots for fitting, and calls
`ExpectedThreat.fit` and `.rate`. Successful movement actions receive xT values;
other actions remain unrated following upstream semantics. Output includes the
model grid with hash, action values, and separate VAEP training feature and label
files. The labels contain future outcomes and must never enter inference evidence.
No VAEP classifier is silently trained. The default report states `vaep_status:
"untrained"` and has no claimed real-world value accuracy.

Optional `probabilities` points to a Parquet table with unique `(game_id, action_id)`
keys and `scores` / `concedes` probabilities for every evaluated action. Supply
`vaep_provenance` containing `model_sha256` and `training_game_ids`. The adapter
rejects training overlap, missing probabilities, and values outside [0,1], then
calls socceraction's VAEP formula separately for each game/period. Producing
probabilities from a trained, calibrated classifier is a separate specialist's job.

A deterministic synthetic train/evaluation fixture exercised actual xT fitting
and VAEP preparation under `artifacts/integrations/action-value-synthetic/`; it
produced 20 rated movements out of 30 evaluation actions. Synthetic output is
explicitly labeled and cannot establish soccer decision quality. The local Metrica
CSV events are not a supported socceraction provider and are not guessed into
SPADL.

A second real experiment uses distinct official StatsBomb Euro 2020 matches:
Turkey–Italy (3788741, 2,206 actions) for training and Denmark–Finland
(3788742, 1,953 actions) for evaluation. The upstream xT fit converged in 57
iterations and rated 1,476 successful evaluation movements. Source hashes, match
IDs and outputs are recorded in `results/experiments/statsbomb-xt-report.json`;
artifacts are under `artifacts/integrations/action-value-statsbomb/`. This proves
the real conversion/fit/rate pipeline operates across disjoint matches. One
training match is insufficient for a production value model, and no accuracy or
decision-quality claim follows from these counts. VAEP remains untrained.

## Upstream references

- [TrackLab official source and installation](https://github.com/TrackingLaboratory/tracklab)
- [TrackLab OC-SORT wrapper contract](https://github.com/TrackingLaboratory/tracklab/blob/main/tracklab/wrappers/track/oc_sort_api.py)
- [Kloppy Metrica loading](https://kloppy.pysport.org/user-guide/loading-data/metrica/)
- [socceraction expected threat](https://socceraction.readthedocs.io/en/latest/documentation/valuing_actions/xT.html)
- [socceraction StatsBomb loader](https://github.com/ML-KULeuven/socceraction/blob/master/socceraction/data/statsbomb/loader.py)
- [StatsBomb open data and attribution](https://github.com/statsbomb/open-data)
