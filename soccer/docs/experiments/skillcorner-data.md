# SkillCorner public tracking and dynamic events

The adapter imports the actual SkillCorner V3 open-data release from [SkillCorner/opendata](https://github.com/SkillCorner/opendata), pinned to commit `c1e17a0cc3e07e1774b52d929c1a0b85115143fc`. It reuses the published tracking, Dynamic Events, and phases of play. These are provider/model-derived data, not human ground truth for our perception pipeline.

The upstream corpus contains ten Australian A-League 2024/25 matches. `split-manifest.json` sorts the entire catalog by match date and ID, assigning the earliest six games to training, the next two to validation, and the last two to test. This is our deterministic development split, not an official SkillCorner split. Games cannot cross splits, but teams and players recur.

## Run

No new dependencies are needed beyond the base SoccerViz environment. Save a request such as:

```json
{
  "schema_version": 1,
  "operation": "fetch-import",
  "match_id": 1886347,
  "cache": "artifacts/public-data/skillcorner",
  "out": "artifacts/integrations/my-skillcorner-game",
  "max_frames": 600
}
```

```sh
PYTHONPATH=src .venv/bin/python -m soccerviz.providers.skillcorner \
  --request request.json --response skillcorner-response.json
```

`out` must be a new directory. `max_frames` bounds the complete JSONL prefix downloaded, including empty/pre-match frames. Full Dynamic Events and phase CSVs are downloaded, with a 12 MB bound per file. The source match's tracking file can be roughly 89 MB; the default does not fetch it all. Increase `max_frames` deliberately when more coverage is needed.

For offline imports use `operation: "import"` with local `metadata`, `tracking`, `events`, `phases`, `matches`, and `out` paths. The optional `source_manifest` verifies every input against a previously captured download manifest. Without it the report explicitly says that local files were hashed without upstream verification. Download caching verifies the cached SHA-256 before reuse. Git LFS full-source hashes and sizes are separate from the downloaded prefix hash; a prefix is never reported as a fully verified LFS object.

## Output contract

- `observations.parquet`: long-form players and ball, source match/frame/period/timestamps, provider entity/team/trackable IDs, actual metre coordinates, detection flag, missingness, and provider provenance.
- `frames.parquet`: exact source frame clocks, source possession player/group, and untouched camera projection polygon JSON. Null periods and timestamps survive ingestion.
- `events.parquet`: every original Dynamic Events column plus a match-qualified event UID and tracking coverage flags.
- `phases.parquet`: complete published phase intervals with match-qualified IDs and coverage flags.
- `match-metadata.json`: original pitch, rosters, period boundaries, and team orientation metadata.
- `split-manifest.json`: deterministic ten-game catalog and assignments.
- `report.json`: verified source hashes, output hashes, missingness and coverage, event/subtype counts, and sums of published in-possession phase durations.

Original CSV attributes remain nullable strings, except frame boundaries/periods and durations, which are typed for interval queries. This retains identifiers and missing attribute values without silent float coercion. Event IDs are only unique within a match, so downstream joins must use `event_uid` or `(match_id, event_id)`.

Tracking `status` is `detected`, `extrapolated`, `unknown_detection_status`, or `unavailable`. A provider's `is_detected=false` position is retained as extrapolated rather than being scored as an observed player. Missing expected players remain explicit rows according to roster playing intervals; unused substitutes are not counted as missing detections. Nothing is interpolated.

Tracking coordinates retain the original centered pitch axes and actual pitch dimensions. Dynamic Events follow a separate coordinate convention: possession-team orientation is normalized left-to-right, except the original `attacking_side` fields. This is documented in the [official Dynamic Events specification](https://26560301.fs1.hubspotusercontent-eu1.net/hubfs/26560301/Guides/Dynamic%20Events/20250216%20-%20Dynamic%20Events%20CSV%20Specifications.pdf), page 11. The adapter deliberately does not join tracking/event positions or assume a 105×68 pitch. Native V3 parsing preserves source fields; it does not force a lossy Kloppy conversion.

`timestamp_s` is parsed from the exact match timestamp. `source_frame_time_s = frame_id / 10` remains separate because these clocks can differ, notably around pre-match footage and half-time. Event frame references are never replaced by synthetic timestamps. `tracking_interval_fully_available` means the complete inclusive source frame interval exists in the imported prefix; it does not imply every player or ball position is detected or that provider identities are correct.

## Executed real-data checks

| Match | Split | Imported source frames | Frames with player positions | Full dynamic events | Off-ball runs | Published phases | Events with complete frame coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| 1886347: Auckland–Newcastle | train | 600 | 590 | 5,079 | 599 | 454 | 65 |
| 2011166: Wellington–Melbourne Victory | validation | 2,900 | 499 | 3,966 | 437 | 429 | 90 |

The second game starts at frame 2,300, so its bounded prefix includes substantial pre-match footage. The importer exposed this rather than substituting invented observations. Only 499 frames in that prefix contain player positions. The two complete event files contain 9,045 published events and 883 phase intervals. These counts establish working ingestion and descriptive summaries, not tactical accuracy.

Artifacts are in `artifacts/integrations/skillcorner/game-1886347` and `artifacts/integrations/skillcorner/game-2011166`. Reproducible source and artifact hashes are recorded in `results/experiments/skillcorner-1886347.json` and `results/experiments/skillcorner-2011166.json`.

The eleven adapter tests cover source clocks/axes, detection versus extrapolation, inactive substitutes, missingness, game-disjoint splits, interval coverage, source tampering, cross-match events, invalid identities/frame IDs/flags, download limits, and worker operation rejection. The provider's run/phase labels can drive UI exploration and hypothesis development. They cannot independently validate the same tracking-derived hypotheses.

## Shared harness catalog smoke

The unified worker dispatch was also executed on the cached Auckland–Newcastle files using `results/experiments/skillcorner-cli-request.json`:

```sh
.venv/bin/soccerviz datasets run skillcorner \
  --request results/experiments/skillcorner-cli-request.json \
  --response results/experiments/skillcorner-cli-response.json \
  --name 'SkillCorner Auckland Newcastle public-data smoke' --split train \
  --artifact artifacts/integrations/skillcorner/cli-game-1886347/observations.parquet \
  --artifact artifacts/integrations/skillcorner/cli-game-1886347/events.parquet \
  --artifact artifacts/integrations/skillcorner/cli-game-1886347/phases.parquet \
  --artifact artifacts/integrations/skillcorner/cli-game-1886347/frames.parquet
```

This registered immutable catalog record `6d20f494aa5df72534fed75924f8a915f12addab40c383dd42ec3216af5e1e3a`; `soccerviz datasets verify` confirmed all four Parquet files and the response report matched their registered hashes. The request uses this workspace's absolute cache paths. To rerun, choose fresh `out` and response paths; immutable reports are intentionally not overwritten.
