# CV Brain — state + token contract (v1)

Ratified direction (2026-07-17→18): vision is a primitive,
not the brain. The brain is a sequence model over tokenized observation
streams that outputs latent game state + events; every sensor (detector,
OCR, audio, score bug, PBP at training time) is a swappable observation
channel. This document is the contract every extractor writes into and the
brain reads from. Code home: `montehall_cv/brain/` (schema in
`store/schemas.py::TOKENS_SCHEMA`).

## 1. Latent state (the brain's output space)

Per time step the brain maintains a belief over:

- **Lineup slots**: 10 on-court entity slots × court position (ft,
  court frame as in `POSITIONS_SCHEMA`), each slot carrying a
  distribution over roster identities (the permutation). Identity is
  never forced: slots may be anonymous (entity-first, late binding —
  the 2026-07-09 ruling stands).
- **Ball**: court position + mode ∈ {held, dribbled, ballistic, dead}
  (the latent-state vocabulary), holder slot
  when mode=held.
- **Possession**: team cluster in control (observed int per
  `POSSESSIONS_SCHEMA`; which basket is attacked stays a downstream
  join).
- **Clock**: (period, clock_s) belief; **Score**: (home, guest).
- **Segment boundary**: dead-ball/whistle state — the only points where
  lineup membership may change (substitution structure).

Event outputs use the ESPN PBP vocabulary as-is at v1 (`type_text`
passthrough + shooting/scoring/score_value flags — the exact fields
the PBP scorer grades). A canonical enum is deferred until the brain's
error attribution demands one.

## 2. Observation channels (token vocabulary)

One parquet row per token (`TOKENS_SCHEMA`), ordered by
`(t_ms, channel_priority, entity_id)` with `token_idx` the stable
ordinal. `t_ms` derives from container pts (VFR-safe, records.py timing
contract) or `video_t*1000` for alignment-derived channels.

| channel | source artifact | rate | live/train | fields used |
|---|---|---|---|---|
| `clock` | `clock_reads` (conf ≥ 0.85) | native (~2s stride) | live | value=clock_s, period, conf |
| `cut` | gaps > 5s between confident clock reads | per gap | live | value=gap_s (replay/cut heuristic) |
| `score_delta` | `score_events` | per change | live | text=side, value=delta, conf |
| `rim` | `localized` cls=RIM | 1 Hz grid | live | court_x/y, conf |
| `player` | `positions` cls=PERSON | 5 Hz grid per entity | live | entity_id, team_cluster, court_x/y, conf=court_conf |
| `ball` | `positions` cls=BALL | 5 Hz grid | live | court_x/y |
| `ball_control` | `ball_controls` | 5 Hz grid per entity | live | entity_id, team_cluster, value=dist_ft |
| `jersey_read` | `identity` + `contact_reads` (2026-08-01 widening — payload `stage` distinguishes; contact verdicts emit at track start, per-tracklet aggregate) | per accepted read / per read tracklet | live | text=candidate, conf=prob, payload={track_id, stage[, n_crops]} |
| `name_call` | `name_calls` (whisper + exact surname match) | per match | live | text=surname, conf, payload={athlete_ids, jerseys, ambiguous} |
| `pbp_anchor` | `pbp_alignment` | per aligned play | **TRAIN-ONLY** | text=type_text, value=score_value, period, payload={play_id, athlete_ids, shooter_jersey, coords, flags} |

`TRAIN_ONLY_CHANNELS = {pbp_anchor}` — the brain's dataloader includes
them as supervision targets, never as inference inputs. Future channels
append here (sim traces ride the same schema with `game_key`
`sim_<trace_id>`; face or any other sensor would be one more row type —
sensors are policy, the schema doesn't change).

## 3. What is NOT a token (confabulation guard)

Model verdicts and derived layers of the old pipeline never enter the
stream: `shot_events`, `box_events`, `box_score`, `possessions`,
`atomic_events`, `plays`, harness verdicts. The brain learns from
primitives and external truth only — an estimator fed its predecessor's
conclusions inherits its predecessor's hallucinations (the legacy-pipeline lesson).

## 4. File format + mechanics

- Stage dir `tokens/` inside the game dir (align-only games: inside
  `align/eid_<id>/`; full jobs: inside the job dir), ArtifactWriter
  parts + `_SUCCESS`, plus `tokens_meta.json` (version, per-channel
  counts, sources present). Resumable: `_SUCCESS` skips; zero tokens =
  refusal, not success (the extract 0-det rule).
- Downsample grids: `PLAYER_GRID_MS = 200` (5 Hz), `RIM_GRID_MS = 1000`. First row per
  (channel, entity, bucket) wins.
- Density tiers are expected and fine: align-only games emit
  clock/cut/pbp tokens today; perception channels appear when a
  per-game extract pass exists. The brain trains on what's present
  (channel dropout is a feature — it's the sensor-swap property).
- Name-call matching is EXACT surname token match against the ESPN
  boxscore roster (lowercased, word-boundary; surnames < 3 chars
  skipped). Shared surnames emit one token with `ambiguous=true` and
  all candidate athlete_ids — never a guess.

## 5a. Coach annotation channel (ratified direction 2026-07-31; cells open)

Coach-defined annotations (beyond identity votes) become an observation/
supervision channel per thesis §11. The schema row lands here only after
the thesis §11 cells resolve (vocabulary, per-team model form, thresholds) —
do not freeze field semantics before those rulings. Expected shape rides
§2 mechanics unchanged: one row per annotation, `channel=coach_annotation`,
payload carrying {concept, segment span, author, vote/edit lineage}.
Identity votes (collected in production since 07-10) are the
precedent and become permutation-training anchors under thesis §10.4.

## 5. Sequencing note (ratified 2026-07-18)

Brain v0 trains on token streams (hours per iteration on one DGX Spark);
detector-ladder rungs and other primitive upgrades fire when the brain's
error attribution names them — the detector ladder is parts supply, not the
spine. Eval stays PBP-truth scoring on sealed holdouts plus
precision-at-coverage per stat line (abstention counted as honesty).
Tripwire: v0 must snap, not grind (match/beat v3 event recall .94 within
~2 weeks of first training run, else the token schema is missing
structure — do not add math).
