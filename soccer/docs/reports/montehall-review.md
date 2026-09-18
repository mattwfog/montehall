# Montehall reference review

Inspected 2026-09-07. Mission: establish what the existing basketball CV work contributes
to a soccer project for elite tactical analysis and managerial decision support.

This is a source-code and document review, not an evaluation of deployed
models. Historical numbers in its documents were not remeasured and are not
presented here as current performance. The review was written against an
earlier tree than the one in `basketball/`, so some details differ.

## The relevant intent

[The brain thesis](../../../basketball/docs/cv-brain-thesis.md)
places the product in a learned game-state estimator, with vision, OCR, audio,
and other observations supplying evidence. Its July 31 extension explicitly
describes coach-defined annotations for schemes and custom events as training
supervision. That is the closest existing precedent for this project's tactical
ambition; it is a documented direction, not proof that coach-personalized models
are operational.

The [token contract](../../../basketball/docs/cv-brain-token-contract.md)
preserves per-entity observations and separates training-only play-by-play
anchors from inference inputs. It also excludes downstream model verdicts from
the primitive evidence stream. These are useful precedents for keeping tactical
claims traceable to what was actually observed.

## What exists in code

All paths below are relative to `basketball/` in this repository.

| Area | Source | Finding and soccer relevance |
| --- | --- | --- |
| Video timing | `montehall_cv/pipeline/video.py` | PyAV decode uses presentation timestamps, with DTS fallback. Useful for matching video, tracking, audio, and event timelines. |
| Evidence and identity | `montehall_cv/store/records.py` | Detections, tracklets, identity hypotheses, and evidence are separate records. A tracking identifier need not imply a known player name. |
| Stage persistence | `montehall_cv/store/artifacts.py` | Buffered Parquet parts, temporary-file rename, completion markers, and per-column data census. Useful pattern; not a complete content-addressed cache or checkpoint system. |
| Perception | `montehall_cv/pipeline/detect.py`, `track.py` | RF-DETR detection and ByteTrack tracking, with BoT-SORT plus an own-trained ReID model as the upgrade path. Soccer suitability has not been measured. |
| Structured observations | `montehall_cv/brain/tokens.py`, `dataset.py` | Channel vocabulary and training-only PBP exclusion exist in code. The v0 featurizer aggregates players into team centroids, which loses individual spatial relationships. |
| Learned identity | `montehall_cv/brain/dataset_slots.py`, `train_slots.py` | Separate per-track GRU plus global-context GRU, scoring roster jersey embeddings. The training path uses simulated identity labels. This is not a tactical decision model. |
| Joint continuity | lineup solver (in the reviewed tree; not included in `basketball/`) | Mixed-integer segment-level assignment of bodies to anonymous slots, with team, overlap, motion, and jersey constraints plus abstention. It is an optimization solver, distinct from the learned GRU model. |
| Game state | `montehall_cv/pipeline/game_state.py` | Possession and attacking direction feed shot attribution. The module explicitly excludes lineup state and live/dead segmentation from its own v1 scope. |
| Evaluation isolation | `montehall_cv/eval/holdouts.py` | Named reserved games and development-source guards provide a precedent for separating training, tuning, and evaluation. |

## What cannot transfer unchanged

- The lineup solver fixes five slots per team. Soccer needs active-player
  membership over time, including substitutes and dismissals, rather than a
  permanent fixed count.
- The slot featurizer normalizes to a 94-by-50-foot court and caps observed
  tracks at 16. Soccer needs pitch geometry and track capacity appropriate to
  its footage; simply renaming the package would preserve wrong assumptions.
- Rim geometry, shot attribution, held/dribbled ball modes, clock behavior,
  and basketball event supervision need sport-specific treatment.
- Identity and tactical role are different targets: a player can remain the
  same person while changing position, responsibility, or role across phases.
- Anonymous but stable tracks can support some team-shape analysis. Named
  player comparisons and substitution advice need stronger identity evidence.

Broadcast reconstruction has incomplete spatial evidence. Positions outside
the camera view can be estimated, but should remain distinguishable from
observations. Silently treating those estimates as measured
positions would particularly undermine conclusions about defensive width,
far-side runs, and protection against counterattacks.

## What the managerial ambition adds

The following are implications of the goal, not an implemented architecture:

1. Recover who and what is visible, where they are on the pitch, and how much
   uncertainty remains across cuts, occlusions, and off-screen intervals.
2. Learn team and player behavior through time: phase-dependent shape, pressing,
   movement, interactions, and responses to opponents.
3. Relate that behavior to match context and outcomes, retrieve supporting
   sequences, and evaluate potential adjustments.

The third task is a separate evidence problem. Predicting what usually happens
does not by itself establish what would happen after changing a lineup or
tactic. Evaluation should distinguish predictive accuracy, analyst agreement,
and evidence that a recommendation improves decisions. Video alone also does
not provide all managerial context, such as availability and training load.

Montehall's useful lesson here is to measure the intended coaching output.
Detector accuracy alone cannot establish correct pressing analysis or useful
substitution advice. Any future evaluation should expose evidence coverage,
abstention, and failure cases alongside accuracy, with whole matches separated
across training and evaluation and explicit tests on unfamiliar teams/cameras.

## Soccer references checked

- [SoccerNet Game State Reconstruction](https://www.soccer-net.org/tasks/game-state-reconstruction)
  benchmarks recovering pitch positions and attributes such as role, team, and
  jersey number from broadcast footage. It is a relevant perception evaluation
  reference; success there would not establish tactical or managerial quality.
- [TacticAI, Google DeepMind](https://deepmind.google/blog/tacticai-ai-assistant-for-football-tactics/)
  uses player graphs for corner-kick prediction, retrieval, and suggested
  adjustments, evaluated with Liverpool FC experts. It supports the feasibility
  of a bounded tactical assistant, rather than proving a general match manager.
- [Metrica Sports sample data](https://github.com/metrica-sports/sample-data)
  provides sample tracking and event data. It is a candidate for exploring
  tactical calculations before a video extractor is ready, not a selected
  training corpus or an established paired-video source for this project.
