# montehall-cv

Entity-first CV pipeline for basketball game film, built from scratch on a
license-clean component menu (see `NOTICES.md`). The design record lives in `docs/`.

Core commitments the code enforces:

- **Tracklet-first**: `track_id` is the persistent object; identity (team/jersey/player)
  is late-bound via `IdentityHypothesis` + `Evidence` rows — committed identity is a
  projection, never a stored primitive.
- **VFR-safe timing**: `ts_ms` comes from container pts. Never derive time from
  frame counts (the corpus contains variable-frame-rate captures).
- **Incremental + resumable stages**: every stage writes parquet part files as rows
  are produced and stamps `_SUCCESS` on completion; re-runs skip completed stages.
- **License floor**: MIT/Apache/BSD only, weights included. No ultralytics, no boxmot,
  no CC-BY-NC weights. Anything fine-tuned ships on our own or clean data.

## Where it runs

This is a standalone uv project. Development ran on an NVIDIA DGX Spark (GB10, aarch64).
GPU execution happens inside an NGC PyTorch container (torch preinstalled, aarch64):

```bash
pip install -e ".[gpu]"   # inside the container; reuses the image's torch
python -m montehall_cv.pipeline.run_extract \
    --video /data/video.mp4 --out /data/out --job-id <uuid>
```

Local (non-GPU) development installs the base package only; detection/tracking
imports are lazy so the store/court/video modules work without torch:

```bash
uv sync && uv run pytest
```

## Layout

- `montehall_cv/store/` — record contracts, pyarrow schemas, resumable parquet stage artifacts, VLM verdict cache
- `montehall_cv/pipeline/` — the perception and state chain, video to box score:
  - decode, detect, track: `video.py`, `detect.py` (RF-DETR), `track.py` (ByteTrack), `run_extract.py`
  - court: `court_seg.py`, `homography.py`, `run_court.py`, canonical space in `court.py` and `zones.py`
  - the three state stores: `ball_track.py` (the ball spine), `person_registry.py` (people-only
    identity), `game_state.py` (possession, period, attack direction)
  - continuity under the registry: `quark.py` (per-body seekers with a calibrated physics layer)
  - shots and stats: `rim.py`, `flight.py`, `run_shots.py`, `run_freethrows.py`, `shot_attribution.py`,
    `assists.py`, `run_boxscore.py`
  - identity evidence: `jersey.py`, `run_jersey.py`, `contact_sheet.py`, `event_identity.py`, `embed.py`
  - `run_all.py` — full-chain driver; `render.py` — annotated video output
- `montehall_cv/brain/` — token vocabulary and tokenizer, simulated state traces, announcer name-call
  mining, and the recurrent state estimators (events, ball state, per-track identity slots) with their evals
- `montehall_cv/eval/` — score-bug clock OCR, play-by-play alignment to video time, truth files, sealed holdouts
- `montehall_cv/harness/` — possession adjudication over symbolic possession traces, with swappable
  model backends (`haiku`, `jev`), calibration scoring and a side-by-side comparison
- `montehall_cv/training/` — dataset builders, broadcast label mining, synthetic jersey data, and the
  detector, OCR and ReID training scripts
- `montehall_cv/runner/` — job runner and results adapter
- `scripts/` — probes and one-off experiments; `scripts/score_e2e_vs_anchors.py` is the single scorer
  that writes `SCORECARD.json`
- `docs/` — start with [`cv-state-estimation-paper.md`](docs/cv-state-estimation-paper.md); the
  [thesis](docs/cv-brain-thesis.md), [token contract](docs/cv-brain-token-contract.md) and
  [identity design](docs/cv-identity-design.md) are the working design record
- `tests/` — about 300 tests, no GPU needed
