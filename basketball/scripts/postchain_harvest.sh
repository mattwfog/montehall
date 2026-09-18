#!/usr/bin/env bash
# Post-chain harvest for clips that missed the main overnight loop (e.g. the
# clip-pull race). Waits for the main chain's COMPLETE line, then harvests
# each named job id with the freshly-trained v1 detector weights.
# Usage: postchain_harvest.sh <job_id> [job_id...]
set -u
cd /work/montehall-cv
export PYTHONPATH=/work/montehall-cv
ANTHROPIC_API_KEY=$(cat /root/.anthropic_key)
export ANTHROPIC_API_KEY
COURT=/work/models/basketball/court_segmentation_v4.pth
V1_EMA=/work/models/finetune-v1/checkpoint_best_ema.pth
MAIN_LOG=/work/logs/overnight-v1.log
MAX_WAIT_HOURS=${MAX_WAIT_HOURS:-14}
log() { echo "[$(date -u +%FT%TZ)] $*"; }

log "waiting for main chain COMPLETE"
waited=0
until /usr/bin/grep -q "overnight v1 chain COMPLETE" "$MAIN_LOG" 2>/dev/null; do
  if [ "$waited" -ge $((MAX_WAIT_HOURS * 3600)) ]; then
    log "main chain not COMPLETE after ${MAX_WAIT_HOURS}h — aborting"; exit 1
  fi
  sleep 120; waited=$((waited + 120))
done

WEIGHTS="$V1_EMA"
[ -f "$WEIGHTS" ] || { log "v1 weights missing, falling back to v0"; WEIGHTS=/work/models/finetune-v0/checkpoint_best_ema.pth; }

for J in "$@"; do
  V=/work/videos/$J.mp4
  D=/work/datasets/harvest-v1/$J
  [ -f "$V" ] || { log "no video for $J"; continue; }
  if [ -f "$D/train/_annotations.coco.json" ]; then log "skip $J"; continue; fi
  log "harvest $J (weights: $WEIGHTS)"
  python -m montehall_cv.pipeline.run_extract_fast --video "$V" --out /work/out --job-id "$J" --weights "$WEIGHTS" \
    || { log "extract FAILED $J"; continue; }
  python -m montehall_cv.pipeline.run_court --video "$V" --out /work/out --job-id "$J" --weights "$COURT" \
    || { log "court FAILED $J"; continue; }
  python -m montehall_cv.pipeline.run_rim --video "$V" --out /work/out --job-id "$J" \
    || { log "rim FAILED $J"; continue; }
  python -m montehall_cv.training.build_dataset --video "$V" --job-dir "/work/out/$J" --dataset-dir "$D" --max-frames 1200 \
    || log "dataset FAILED $J"
done
log "postchain harvest COMPLETE"
