#!/usr/bin/env bash
# ReID leg of the overnight v1 run (runs INSIDE cvbench, launched detached,
# in parallel with overnight_v1.sh):
#   1. pull TrackID3x3 videos from Drive (resumable; CC BY 4.0 dataset only,
#      baseline outputs/checkpoints deliberately excluded)
#   2. crop identity dataset from repo ground truth (CPU — overlaps GPU train)
#   3. wait for the detector chain's .done marker, then train OSNet from scratch
set -u
cd /work/montehall-cv
export PYTHONPATH=/work/montehall-cv
VIDEOS_LIST=${VIDEOS_LIST:-/work/logs/trackid3x3-videos.json}
GT_ROOT=/work/trackid3x3-repo/ground_truth
VIDEOS_DIR=/work/datasets/trackid3x3/videos
CROPS_DIR=/work/datasets/trackid3x3-reid
DETECTOR_DONE=/work/models/finetune-v1/.done
MAX_WAIT_HOURS=${MAX_WAIT_HOURS:-14}
log() { echo "[$(date -u +%FT%TZ)] $*"; }

log "=== R1: TrackID3x3 raw footage download (per-file resume) ==="
# Raw game footage only — the Drive tree's other .mp4s are baseline
# visualization renders with boxes burned in (poison for training crops).
if [ ! -f "$VIDEOS_LIST" ]; then log "R1 FATAL: $VIDEOS_LIST missing (Drive tree listing)"; exit 1; fi
if [ -f "$VIDEOS_DIR/.download_done" ]; then log "R1 skip"; else
  if VIDEOS_LIST="$VIDEOS_LIST" python - <<'PY'
import json, os, sys
import gdown

entries = json.load(open(os.environ["VIDEOS_LIST"]))
root = "/work/datasets/trackid3x3"
failed = 0
for entry in entries:
    out = os.path.join(root, entry["path"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        continue
    try:
        gdown.download(url=entry["url"], output=out, quiet=True, resume=True)
        print(f"[dl] {entry['path']}", flush=True)
    except Exception as exc:
        failed += 1
        print(f"[FAIL] {entry['path']}: {exc}", flush=True)
print(f"failed={failed}", flush=True)
sys.exit(1 if failed else 0)
PY
  then
    touch "$VIDEOS_DIR/.download_done"
  else
    log "R1 had failures — continuing with what landed (re-run resumes)"
  fi
fi

log "=== R2: identity crop extraction ==="
if [ -f "$CROPS_DIR/.done" ]; then log "R2 skip"; else
  if python -m montehall_cv.training.trackid_to_reid \
       --gt-root "$GT_ROOT" --videos-root "$VIDEOS_DIR" --out "$CROPS_DIR"; then
    touch "$CROPS_DIR/.done"
  else
    log "R2 FAILED — aborting ReID leg"; exit 1
  fi
fi

log "=== R3: wait for detector chain (marker: $DETECTOR_DONE) ==="
waited=0
while [ ! -f "$DETECTOR_DONE" ]; do
  if [ "$waited" -ge $((MAX_WAIT_HOURS * 3600)) ]; then
    log "R3 detector still not done after ${MAX_WAIT_HOURS}h — training ReID anyway (GPU shared)"
    break
  fi
  sleep 120; waited=$((waited + 120))
done

log "=== R4: OSNet from-scratch training ==="
if [ -f /work/models/reid-v1/summary.json ]; then log "R4 skip"; else
  python -m montehall_cv.training.train_reid \
    --dataset "$CROPS_DIR" --out /work/models/reid-v1 --epochs 60 \
    || { log "R4 FAILED"; exit 1; }
fi

log "=== ReID leg COMPLETE ==="
