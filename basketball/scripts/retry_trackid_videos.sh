#!/usr/bin/env bash
# Backoff retry for TrackID3x3 footage that Google Drive rate-limited, then
# rebuild the ReID crop set with whatever landed. Safe alongside the GPU
# chains: network + CPU only. Runs until all files land or attempts cap out;
# only rebuilds crops (and clears the crops .done marker) if downloads improved.
set -u
cd /work/montehall-cv
export PYTHONPATH=/work/montehall-cv
VIDEOS_LIST=${VIDEOS_LIST:-/work/logs/trackid3x3-videos.json}
CROPS_DIR=/work/datasets/trackid3x3-reid
MAX_ATTEMPTS=${MAX_ATTEMPTS:-10}
SLEEP_S=${SLEEP_S:-1200}
log() { echo "[$(date -u +%FT%TZ)] $*"; }

fetch() {
  VIDEOS_LIST="$VIDEOS_LIST" python - <<'PY'
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
        print(f"[FAIL] {entry['path']}: {str(exc)[:120]}", flush=True)
print(f"failed={failed}", flush=True)
sys.exit(min(failed, 120))
PY
}

count_present() { find /work/datasets/trackid3x3/videos -type f \( -name "*.mp4" -o -name "*.MOV" -o -name "*.mov" \) 2>/dev/null | wc -l; }

before=$(count_present)
log "retry loop start: $before/112 present"
attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  log "attempt $attempt"
  fetch
  remaining=$?
  if [ "$remaining" -eq 0 ]; then log "all files present"; break; fi
  log "$remaining still failing — sleeping ${SLEEP_S}s (Drive quota backoff)"
  attempt=$((attempt + 1))
  sleep "$SLEEP_S"
done

after=$(count_present)
log "downloads: $before -> $after of 112"
if [ "$after" -gt "$before" ]; then
  log "rebuilding ReID crops with the enlarged footage set"
  rm -f "$CROPS_DIR/.done"
  if python -m montehall_cv.training.trackid_to_reid \
       --gt-root /work/trackid3x3-repo/ground_truth \
       --videos-root /work/datasets/trackid3x3/videos --out "$CROPS_DIR"; then
    touch "$CROPS_DIR/.done"
    log "crops rebuilt"
  else
    log "crop rebuild FAILED — previous crops remain usable"
    touch "$CROPS_DIR/.done"
  fi
else
  log "no new files landed — crops unchanged"
fi
log "retry loop COMPLETE"
