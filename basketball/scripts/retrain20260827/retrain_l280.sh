#!/bin/bash
# RETRAIN ON THE 280-GAME PERCEPTION CORPUS (2026-08-27).
#
# The estimator was last trained 2026-07-31 (abl-20260730/l-balanced,
# 447 games) when /work/out-extract held ~30 channel-complete real
# games. The 08-01..03 burn landed 282 out-extract token dirs (280 with
# ball+player channels, median 30k tokens / 1,433 ball rows per game)
# and no estimator has trained on them since — the paper's own error
# attribution named real perception channels as the missing input.
#
# Two arms, same recipe family as ladder_20260730.sh, run serially:
#   l280-plain     sim x5, align, out-extract x1  (rich now dominates
#                  the token mass on its own; ~1.5x the 07-31 epoch cost)
#   l280-balanced  the l-balanced recipe verbatim: sim x5, out-extract x5
#                  (~8x the 07-31 epoch cost — runs second)
# Each arm: train -> eval-sim (5 sealed sim holdouts) -> eval-real-fix
# (v2-perception holdout dirs) -> eval-real-v3real (genuine-v3 dirs),
# the earlier post-extraction eval idiom, so numbers compare row-for-row with
# /work/models/abl-20260730/*/eval-real-{fix,v3real}. Sealed real
# holdouts are excluded by id inside the dataloader (holdouts.py).
# Weights + eval JSONs land under /work/models/retrain-20260827/<arm>/
# as produced; log ends RETRAIN_ALL_DONE.
set -u
LOG=$HOME/cv-bench/logs/retrain-l280-20260827.log
exec >>"$LOG" 2>&1
echo "RETRAIN_START $(date -Is)"
docker exec -w /work/montehall-cv cvbench sh -c '
set -u
export PYTHONPATH=/work/montehall-cv
EXC="sim_0007_045 sim_0007_046 sim_0007_047 sim_0007_048 sim_0007_049"
SIMHOLD="/work/sim/sim_0007_045 /work/sim/sim_0007_046 /work/sim/sim_0007_047 /work/sim/sim_0007_048 /work/sim/sim_0007_049"
FIX="/work/out-harvest/cal_fsu_harvest_fix /work/out-harvest/clemson_duke_v3det"
V3NEW="/work/out-harvest/cal_fsu_v3real /work/out-harvest/clemson_duke_v3real"
BASE=/work/models/retrain-20260827
mkdir -p $BASE

echo "=== CONSTRUCTED VALUES (dry-run echo)"
for r in /work/sim /work/align /work/out-extract; do
  echo "root $r: $(ls -d $r/*/tokens/_SUCCESS 2>/dev/null | wc -l) tokenized games"
done
echo "exclude-keys: $EXC"
echo "sim holdouts: $SIMHOLD"
echo "fix dirs: $FIX"
echo "v3real dirs: $V3NEW"
for d in $SIMHOLD $FIX $V3NEW; do [ -f $d/tokens/_SUCCESS ] || echo "MISSING_TOKENS $d"; done

# A CUDA process started within seconds of the previous one exiting can
# OOM at model.to(cuda) on this box (the 07-30 s-perc case; the first
# l280-plain real-fix eval hit it 8s after training exited). The earlier
# eval idiom sleeps between stages; evals additionally retry.
evalstage() { # name outname gamedirs...
  local name="$1" outname="$2"; shift 2
  echo "=== EVAL $name $outname $(date -Is)"
  local i
  for i in 1 2 3 4; do
    sleep 10
    python -m montehall_cv.brain.eval_v0 --weights $BASE/$name/last.pt \
      --game-dirs "$@" --out $BASE/$name/$outname --channels all && return 0
    echo "EVAL_RETRY $name $outname attempt=$i"
  done
  echo "EVAL_FAIL $name $outname"
}

run() {
  name=$1; shift
  echo "=== TRAIN $name $(date -Is)"
  python -m montehall_cv.brain.train_v0 --out $BASE/$name --epochs 12 \
    --exclude-keys $EXC --channels all "$@" \
    || { echo "TRAIN_FAIL $name"; return 1; }
  evalstage $name eval-sim $SIMHOLD
  evalstage $name eval-real-fix $FIX
  evalstage $name eval-real-v3real $V3NEW
  echo "ARM_DONE $name $(date -Is)"
}

run l280-plain --data-roots /work/sim /work/align /work/out-extract \
  --oversample /work/sim=5
run l280-balanced --data-roots /work/sim /work/align /work/out-extract \
  --oversample /work/sim=5 /work/out-extract=5

echo "RETRAIN_ALL_DONE $(date -Is)"
'
echo "RETRAIN_END $(date -Is)"
