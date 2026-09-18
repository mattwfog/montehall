#!/usr/bin/env bash
# Overnight v1 chain (runs INSIDE the cvbench container, launched detached):
#   J jersey models (synth 20k) -> H multi-clip harvest (extract+court+rim-VLM)
#   -> M merge COCO sources -> T RF-DETR v1 long fine-tune -> E coverage eval.
# Every stage skips when its output marker exists (safe to relaunch); a failed
# harvest clip is logged and skipped, it never kills the chain.
set -u
cd /work/montehall-cv
export PYTHONPATH=/work/montehall-cv
ANTHROPIC_API_KEY=$(cat /root/.anthropic_key)
export ANTHROPIC_API_KEY
COURT=/work/models/basketball/court_segmentation_v4.pth
V0_EMA=/work/models/finetune-v0/checkpoint_best_ema.pth
EVAL_VIDEO=${EVAL_VIDEO:-/work/testdata/07df788c/video.mp4}
EPOCHS=${EPOCHS:-40}
log() { echo "[$(date -u +%FT%TZ)] $*"; }

log "=== stage J: jersey models ==="
if [ -f /work/models/jersey-v1/digits_resnet34.pt ]; then log "J skip (model exists)"; else
  python -m montehall_cv.training.train_jersey \
    --dataset /work/datasets/synthjersey-v0 --out /work/models/jersey-v1 --epochs 8 \
    || log "J FAILED (continuing)"
fi

log "=== stage H: harvest clips ==="
for V in /work/videos/*.mp4; do
  [ -e "$V" ] || { log "H: no clips found"; break; }
  J=$(basename "$V" .mp4)
  D=/work/datasets/harvest-v1/$J
  if [ -f "$D/train/_annotations.coco.json" ]; then log "H skip $J"; continue; fi
  log "H extract $J"
  python -m montehall_cv.pipeline.run_extract_fast --video "$V" --out /work/out --job-id "$J" --weights "$V0_EMA" \
    || { log "H extract FAILED $J"; continue; }
  log "H court $J"
  python -m montehall_cv.pipeline.run_court --video "$V" --out /work/out --job-id "$J" --weights "$COURT" \
    || { log "H court FAILED $J"; continue; }
  log "H rim $J"
  python -m montehall_cv.pipeline.run_rim --video "$V" --out /work/out --job-id "$J" \
    || { log "H rim FAILED $J"; continue; }
  log "H dataset $J"
  python -m montehall_cv.training.build_dataset --video "$V" --job-dir "/work/out/$J" --dataset-dir "$D" --max-frames 1200 \
    || log "H dataset FAILED $J"
done

log "=== stage M: merge ==="
if [ -f /work/datasets/merged-v1/train/_annotations.coco.json ]; then log "M skip"; else
  SRCS="--source /work/datasets/selftrain-v0"
  for D in /work/datasets/harvest-v1/*/; do
    [ -f "${D}train/_annotations.coco.json" ] && SRCS="$SRCS --source ${D%/}"
  done
  # shellcheck disable=SC2086
  python -m montehall_cv.training.merge_coco $SRCS --out /work/datasets/merged-v1 \
    || { log "M FAILED — aborting chain"; exit 1; }
fi

log "=== stage T: RF-DETR v1 fine-tune ($EPOCHS epochs) ==="
if [ -f /work/models/finetune-v1/.done ]; then log "T skip"; else
  if python -m montehall_cv.training.train_detector \
       --dataset-dir /work/datasets/merged-v1 --output-dir /work/models/finetune-v1 --epochs "$EPOCHS"; then
    touch /work/models/finetune-v1/.done
  else
    log "T FAILED — resume with --resume /work/models/finetune-v1/last.ckpt"; exit 1
  fi
fi

log "=== stage E: coverage eval vs pretrained ==="
# Eval BOTH best_total and last: rfdetr's best-checkpoint selection freezes
# at its pre-restart high-water mark after a crash-resume (v1 shipped an
# epoch-1 best_ema), so 'best' alone can silently evaluate a stale model.
if [ -f "$EVAL_VIDEO" ]; then
  for ckpt in checkpoint_best_total.pth last.ckpt; do
    out="/work/models/finetune-v1/eval_coverage_${ckpt%%.*}.json"
    if [ -f "$out" ]; then log "E skip $ckpt"; continue; fi
    [ -f "/work/models/finetune-v1/$ckpt" ] || { log "E skip $ckpt (absent)"; continue; }
    python -m montehall_cv.training.eval_finetune \
      --video "$EVAL_VIDEO" --finetuned "/work/models/finetune-v1/$ckpt" \
      > "$out" \
      || log "E FAILED on $ckpt"
  done
else
  log "E skip (no eval video at $EVAL_VIDEO)"
fi

log "=== overnight v1 chain COMPLETE ==="
