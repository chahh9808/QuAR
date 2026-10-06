#!/usr/bin/env bash
# TENT / EATA / SAR at seeds 1 and 2, the other two thirds of Table 7's accuracy column.
#
# Seed 0 comes from run_backprop_tta.sh. tab:backprop_acc reports a 3-seed mean, so all three
# seeds are needed: SAR 49.77+-0.11, TENT 50.05+-0.13, EATA 52.79+-0.53. Reading seed 0 alone
# gives 49.65 / 50.18 / 52.28 and looks like a mismatch against the paper.
#
# Setting is identical to run_backprop_tta.sh, which is the point:
#   W6A6 ViT-B | ImageNet-C all-15 | sev5 | RobustBench 5k subset
#   bs=64 | episodic | official SAR-repo hyperparameters
#
#
# Usage
#   DATA=... DATA_C=... bash scripts/run_backprop_seeds12.sh [GPU_BASE]
#   SEEDS="1 2" is the default; SEEDS="0 1 2" re-runs seed 0 here too.
set -uo pipefail

GPU_BASE=${1:-0}
SEEDS="${SEEDS:-1 2}"

PYTHON="${PYTHON:-python3}"
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root (…/Data/CLS-LOC)}"
DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
OUTBASE=$QUAR/runs/backprop_seeds12_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTBASE"
echo "$OUTBASE" >"$QUAR/runs/last_backprop_seeds12_dir.txt"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
W6A6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6"

BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval \
  --batch_size 64 --workers 4 --corruption all \
  --imagenet_5k_indices_file $QUAR/robustbench/data/imagenet_test_image_ids.txt \
  --imagenet_class_map_file $QUAR/robustbench/data/imagenet_class_to_id_map.json"

run() {
  local gpu=$1 alg=$2 seed=$3
  local tag="${alg}_s${seed}"
  mkdir -p "$OUTBASE/$tag"
  log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "$QUAR/main.py" $BASE $W6A6 \
    --algorithm "$alg" --seed "$seed" \
    --output "$OUTBASE/$tag" --tag "_$tag" >"$OUTBASE/$tag.log" 2>&1 \
    && log "[$tag] done" || log "[$tag] FAIL $?"
}

cd "$QUAR"
log "OUT=$OUTBASE | TENT/EATA/SAR x seeds {$SEEDS} | W6A6 all-15 5k episodic"

gpu=$GPU_BASE
for s in $SEEDS; do
  for alg in tent eata sar; do
    ( run "$gpu" "$alg" "$s" ) &
    gpu=$((gpu + 1))
  done
done
wait
log "ALL DONE: $OUTBASE"

echo "=== RESULTS (method seed : mean top-1) ==="
for alg in tent eata sar; do
  for s in $SEEDS; do
    a=$(grep -oE "mean acc of corruption: [0-9.]+" "$OUTBASE/${alg}_s${s}.log" 2>/dev/null \
        | tail -1 | grep -oE "[0-9.]+$")
    printf "%-6s s%s = %s\n" "$alg" "$s" "${a:-NA}"
  done
done
echo "(seed 0, from run_backprop_tta.sh: tent 50.18, eata 52.28, sar 49.65)"
echo "(published 3-seed means: TENT 50.05, EATA 52.79, SAR 49.77)"
