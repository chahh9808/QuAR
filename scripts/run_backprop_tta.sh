#!/usr/bin/env bash
# Backprop-based TTA baselines (TENT, EATA, SAR) on W6A6 ViT-B.
#
# Setting (matches the main-table protocol, run_table1_w6a6.sh):
#   W6A6 ViT-B | ImageNet-C all-15 | sev5 | RobustBench 5k subset | seed 0
#   bs=64 | episodic (per-corruption reset) | no ECE special-casing
#
# Official hyperparameters (external_repos/SAR/main.py):
#   ViT lr  = (0.001/64)*bs = 0.001 at bs=64 (computed inside build functions)
#   TENT    : SGD(lr, momentum=0.9), steps=1, episodic=False
#   EATA    : Fisher over 2000 clean samples; fisher_alpha=2000,
#             e_margin=log(1000)*0.40, d_margin=0.05; SGD(lr, momentum=0.9)
#   SAR     : SAM(SGD, lr, momentum=0.9, rho=0.05), margin_e0=log(1000)*0.40
#   : SGD(all params, lr=2.5e-4, momentum=0.9, wd=5e-4, nesterov);
#             LayerNorm mean+var L1-match (BN->LN modification, documented)
#
# GPU layout (one method per GPU): G0=TENT G1=EATA G2=SAR
# Usage: bash run_backprop_tta.sh [GPU_BASE]
set -eo pipefail

GPU_BASE=${1:-0}
G0=$((GPU_BASE+0)); G1=$((GPU_BASE+1)); G2=$((GPU_BASE+2))

PYTHON="${PYTHON:-python3}"
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root (…/Data/CLS-LOC)}"
DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
OUTBASE=$QUAR/runs/backprop_tta_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTBASE"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
W6A6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6"

BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval \
  --batch_size 64 --workers 8 --corruption all --seed 0 \
  --imagenet_5k_indices_file $QUAR/robustbench/data/imagenet_test_image_ids.txt \
  --imagenet_class_map_file $QUAR/robustbench/data/imagenet_class_to_id_map.json"

run_method() {
  local gpu=$1 tag=$2; shift 2
  mkdir -p "$OUTBASE/$tag"
  log "[$tag GPU=$gpu] starting"
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u main.py $BASE $W6A6 \
    --output "$OUTBASE/$tag" --tag "_${tag}_w6a6" \
    "$@" \
    > "$OUTBASE/$tag.log" 2>&1 \
    && log "[$tag] done" \
    || log "[$tag] FAILED (exit $?)"
}

run_TENT()   { cd "$QUAR"; run_method $G0 TENT   --algorithm tent; }
run_EATA()   { cd "$QUAR"; run_method $G1 EATA   --algorithm eata; }
run_SAR()    { cd "$QUAR"; run_method $G2 SAR    --algorithm sar; }

cd "$QUAR"
log "Output: $OUTBASE"
log "GPUs: G0=$G0(TENT) G1=$G1(EATA) G2=$G2(SAR)"
log "W6A6 | full-15 | bs=64 | seed 0 | episodic | official hyperparams"

run_TENT & run_EATA & run_SAR &
wait

log "All backprop-TTA runs complete. Output: $OUTBASE"
echo "=== mean top-1 per method ==="
for tag in TENT EATA SAR; do
  acc=$(grep -oE "mean acc of corruption: [0-9.]+" "$OUTBASE/$tag.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
  printf "%-8s mean_top1=%s\n" "$tag" "${acc:-NA}"
done
echo "$OUTBASE" > "$QUAR/runs/last_backprop_tta_dir.txt"
