#!/usr/bin/env bash
# Table tab:w4a8, the asymmetric deployment bit-width (4-bit weights, 8-bit activations).
#
# Protocol as Table 1: ImageNet-C severity 5, all 15 corruptions, RobustBench 5k, batch 64,
# episodic, 3 seeds. W4A8 uses AdaLog rather than PTQ4ViT (PTQ4ViT scores 76.62 clean at W4A8
# against AdaLog's 85.22). The per-method flag builders are those of run_full50k_table1.sh, so
# the W4A8 column is measured the same way as the rest of Table 1.
#
# Usage:  bash scripts/run_w4a8_table.sh            # all methods, seeds 0-2
#         METHODS="quar" SEEDS="0" bash scripts/run_w4a8_table.sh
# Resumable: a cell whose log already holds 15 "Top-1 Accuracy" lines is skipped.
set -uo pipefail
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python3}"
DATA="${DATA:?set DATA to the ImageNet root (…/Data/CLS-LOC)}"
DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt
CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
FOA_STATS=$QUAR/foa_stats
FOZO_STATS=$QUAR/stats/train_stats_2.pt
OUT="${OUT:-$QUAR/runs/w4a8_table_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"
NODE1="$(cat /sys/devices/system/node/node1/cpulist 2>/dev/null || echo 0-71)"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; read -ra GA <<< "$GPUS"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }

BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --corruption all \
--imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP --batch_size 64 --workers 4"

# W4A8 is AdaLog, not PTQ4ViT.
QF="--quant --quant_w_bit 4 --quant_a_bit 8 --adalog_checkpoint $QUAR/checkpoints/w4a8_checkpoint_opt.pt"

# QuAR source statistics collected on the AdaLog-quantized W4A8 model.
QAC_STATS="${QAC_STATS:-$QUAR/stats/qac_stats_w4a8_adalog_all_layers_seed0.pt}"
CLS_STATS="${CLS_STATS:-$QUAR/stats/cls_stats_w4a8_adalog_seed0.pt}"

mflags(){ case "$1" in
  noadapt) echo "--algorithm no_adapt";;
  neo)     echo "--algorithm no_adapt --neo_enable";;
  t3a)     echo "--algorithm t3a --t3a_filter_k 20 --arch vit_base_patch16_224";;
  foa)     echo "--algorithm foa --popsize 1 --foa_stats_dir $FOA_STATS";;
  zoa)     echo "--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 \
--official_zoa_lr_eps 0.0002 --zoa_layer_balance none --official_zoa_stats_file $QUAR/zoa_stats/w4a8.pt";;
  fozo)    echo "--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 --fozo_stats_file $FOZO_STATS";;
  quar)    echo "--algorithm quar --quar_local_stats_file $QAC_STATS --quar_global_stats_file $CLS_STATS";;
  *) echo "BADMETHOD"; return 1;;
esac; }

# No Adapt carries no seed spread in the paper and has no stochastic component, so it runs once.
METHODS="${METHODS:-noadapt t3a foa zoa neo fozo quar}"
SEEDS="${SEEDS:-0 1 2}"
is_done(){ [ -f "$1" ] && [ "$(grep -c 'Top-1 Accuracy' "$1" 2>/dev/null)" -eq 15 ]; }

cd "$QUAR"
Q="$OUT/queue"; : >"$Q"
for m in $METHODS; do
  for s in $SEEDS; do
    [ "$m" = noadapt ] && [ "$s" != 0 ] && continue
    echo "$m $s" >>"$Q"
  done
done
log "queued $(wc -l < "$Q") cells -> $OUT"

i=0
while read -r m s; do
  tag="${m}_s${s}"; g=${GA[$((i % ${#GA[@]}))]}; i=$((i+1))
  if is_done "$OUT/$tag.log"; then log "[$tag] SKIP (already 15 corruptions)"; continue; fi
  mkdir -p "$OUT/$tag"
  ( log "[$tag g$g] start"
    CUDA_VISIBLE_DEVICES=$g taskset -c "$NODE1" $PY -u "$QUAR/main.py" $BASE $QF \
      $(mflags "$m") --seed "$s" --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 \
      && log "[$tag] done" || log "[$tag] FAIL exit=$?" ) &
  while [ "$(jobs -rp | wc -l)" -ge "${#GA[@]}" ]; do sleep 15; done
done < "$Q"
wait

log "---- means over 15 corruptions ----"
{ printf "%-10s %-6s %s\n" method seed mean
  for f in "$OUT"/*.log; do
    b=$(basename "$f" .log); [ "$b" = driver ] && continue
    n=$(grep -c 'Top-1 Accuracy' "$f" 2>/dev/null)
    [ "$n" -eq 15 ] 2>/dev/null || { printf "%-10s %-6s incomplete (%s/15)\n" "${b%_s*}" "${b##*_s}" "$n"; continue; }
    mean=$(grep -oE 'Top-1 Accuracy: [0-9.]+' "$f" | awk '{s+=$3; n++} END{printf "%.2f", s/n}')
    printf "%-10s %-6s %s\n" "${b%_s*}" "${b##*_s}" "$mean"
  done; } | tee -a "$OUT/summary.txt"
log "done -> $OUT/summary.txt"

# VALIDATION. Two cells of this script were run against the paper before it was committed:
# No Adapt reproduced 54.12 exactly, and QuAR seed 0 gave 60.19 against a published 60.11 +- 0.07.
# The paper's W4A8 column, for comparison against summary.txt:
#   No Adapt 54.12   T3A 53.74+-0.01   FOA 57.55+-0.14   ZOA 58.10+-0.01
#   FOZO 57.88+-0.14   NEO 58.01+-0.02   QuAR 60.11+-0.07
