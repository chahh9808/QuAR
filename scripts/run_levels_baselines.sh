#!/usr/bin/env bash
# The four baselines missing from the per-channel code-resolution table (tab:levels):
# T3A, FOA, FOZO, ZOA at W6A6. The existing mismatch16cum run only carried
# No Adapt / Final-only / NEO / Local-only / Ours, and three of those are identical
# because they act at the head, downstream of the 16 measured interfaces. FOA, FOZO
# and ZOA all write upstream of blocks 8-11 (input prompts, and blocks 1-8 LayerNorm
# affine for ZOA), so they can move these activations and must be measured rather
# than assumed. T3A replaces only the classifier, so it is expected to be exactly 0.
#
# Config copied verbatim from runs/mismatch16cum_20260814_115630 (the run tab:levels
# uses): W6A6 PTQ4ViT, ImageNet-C severity 5, all 15 corruptions, RobustBench 5k,
# batch 64, workers 3, seed 0, episodic, mismatch diagnostic over
# block8_9_10_11_all including fc2 (16 modules, 12,288 channels).
# Baseline flags copied verbatim from run_w1_codedist_full.sh.
set -uo pipefail
PY="${PY:-python3}"; F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$F/robustbench/data/imagenet_test_image_ids.txt; CMAP=$F/robustbench/data/imagenet_class_to_id_map.json
PTQ=$F/checkpoints/ptq4vit_w6a6_vitb.pt
OUT=$F/runs/levels_baselines_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
echo "$OUT" > "$F/runs/last_levels_baselines.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --corruption all \
--batch_size 64 --workers 3 --seed 0 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ \
--quar_mismatch_diag --quar_mismatch_diag_include_fc2 --quar_mismatch_diag_scope block8_9_10_11_all"
mf(){ case "$1" in
  T3A)  echo "--algorithm t3a --t3a_filter_k 20 --arch vit_base_patch16_224";;
  FOA)  echo "--algorithm foa --popsize 1 --foa_stats_dir $F/foa_stats";;
  FOZO) echo "--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 --fozo_stats_file $F/stats/train_stats_2.pt";;
  ZOA)  echo "--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 \
--zoa_layer_balance none --official_zoa_stats_file $F/zoa_stats/w6a6.pt";;
esac; }
cd "$F"; i=0
for m in T3A FOA FOZO ZOA; do
  g=$((i % 8)); i=$((i+1)); mkdir -p "$OUT/$m"
  ( log "[$m g$g] start"
    CUDA_VISIBLE_DEVICES=$g $PY -u "$F/main.py" $BASE $(mf "$m") \
      --output "$OUT/$m" --tag "_$m" >"$OUT/$m.log" 2>&1 \
      && log "[$m] done" || log "[$m] FAIL exit=$?" ) &
done
wait; log "LEVELS_BASELINES_DONE $OUT"
for m in T3A FOA FOZO ZOA; do
  f=$(find "$OUT/$m" -name "qac_mismatch_stats.pt" 2>/dev/null | head -1)
  echo "$m  stats=${f:-MISSING}"
done
