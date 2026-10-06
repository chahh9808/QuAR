#!/usr/bin/env bash
# Clean-source / no-shift sanity check: does QuAR hurt accuracy when there is NO shift?
# Evaluate No Adapt vs Ours (deployed config) on CLEAN ImageNet val (--corruption original), 5k subset,
# seed 0, bs64, for all 4 quantized bit-widths. Under no shift the test-stream moments should match the
# frozen source PTQ-calibration moments, so the recalibration should be ~identity and Ours ~= No Adapt.
# Identical Ours config to run_alpha05_precision.sh (QAC block8_9_10_11_all, lambda=0.5, + global CLS).
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
PTQ_W8A8=$QUAR/checkpoints/ptq4vit_w8a8_vitb.pt
ADALOG_W4A4=$QUAR/checkpoints/w4a4_checkpoint_opt.pt; ADALOG_W3A3=$QUAR/checkpoints/w3a3_checkpoint_opt.pt
OUT=$QUAR/runs/clean_noshift_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_clean_noshift_dir.txt"
A=0.5
qf(){ case "$1" in
  w8a8) echo "--quant --quant_w_bit 8 --quant_a_bit 8 --ptq_checkpoint $PTQ_W8A8";;
  w6a6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6";;
  w4a4) echo "--quant --quant_w_bit 4 --quant_a_bit 4 --adalog_checkpoint $ADALOG_W4A4";;
  w3a3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $ADALOG_W3A3";; esac; }
QAC(){ echo "--quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda $A --quar_local_stats_file $QUAR/stats/qac_stats_$1_all_layers_seed0.pt --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag"; }
CLS(){ echo "--quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $QUAR/stats/cls_stats_$1_seed0.pt --quar_global_target_ema_alpha 0.05"; }
mf(){ local v=$1 p=$2; case "$v" in
  NOADAPT) echo "--algorithm no_adapt";;
  OURS)    echo "--algorithm no_adapt $(QAC $p) $(CLS $p)";; esac; }
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
run(){ local gpu=$1 p=$2 v=$3; local t="${p}_${v}"; mkdir -p "$OUT/$t"; log "[$t g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py --data $DATA --data_corruption $DATA_C --level 5 \
    --imagenet_5k_eval --batch_size 64 --workers 8 --corruption original --seed 0 \
    --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $(qf $p) $(mf $v $p) \
    --output "$OUT/$t" --tag "_$t" >"$OUT/$t.log" 2>&1 && log "[$t] done" || log "[$t] FAIL exit=$?"; }
cd "$QUAR"; i=0
for p in w8a8 w6a6 w4a4 w3a3; do for v in NOADAPT OURS; do run $(( i % 8 )) "$p" "$v" & i=$(( i+1 )); done; done
wait
log "=== CLEAN (no-shift) top-1, 5k subset, seed 0 ==="
for p in w8a8 w6a6 w4a4 w3a3; do
  na=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/${p}_NOADAPT.log" | grep -oE "[0-9.]+$" | tail -1)
  ou=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/${p}_OURS.log"    | grep -oE "[0-9.]+$" | tail -1)
  d=$(awk -v a="$na" -v b="$ou" 'BEGIN{if(a!=""&&b!="")printf "%+.4f", b-a; else print "NA"}')
  log "$p  NoAdapt=$na  Ours=$ou  (Ours-NoAdapt=$d)"
done
log "ALL DONE: $OUT"
