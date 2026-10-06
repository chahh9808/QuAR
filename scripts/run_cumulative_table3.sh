#!/usr/bin/env bash
# Cumulative-estimator variant of Table 3 (tab:scope): recalibration-scope ablation, Local-only
# Stream-QAC (NO cls_shift), lambda=0.5, seed0, ImageNet-C sev5 15-corr 5k, bs64, episodic, W6A6+W4A4.
# Recipe = the published Local-only flags + --recal_cumulative. 12 scope configs x 2 bitwidths = 24.
# Scope strings verified from code (_qac_select_modules) + logs. EMA refs in comments per cell.
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
ADALOG_W4A4=$QUAR/checkpoints/w4a4_checkpoint_opt.pt
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
OUT=$QUAR/runs/cum_table3_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_cum_table3_dir.txt"
GPUS="${GPUS:-0 1 2 3 4 5}"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
qf(){ case "$1" in w6a6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6";;
                   w4a4) echo "--quant --quant_w_bit 4 --quant_a_bit 4 --adalog_checkpoint $ADALOG_W4A4";; esac; }
base(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 8 \
  --corruption all --seed 0 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $(qf $1)"; }
# Local-only (NO --cls_shift), lambda=0.5, + cumulative
localf(){ local bw=$1 scope=$2; echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope $scope \
  --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 \
  --quar_local_stats_file $QUAR/stats/qac_stats_${bw}_all_layers_seed0.pt \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag --recal_cumulative"; }
runcell(){ local gpu=$1 bw=$2 scope=$3; local tag="${bw}_${scope}"; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $(base $bw) $(localf $bw $scope) --output "$OUT/$tag" --tag "_$tag" \
    >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
worker(){ local gpu=$1; while :; do
  local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"); [ -z "$line" ] && break
  runcell "$gpu" $line; done; }
cd "$QUAR"
SCOPES="block8_9_10_11_attn block8_9_10_11_valid block8_9_10_11_all block0_to_3_all block4_to_7_all \
        block11_all block10_11_all block9_10_11_all block7_to_11_all block6_to_11_all block5_to_11_all all"
Q="$OUT/queue"; : >"$Q"
for bw in w6a6 w4a4; do for sc in $SCOPES; do echo "$bw $sc" >>"$Q"; done; done
log "OUT=$OUT | $(wc -l <"$Q") scope cells (Local-only cumulative) on GPUs [$GPUS]"
for g in $GPUS; do worker $g & done
wait
log "=== Table 3 cumulative (Local-only, mean top-1) -- cf EMA refs ==="
declare -A E6 E4
E6[block8_9_10_11_attn]=50.41; E6[block8_9_10_11_valid]=51.52; E6[block8_9_10_11_all]=52.82
E6[block0_to_3_all]=35.73; E6[block4_to_7_all]=43.95; E6[block11_all]=48.51; E6[block10_11_all]=50.96
E6[block9_10_11_all]=51.91; E6[block7_to_11_all]=52.93; E6[block6_to_11_all]=52.75; E6[block5_to_11_all]=52.45; E6[all]=36.09
E4[block8_9_10_11_attn]=49.75; E4[block8_9_10_11_valid]=50.91; E4[block8_9_10_11_all]=52.33
E4[block0_to_3_all]=34.07; E4[block4_to_7_all]=42.69; E4[block11_all]=48.23; E4[block10_11_all]=50.33
E4[block9_10_11_all]=51.46; E4[block7_to_11_all]=52.46; E4[block6_to_11_all]=51.74; E4[block5_to_11_all]=51.24; E4[all]=32.49
for sc in $SCOPES; do
  v6=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/w6a6_${sc}.log" 2>/dev/null|grep -oE "[0-9.]+$"|awk '{s+=$1;n++} END{if(n)printf "%.2f",s/n; else print "NA"}')
  v4=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/w4a4_${sc}.log" 2>/dev/null|grep -oE "[0-9.]+$"|awk '{s+=$1;n++} END{if(n)printf "%.2f",s/n; else print "NA"}')
  log "  $sc : W6A6 cum=$v6 (EMA ${E6[$sc]}) | W4A4 cum=$v4 (EMA ${E4[$sc]})"
done
log "ALL DONE: $OUT"
