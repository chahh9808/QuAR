#!/usr/bin/env bash
# Continual (tab:t3) Ours with EMA alpha=0.1 (instead of deployed 0.05), to MATCH NEO-Continual's
# EMA momentum (alpha=0.1) for an apples-to-apples continual comparison. lambda=0.5 (canonical),
# 10-round x 15-corruption continual stream, 5k subset, bs64, seeds 0/1/2, W3A3 & W6A6.
# Both EMA rates set to 0.1: --quar_local_stream_alpha 0.1 AND --quar_global_target_ema_alpha 0.1.
# Refs (same protocol): Ours-EMA(a=0.05) W3A3 35.35 / W6A6 53.70 ; NEO-Continual(a=0.1) 31.67 / 51.15.
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
ADALOG_W3A3=$QUAR/checkpoints/w3a3_checkpoint_opt.pt
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
OUT=$QUAR/runs/continual_ema_a01_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_continual_ema_a01_dir.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
quant_flags(){ case "$1" in
  w3a3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $ADALOG_W3A3";;
  w6a6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6";;
esac; }
base(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval \
  --batch_size 64 --workers 8 --corruption all --seed $1 --corruption_repeat 10 --continual \
  --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $(quant_flags $2)"; }
# Ours-EMA, lambda=0.5, EMA alpha=0.1 on BOTH local QAC stream and global CLS
ours(){ local bw=$1; echo "--algorithm no_adapt \
  --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 \
  --quar_local_stats_file $QUAR/stats/qac_stats_${bw}_all_layers_seed0.pt --quar_local_stream_ema --quar_local_stream_alpha 0.1 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
  --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 \
  --quar_global_stats_file $QUAR/stats/cls_stats_${bw}_seed0.pt --quar_global_target_ema_alpha 0.1"; }
run(){ local gpu=$1 bw=$2 seed=$3; local tag="ours_a01_${bw}_s${seed}"; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $(base $seed $bw) $(ours $bw) --output "$OUT/$tag" --tag "_$tag" \
    >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
cd "$QUAR"
i=0
for bw in w3a3 w6a6; do for s in 0 1 2; do run $(( i % 6 )) "$bw" "$s" & i=$(( i+1 )); done; done
wait
log "=== Ours-EMA alpha=0.1 continual (10-round Avg) ==="
for bw in w3a3 w6a6; do
  v=$(for s in 0 1 2; do grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/ours_a01_${bw}_s${s}.log" 2>/dev/null|grep -oE "[0-9.]+$"|awk '{x+=$1;k++}END{if(k)print x/k}'; done | awk '{s+=$1;n++}END{if(n)printf "%.2f (%dseed)",s/n,n;else print "NA"}')
  log "  ours_a01 $bw = $v"
done
log "ALL DONE: $OUT  (cf. Ours a=0.05: W3A3 35.35/W6A6 53.70 ; NEO-Cont a=0.1: 31.67/51.15)"
