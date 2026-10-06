#!/usr/bin/env bash
# Small-batch behavior of PLAIN cumulative (--recal_cumulative, NO drift/periodic reset).
# Does cumulative keep the deployed EMA's small-batch robustness? iid 5k 15-corr episodic, W6A6 seed0.
# Compare to EMA tab:sens batch row: bs1 52.40 / bs4 53.51 / bs8 53.87 / bs16 54.12 / bs64 54.07.
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
QAC=$QUAR/stats/qac_stats_w6a6_all_layers_seed0.pt; CLS=$QUAR/stats/cls_stats_w6a6_seed0.pt
W6A6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"
OUT=$QUAR/runs/cumbs_w6a6_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_cumbs_dir.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
# PLAIN cumulative: --recal_cumulative only (no --recal_drift_reset / --recal_reset_period)
ours(){ echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
  --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QAC \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
  --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
  --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $CLS --quar_global_target_ema_alpha 0.05 \
  --recal_cumulative"; }
runcmd(){ local gpu=$1 bs=$2; local tag="cum_bs${bs}"; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py \
    --data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size $bs --workers 8 \
    --corruption all --seed 0 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
    $W6A6 $(ours) --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
cd "$QUAR"
log "OUT=$OUT | plain cumulative (no reset) bs sweep {1,4,8,16,64}, iid 5k 15-corr episodic"
Q=$OUT/queue; : >"$Q"; for bs in 1 4 8 16 64; do echo "$bs" >>"$Q"; done
worker(){ local gpu=$1; while :; do
  local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"); [ -z "$line" ] && break
  runcmd "$gpu" $line; done; }
for g in 0 1 2 3 6; do worker $g & done
wait
log "=== plain cumulative bs sweep (15-corr mean top-1) ==="
for bs in 1 4 8 16 64; do
  tag="cum_bs${bs}"; [ -f "$OUT/$tag.log" ] || continue
  v=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/$tag.log" | grep -oE "[0-9.]+$" | awk '{s+=$1;n++} END{if(n>0)printf "%.2f (n=%d)",s/n,n; else print "NA"}')
  log "cum bs=$bs = $v"
done
log "ALL DONE: $OUT  (EMA tab:sens refs: bs1 52.40 / bs4 53.51 / bs8 53.87 / bs16 54.12 / bs64 54.07; cum bs64 ~53.6)"
