#!/usr/bin/env bash
# Cumulative-estimator variant of Table 6 (tab:sens): lambda (variance-match) sweep + the bs=128 cell.
# Ours = no_adapt + QAC(local, block8_9_10_11_all) + CLS(global), W6A6 seed0, episodic, 15-corr 5k.
# Recipe = published Ours flags + --recal_cumulative; lambda = --quar_local_lambda in {0,0.1,0.3,0.5,0.7,1.0}
# (lambda=0 = mean-only). bs sweep already cumulative-measured (run_cumulative_bs_sweep: bs1 54.31/
# 4 54.35/8 54.49/16 54.34/64 54.37); only bs=128 missing -> added here. EMA refs (tab:sens):
#   lambda 0/0.1/0.3/0.5/0.7/1.0 = 51.28/52.38/53.66/54.07/54.26/53.93 ; bs128 = 53.79.
# (alpha (EMA-rate) row is intentionally NOT re-run: it is meaningless for the cumulative estimator.)
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
QACB3=$QUAR/stats/qac_stats_w6a6_all_layers_seed0.pt; CLS=$QUAR/stats/cls_stats_w6a6_seed0.pt
OUT=$QUAR/runs/cum_table6lam_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_cum_table6lam_dir.txt"
GPUS="${GPUS:-6 7}"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
W6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6"
CLSF="--quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $CLS --quar_global_target_ema_alpha 0.05"
qac(){ echo "--quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda $1 --quar_local_stats_file $QACB3 --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag --recal_cumulative"; }
base(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size $1 --workers 8 \
  --corruption all --seed 0 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $W6"; }
runcell(){ local gpu=$1 kind=$2 val=$3; local tag="${kind}_${val/./p}"; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  local cmd
  if [ "$kind" = lam ]; then cmd="$(base 64) --algorithm no_adapt $(qac $val) $CLSF"
  else                        cmd="$(base $val) --algorithm no_adapt $(qac 0.5) $CLSF"; fi   # kind=bs
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $cmd --output "$OUT/$tag" --tag "_$tag" \
    >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
worker(){ local gpu=$1; while :; do
  local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"); [ -z "$line" ] && break
  runcell "$gpu" $line; done; }
cd "$QUAR"
Q="$OUT/queue"; : >"$Q"
for lam in 0 0.1 0.3 0.5 0.7 1.0; do echo "lam $lam" >>"$Q"; done
echo "bs 128" >>"$Q"
log "OUT=$OUT | $(wc -l <"$Q") cells (lambda sweep + bs128, cumulative) on GPUs [$GPUS]"
for g in $GPUS; do worker $g & done
wait
log "=== Table 6 cumulative (lambda sweep + bs128, mean top-1) -- cf EMA refs ==="
declare -A EL; EL[0]=51.28; EL[0.1]=52.38; EL[0.3]=53.66; EL[0.5]=54.07; EL[0.7]=54.26; EL[1.0]=53.93
for lam in 0 0.1 0.3 0.5 0.7 1.0; do
  v=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/lam_${lam/./p}.log" 2>/dev/null|grep -oE "[0-9.]+$"|awk '{s+=$1;n++} END{if(n)printf "%.2f",s/n; else print "NA"}')
  log "  lambda=$lam : cum=$v (EMA ${EL[$lam]})"
done
vb=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/bs_128.log" 2>/dev/null|grep -oE "[0-9.]+$"|awk '{s+=$1;n++} END{if(n)printf "%.2f",s/n; else print "NA"}')
log "  bs=128 : cum=$vb (EMA 53.79)  [bs 1/4/8/16/64 cumulative already: 54.31/54.35/54.49/54.34/54.37]"
log "ALL DONE: $OUT"
