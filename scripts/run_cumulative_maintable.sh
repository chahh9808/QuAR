#!/usr/bin/env bash
# Cumulative-estimator verification of the MAIN episodic table (tab:main), at the CANONICAL lambda=0.5
# (--quar_local_lambda 0.5; user-confirmed paper value -- NOT the 1.0 in run_seed_sweep_*). Question: does
# QuAR keeps a clear gain over baselines when the EMA target-moment estimator is swapped for the
# forgetting-free cumulative (alpha_n=1/(bc+1))? Recipe = OURS flags + --recal_cumulative (verified to
# reproduce 54.37 at W6A6). Also runs EMA(lambda=0.5) seed0 per bw as a matched within-batch baseline
# + a matched EMA(lambda=0.5) seed0 run per bit-width as a within-batch reference
# (EMA 3-seed reads W3A3 35.37 / W4A4 53.44 / W6A6 54.17 / W8A8 60.51).
# The cumulative rows are what tab:main publishes: W3A3 35.98 / W4A4 53.58 / W6A6 54.35 / W8A8 60.44.
# Episodic, 5k subset, bs64, seeds 0/1/2 (cumulative); seed0 only (EMA spot-check). GPUs 6,7.
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
PTQ_W8A8=$QUAR/checkpoints/ptq4vit_w8a8_vitb.pt
ADALOG_W4A4=$QUAR/checkpoints/w4a4_checkpoint_opt.pt; ADALOG_W3A3=$QUAR/checkpoints/w3a3_checkpoint_opt.pt
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt
CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
OUT=$QUAR/runs/cum_maintable_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT" > "$QUAR/runs/last_cum_maintable_dir.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
quant_flags(){ case "$1" in
  w3a3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $ADALOG_W3A3";;
  w4a4) echo "--quant --quant_w_bit 4 --quant_a_bit 4 --adalog_checkpoint $ADALOG_W4A4";;
  w6a6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6";;
  w8a8) echo "--quant --quant_w_bit 8 --quant_a_bit 8 --ptq_checkpoint $PTQ_W8A8";;
esac; }
base(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 8 \
  --corruption all --seed $1 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $(quant_flags $2)"; }
# OURS at canonical lambda=0.5; variant=cum adds --recal_cumulative, variant=ema keeps fixed-rate EMA.
ours(){ local bw=$1 variant=$2; local extra=""; [ "$variant" = cum ] && extra="--recal_cumulative"
  echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
    --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QUAR/stats/qac_stats_${bw}_all_layers_seed0.pt \
    --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
    --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
    --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $QUAR/stats/cls_stats_${bw}_seed0.pt \
    --quar_global_target_ema_alpha 0.05 $extra"; }
runcmd(){ local gpu=$1 bw=$2 variant=$3 seed=$4; local tag="${bw}_${variant}_s${seed}"; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $(base $seed $bw) $(ours $bw $variant) \
    --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
cd "$QUAR"
Q=$OUT/queue; : >"$Q"
for bw in w3a3 w4a4 w6a6 w8a8; do
  for s in 0 1 2; do echo "$bw cum $s" >>"$Q"; done   # cumulative: 3 seeds
  echo "$bw ema 0" >>"$Q"                              # EMA: seed0 spot-check/baseline
done
log "OUT=$OUT | $(wc -l <"$Q") runs (12 cum + 4 ema) on GPUs 6,7, lambda=0.5"
worker(){ local gpu=$1; while :; do
  local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"); [ -z "$line" ] && break
  runcmd "$gpu" $line; done; }
for g in 6 7; do worker $g & done
wait
log "=== cumulative vs EMA main table (mean top-1, lambda=0.5) ==="
for bw in w3a3 w4a4 w6a6 w8a8; do
  cumv=$(for s in 0 1 2; do grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/${bw}_cum_s${s}.log" 2>/dev/null | grep -oE "[0-9.]+$" | awk '{x+=$1;k++} END{if(k)print x/k}'; done | awk '{s+=$1;n++} END{if(n)printf "%.2f(%ds)",s/n,n; else print "NA"}')
  emav=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/${bw}_ema_s0.log" 2>/dev/null | grep -oE "[0-9.]+$" | awk '{s+=$1;n++} END{if(n)printf "%.2f",s/n; else print "NA"}')
  log "$bw: cumulative(3seed)=$cumv | EMA(seed0)=$emav"
done
log "ALL DONE: $OUT"
log "published tab:main (cumulative, 3-seed): W3A3 35.98 / W4A4 53.58 / W6A6 54.35 / W8A8 60.44"
log "EMA reference (not the published row):   W3A3 35.37 / W4A4 53.44 / W6A6 54.17 / W8A8 60.51"
