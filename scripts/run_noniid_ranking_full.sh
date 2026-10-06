#!/usr/bin/env bash
# Non-IID RANKING table: full ImageNet-C (15 corruptions, FULL 50k/corruption), faithful SAR online
# imbalanced label shift, W6A6 ViT-B seed0 episodic, bs64, T=100000.
# ir in {1 (iid), 1000 (~50% per-batch concentration, moderate), 500000 (~pure class order, extreme)}.
# Methods: No Adapt (ir-invariant -> ir=1 only), NEO, ZOA, QuAR (EMA alpha=0.001), QuAR (cumulative).
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
QAC=$QUAR/stats/qac_stats_w6a6_all_layers_seed0.pt; CLS=$QUAR/stats/cls_stats_w6a6_seed0.pt
W6A6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"
OUT=$QUAR/runs/noniid_rank_w6a6_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_noniid_rank_dir.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
bfull="--data $DATA --data_corruption $DATA_C --level 5 --batch_size 64 --workers 8 --corruption all --seed 0"
ours(){ local a=$1; echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all \
  --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QAC \
  --quar_local_stream_ema --quar_local_stream_alpha $a --quar_local_stream_nwarm 0 --quar_local_stream_lag \
  --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
  --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $CLS --quar_global_target_ema_alpha $a"; }
NOADAPT="--algorithm no_adapt"
NEO="--algorithm no_adapt --neo_enable"
ZOA="--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 \
  --zoa_layer_balance none --official_zoa_stats_file $QUAR/zoa_stats/w6a6.pt"
declare -A CFG; CFG[noadapt]="$NOADAPT"; CFG[neo]="$NEO"; CFG[zoa]="$ZOA"
CFG[lgar_a001]="$(ours 0.001)"; CFG[lgar_cum]="$(ours 0.05) --recal_cumulative"
sar(){ echo "--stream_order sar --sar_imbalance_ratio $1 --sar_T 100000 --noniid_seed 2023"; }
runcmd(){ local gpu=$1 m=$2 ir=$3; local tag="${m}_ir${ir}"; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $bfull $W6A6 ${CFG[$m]} $(sar $ir) \
    --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
cd "$QUAR"
log "OUT=$OUT | non-iid RANKING: 15-corr full-50k bs64 T=100k, ir{1,1000,500000} x {noadapt,neo,zoa,lgar_a001,lgar_cum}"
Q=$OUT/queue; : >"$Q"
echo "noadapt 1" >>"$Q"   # No Adapt: ir-invariant, one run
for ir in 1 1000 500000; do for m in neo zoa lgar_a001 lgar_cum; do echo "$m $ir" >>"$Q"; done; done
worker(){ local gpu=$1; while :; do
  local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"); [ -z "$line" ] && break
  runcmd "$gpu" $line; done; }
for g in 0 1 2 3; do worker $g & done
wait
log "=== non-iid RANKING (15-corr mean top-1) ==="
for ir in 1 1000 500000; do for m in noadapt neo zoa lgar_a001 lgar_cum; do
  tag="${m}_ir${ir}"; [ -f "$OUT/$tag.log" ] || continue
  v=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/$tag.log" | grep -oE "[0-9.]+$" | awk '{s+=$1;n++} END{if(n>0)printf "%.2f (n=%d)",s/n,n; else print "NA"}')
  log "$tag = $v"
done; done
log "ALL DONE: $OUT"
