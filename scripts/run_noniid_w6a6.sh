#!/usr/bin/env bash
# Non-IID (wild) stress test, W6A6 ViT-B, ImageNet-C 15-corr 5k seed0 episodic.
# Three test-stream orderings (labels used ONLY for ordering, never for adaptation):
#   iid        = default DataLoader shuffle (matches main tables)
#   imbalanced = Dirichlet(delta=0.1) online label shift (SAR/NOTE-style)
#   sorted     = class_sorted (extreme non-IID: GT-class-sorted stream)
# Methods: No Adapt (order-invariant -> iid only), T3A, NEO, ZOA, QuAR.
# Goal: does QuAR's source-anchored recalibration stay robust where optimizer/
# support-set TTA degrade under non-IID streams? Compare each method's drop vs iid.
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
QAC=$QUAR/stats/qac_stats_w6a6_all_layers_seed0.pt; CLS=$QUAR/stats/cls_stats_w6a6_seed0.pt
W6A6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"
OUT=$QUAR/runs/noniid_w6a6_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_noniid_dir.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
basef(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 8 \
  --corruption all --seed 0 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP"; }
# canonical W6A6 method configs (accuracy): QuAR reproduces 54.07; ZOA = run_table1_w6a6.sh
OURS="--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
  --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QAC \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
  --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
  --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $CLS --quar_global_target_ema_alpha 0.05"
NEO="--algorithm no_adapt --neo_enable"
NOADAPT="--algorithm no_adapt"
T3A="--algorithm t3a --t3a_filter_k 20 --arch vit_base_patch16_224"
ZOA="--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 \
  --zoa_layer_balance none --official_zoa_stats_file $QUAR/zoa_stats/w6a6.pt"
declare -A CFG; CFG[ours]="$OURS"; CFG[neo]="$NEO"; CFG[t3a]="$T3A"; CFG[zoa]="$ZOA"; CFG[noadapt]="$NOADAPT"
# stream-order flags
ord_flags(){ case "$1" in
  iid)        echo "--stream_order default" ;;
  imbalanced) echo "--stream_order imbalanced --noniid_dirichlet_delta 0.1 --noniid_seed 0" ;;
  sorted)     echo "--stream_order class_sorted" ;;
esac; }
run(){ local gpu=$1 method=$2 ord=$3; local tag="${method}_${ord}"; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $(basef) $W6A6 ${CFG[$method]} $(ord_flags $ord) \
    --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
cd "$QUAR"
log "OUT=$OUT | non-iid {iid,imbalanced,sorted} x {ours,neo,zoa,t3a} + noadapt(iid). W6A6 15-corr 5k seed0"
# 13 cells over GPUs 0-3 via a simple flock work queue
Q=$OUT/queue; : >"$Q"
for ord in iid imbalanced sorted; do for m in ours neo zoa t3a; do echo "$m $ord" >>"$Q"; done; done
echo "noadapt iid" >>"$Q"
worker(){ local gpu=$1; while :; do
  local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"); [ -z "$line" ] && break
  run "$gpu" $line; done; }
for g in 0 1 2 3; do worker $g & done
wait
log "=== non-iid results (15-corruption mean top-1) ==="
for ord in iid imbalanced sorted; do for m in noadapt ours neo zoa t3a; do
  tag="${m}_${ord}"; [ -f "$OUT/$tag.log" ] || continue
  v=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/$tag.log" | grep -oE "[0-9.]+$" | awk '{s+=$1;n++} END{if(n>0)printf "%.2f (n=%d)",s/n,n; else print "NA"}')
  log "$tag = $v"
done; done
log "ALL DONE: $OUT  (iid refs seed0: NoAdapt 47.69, T3A 47.79, ZOA 51.40, NEO 51.35, Ours 54.07)"
