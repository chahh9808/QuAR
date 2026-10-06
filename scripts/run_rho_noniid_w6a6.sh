#!/usr/bin/env bash
# rho-CLAMP ablation under NON-IID, cumulative (deployed) estimator. W6A6 ViT-B, 15-corr 5k seed0 episodic.
# Question: does the variance-ratio clamp rho earn its keep under non-iid (where class-concentrated
#   batches can drive a per-channel sigma_t -> 0 -> sigma_s/sigma_t explosion)? On IID it costs <=0.2 pts
#   (sens_gap bs64: rho3=54.37 vs rho-inf=54.54), so the clamp must justify itself in the regime it guards.
# Design (cumulative fixed): {default(iid), class_sorted(non-iid)} x {bs8(high conc), bs64(deployed)} x {rho=3, rho=inf(1e6)}.
#   DECISIVE cell-pair: srt_bs8 rho3 vs rho-inf. If clamp helps, rho-inf degrades there and the gap > iid gap.
# Sanity check: iid_bs64 rho3/rho-inf should reproduce sens_gap 54.37 / 54.54.
# Recipe = run_noniid_w6a6.sh OURS + --recal_cumulative (matches sens_gap rho-cell Namespace exactly).
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
QAC=$QUAR/stats/qac_stats_w6a6_all_layers_seed0.pt; CLS=$QUAR/stats/cls_stats_w6a6_seed0.pt
W6A6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"
OUT=$QUAR/runs/rho_noniid_w6a6_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_rho_noniid_dir.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
basef(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size $1 --workers 8 \
  --corruption all --seed 0 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP"; }
# OURS cumulative; $1 = clip_ratio (rho), applied to BOTH qac and cls branches.
ours(){ local rho=$1; echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all \
  --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio $rho --quar_local_lambda 0.5 --quar_local_stats_file $QAC \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag --recal_cumulative \
  --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
  --quar_global_eps 1e-6 --quar_global_clip_ratio $rho --quar_global_stats_file $CLS --quar_global_target_ema_alpha 0.05"; }
ord_flags(){ [ "$1" = sorted ] && echo "--stream_order class_sorted" || echo "--stream_order default"; }
# run: gpu order bs rho_label rho_val
run(){ local gpu=$1 ord=$2 bs=$3 rl=$4 rv=$5; local tag="${ord}_bs${bs}_${rl}"; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start (rho=$rv)"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $(basef $bs) $W6A6 $(ours $rv) $(ord_flags $ord) \
    --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
cd "$QUAR"
log "OUT=$OUT | rho-clamp non-iid ablation, cumulative W6A6 15-corr 5k seed0. 8 cells over GPUs 0-7."
# 8 cells, one per GPU (all GPUs free). gpu order bs rho_label rho_val
run 0 iid    64 rho3   3       &
run 1 iid    64 rhoInf 1000000 &
run 2 iid    8  rho3   3       &
run 3 iid    8  rhoInf 1000000 &
run 4 sorted 64 rho3   3       &
run 5 sorted 64 rhoInf 1000000 &
run 6 sorted 8  rho3   3       &
run 7 sorted 8  rhoInf 1000000 &
wait
log "=== rho-clamp non-iid results (15-corr mean top-1) ==="
for ord in iid sorted; do for bs in 64 8; do for rl in rho3 rhoInf; do
  tag="${ord}_bs${bs}_${rl}"; [ -f "$OUT/$tag.log" ] || continue
  v=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/$tag.log" | grep -oE "[0-9.]+$" | awk '{s+=$1;n++} END{if(n>0)printf "%.2f (n=%d)",s/n,n; else print "NA"}')
  log "$tag = $v"
done; done; done
log "ALL DONE: $OUT  (sanity: iid_bs64 rho3->54.37, rhoInf->54.54 per sens_gap)"
