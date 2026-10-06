#!/usr/bin/env bash
# Full switch of the code-distribution analysis to the value-space Wasserstein-1
# metric, over all 17 recalibration interfaces, 9 configurations x 3 seeds.
#
# Why W1 replaces the occupancy L1. The L1 counts how many BINS mass moved and
# ignores bin width. On PTQ4ViT's twin-uniform post-GELU grid a negative bin is
# 0.0053 wide and a positive one up to 0.9, so a 169x smaller value change scores
# identically (verified: moving 0.1 mass one fine bin gives L1 0.2000 / W1
# 0.000531; one coarse bin gives L1 0.2000 / W1 0.090000). W1 de-quantizes the
# code axis first, so every move is weighted by the distance the value actually
# travelled, and one metric then covers every grid type. Under L1 the fc2
# interfaces had to be dropped and scored -88% when included; under W1 they score
# +25% and no interface needs excluding.
#
# Interfaces (--codedist_scope all17): blocks 8-11 x {qkv, proj, fc1, fc2} plus
# the classifier head input. Per-interface distances are logged for both metrics,
# so any subset aggregate (a 12- or 16-interface set, or the head alone) is recoverable.
#
# Config: W6A6 PTQ4ViT, ImageNet-C sev5 all 15 corruptions, RobustBench 5k,
# bs 64, episodic. Our configurations use the deployed estimator
# (--recal_cumulative), the one tab:main's accuracies come from.
set -uo pipefail
PY="${PY:-python3}"; F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$F/robustbench/data/imagenet_test_image_ids.txt; CMAP=$F/robustbench/data/imagenet_class_to_id_map.json
PTQ=$F/checkpoints/ptq4vit_w6a6_vitb.pt
SRC=$F/stats/qac_stats_w6a6_codehist_all17.pt
OUT=$F/runs/w1_codedist_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$F/runs/last_w1_codedist.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
base(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --corruption all \
--batch_size 64 --workers 4 --seed $1 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ \
--codedist_diag --codedist_src $SRC --codedist_scope all17"; }
LOCAL="--quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
--quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $F/stats/qac_stats_w6a6_all_layers_seed0.pt \
--quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 64"
CLSF="--quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
--quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $F/stats/cls_stats_w6a6_seed0.pt \
--quar_global_target_ema_alpha 0.05"
mf(){ case "$1" in
  NOADAPT) echo "--algorithm no_adapt";;
  T3A)     echo "--algorithm t3a --t3a_filter_k 20 --arch vit_base_patch16_224";;
  FOA)     echo "--algorithm foa --popsize 1 --foa_stats_dir $F/foa_stats";;
  FOZO)    echo "--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 --fozo_stats_file $F/stats/train_stats_2.pt";;
  ZOA)     echo "--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 --zoa_layer_balance none --official_zoa_stats_file $F/zoa_stats/w6a6.pt";;
  NEO)     echo "--algorithm no_adapt --neo_enable";;
  FINAL)   echo "--algorithm no_adapt --recal_cumulative $CLSF";;
  LOCALONLY) echo "--algorithm no_adapt --recal_cumulative $LOCAL";;
  OURS)    echo "--algorithm no_adapt --recal_cumulative $LOCAL $CLSF";;
esac; }
cd "$F"
GPUS=(0 1 2 3 4 5 6 7)
Q="$OUT/queue"; : >"$Q"
for m in NOADAPT T3A FOA FOZO ZOA NEO FINAL LOCALONLY OURS; do for s in 0 1 2; do echo "$m $s" >>"$Q"; done; done
log "$(wc -l <"$Q") runs (9 configs x 3 seeds) on GPUs [${GPUS[*]}] -> $OUT"
i=0
while read -r m s; do
  t="${m}_s${s}"; g=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i+1)); mkdir -p "$OUT/$t"
  ( log "[$t g$g] start"
    CUDA_VISIBLE_DEVICES=$g $PY -u $F/main.py $(base "$s") $(mf "$m") \
      --output "$OUT/$t" --tag "_$t" >"$OUT/$t.log" 2>&1 \
      && log "[$t] done" || log "[$t] FAIL exit=$?" ) &
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
done <"$Q"
wait; log "W1_CODEDIST_DONE $OUT"
