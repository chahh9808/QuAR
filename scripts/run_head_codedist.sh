#!/usr/bin/env bash
# Head-quantizer code-distribution drift, the companion measurement to Figure 1(b).
#
# Figure 1(b) reads the code-distribution L1 at the INTERMEDIATE interfaces
# (blocks 8-11 x {qkv,proj,fc1}), where NEO and T3A come out byte-identical to
# No Adapt because neither touches those activations. This reads the SAME metric
# at the interface NEO does act at: the classifier head's activation quantizer.
#
# Config replicates the fig-1b codedist cfg exactly (W6A6, 15-corr 5k, bs64,
# nwarm=64, lag=False, seed 0) so the two interface groups are comparable; only
# --codedist_scope and --codedist_src differ.
#
# Source: qac_stats_w6a6_codehist_head.pt, built with the SAME ptq checkpoint
# (build_qac_stats.py --quar_local_scope head --max_batches 200 --save_code_hist), so
# source and test share one quantizer scale.
#
# FOA/FOZO/ZOA are absent: 2 forwards per batch would pool the candidate
# forwards into the same histogram accumulator. T3A is absent: it rewrites the
# classifier's prototypes, not its input, so at this interface it is No Adapt.
# Every method here is 1F.
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
SRC=$QUAR/stats/qac_stats_w6a6_codehist_head.pt
OUT=$QUAR/runs/head_codedist_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_head_codedist_dir.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
W6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"
BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 4 \
  --corruption all --seed 0 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $W6 \
  --codedist_diag --codedist_src $SRC --codedist_scope head"
LOCAL="--quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
  --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QUAR/stats/qac_stats_w6a6_all_layers_seed0.pt \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 64"
CLSF="--quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
  --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $QUAR/stats/cls_stats_w6a6_seed0.pt \
  --quar_global_target_ema_alpha 0.05"
run(){ local gpu=$1 tag=$2; shift 2; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $BASE --algorithm no_adapt \
    --output "$OUT/$tag" --tag "_$tag" "$@" \
    >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
cd "$QUAR"
( run 1 NOADAPT ) & ( run 2 NEO --neo_enable ) & ( run 3 FINAL $CLSF ) &
( run 4 LOCAL $LOCAL ) & ( run 5 OURS $LOCAL $CLSF ) &
wait; log "ALL DONE: $OUT"
for t in NOADAPT NEO FINAL LOCAL OURS; do
  v=$(grep -oE "\[CODEDIST\] [a-z_]+ = [0-9.]+" "$OUT/$t.log" 2>/dev/null | grep -oE "[0-9.]+$" \
      | awk '{x+=$1;k++} END{if(k)printf "%.5f (n=%d)",x/k,k; else print "NA"}')
  echo "$t  code-dist L1 = $v"
done
