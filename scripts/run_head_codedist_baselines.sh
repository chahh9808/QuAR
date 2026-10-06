#!/usr/bin/env bash
# Head-quantizer code-distribution drift for the remaining Figure 1(b) methods
# (T3A, FOA, FOZO, ZOA), completing the 7-method terminal group.
#
# IMPORTANT, on the 2-forward methods. The codedist hook accumulates on every
# forward pre-hook call and consults no gate, so for FOA/FOZO/ZOA the histogram
# pools the candidate forward(s) and the deployed forward. That is exactly how
# the INTERMEDIATE numbers already in Figure 1(b) were measured, so measuring
# the terminal interface the same way keeps the two interface groups on one
# protocol. It is a shared limitation of both groups, not a terminal-only one,
# and it should be stated once rather than used to drop methods from one group.
#
# Per-method flags replicate run_codedist_seeds12.sh mflags() verbatim; the only
# differences from that launcher are --codedist_scope head and --codedist_src.
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
SRC=$QUAR/stats/qac_stats_w6a6_codehist_head.pt
OUT=$QUAR/runs/head_codedist_base_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
echo "$OUT">"$QUAR/runs/last_head_codedist_base_dir.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
W6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"
BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 4 \
  --corruption all --seed 0 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $W6 \
  --codedist_diag --codedist_src $SRC --codedist_scope head"
mflags(){ case "$1" in
  T3A)  echo "--algorithm t3a --t3a_filter_k 20 --arch vit_base_patch16_224";;
  FOA)  echo "--algorithm foa --popsize 1 --foa_stats_dir $QUAR/foa_stats";;
  FOZO) echo "--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 --fozo_stats_file $QUAR/stats/train_stats_2.pt";;
  ZOA)  echo "--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 --zoa_layer_balance none --official_zoa_stats_file $QUAR/zoa_stats/w6a6.pt";;
esac; }
run(){ local gpu=$1 tag=$2; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $BASE $(mflags "$tag") \
    --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 \
    && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
cd "$QUAR"
( run 1 T3A ) & ( run 2 FOA ) & ( run 3 FOZO ) & ( run 4 ZOA ) &
wait; log "ALL DONE: $OUT"
for t in T3A FOA FOZO ZOA; do
  v=$(grep -hoE "\[CODEDIST\] [a-z_]+ = [0-9.]+" "$OUT/$t.log" 2>/dev/null | grep -oE "[0-9.]+$" \
      | awk '{x+=$1;k++} END{if(k)printf "%.5f (n=%d)",x/k,k; else print "NA"}')
  echo "$t  code-dist L1 = $v"
done
