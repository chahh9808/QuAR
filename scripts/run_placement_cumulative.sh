#!/usr/bin/env bash
# CUMULATIVE re-run of tab:placement: Pre(explicit)/Post(output_only)/Code(coupled) x {W3A3,W6A6}.
# = run_alpha05_placement.sh recipe (block8_9_10_11_all, lambda=0.5, Local-only) + --recal_cumulative.
# All rows share the SAME cumulative estimator; rows differ ONLY in placement (mode/fold). 5 cells
# (W3A3_CODE is N/A: AdaLog log-quantizer has no affine fold for code-only). 5k 15-corr seed0.
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt; ADALOG_W3A3=$QUAR/checkpoints/w3a3_checkpoint_opt.pt
OUT=$QUAR/runs/placement_cum_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_placement_cum_dir.txt"
A=0.5
qf(){ case "$1" in w6a6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6";; w3a3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $ADALOG_W3A3";; esac; }
qbase(){ echo "--quar_local_enable --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda $A --quar_local_stats_file $QUAR/stats/qac_stats_$1_all_layers_seed0.pt --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag --recal_cumulative"; }
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
run(){ local gpu=$1 tag=$2 bw=$3; shift 3; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py --data $DATA --data_corruption $DATA_C --level 5 \
    --imagenet_5k_eval --batch_size 64 --workers 8 --corruption all --seed 0 \
    --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $(qf $bw) --algorithm no_adapt $(qbase $bw) "$@" \
    --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
cd "$QUAR"
( run 0 W3A3_PRE  w3a3 --quar_local_mode explicit ) &
( run 1 W3A3_POST w3a3 --quar_local_mode folded_quantizer --quar_local_fold_type output_only ) &
( run 2 W6A6_PRE  w6a6 --quar_local_mode explicit ) &
( run 3 W6A6_POST w6a6 --quar_local_mode folded_quantizer --quar_local_fold_type output_only ) &
( run 4 W6A6_CODE w6a6 --quar_local_mode folded_quantizer --quar_local_fold_type scale_offset_coupled ) &
wait
log "=== placement cumulative (15-corr mean top-1) ==="
for t in W3A3_PRE W3A3_POST W6A6_PRE W6A6_POST W6A6_CODE; do
  v=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/$t.log" 2>/dev/null|grep -oE "[0-9.]+$"|awk '{s+=$1;n++} END{if(n)printf "%.2f (n=%d)",s/n,n; else print "NA"}')
  log "  $t = $v"
done
log "ALL DONE: $OUT  (cf EMA: W6A6 PRE 52.82/POST 51.91/CODE 47.24; cumulative PRE should ~53.44 = Local-only)"
