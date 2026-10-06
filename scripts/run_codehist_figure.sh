#!/usr/bin/env bash
# Per-channel code histogram at one interface, for the figure that shows the shift's effect
# on the code distribution rather than the fraction of it a method removes.
#
# Interface: blocks.10.attn.proj. Chosen because it is a single-scale symmetric uniform
# quantizer (so the de-quantized axis has one step and is straightforward to plot) and it is
# one of the starved interfaces, sigma_t/sigma_s = 0.744 with 21.7% of its channels forfeiting
# at least half a bit of code resolution.
#
# Three runs, identical except for the stream and the adapter:
#   SOURCE   clean ImageNet (corruption=original), No Adapt -> the calibrated code distribution
#   NOADAPT  ImageNet-C sev5 all 15, No Adapt              -> what the shift does to it
#   OURS     ImageNet-C sev5 all 15, deployed QuAR         -> what the recalibration returns
# W6A6 PTQ4ViT, RobustBench 5k, bs 64, workers 3, seed 0, episodic, cumulative estimator.
set -uo pipefail
PY="${PY:-python3}"; F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$F/robustbench/data/imagenet_test_image_ids.txt; CMAP=$F/robustbench/data/imagenet_class_to_id_map.json
PTQ=$F/checkpoints/ptq4vit_w6a6_vitb.pt
IFACE=blocks.10.attn.proj
OUT=$F/runs/codehist_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$F/runs/last_codehist.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 \
--workers 3 --seed 0 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ --codehist_dump $IFACE"
OURSF="--recal_cumulative --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all \
--quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 \
--quar_local_stats_file $F/stats/qac_stats_w6a6_all_layers_seed0.pt \
--quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 64 \
--quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
--quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $F/stats/cls_stats_w6a6_seed0.pt \
--quar_global_target_ema_alpha 0.05"
cd "$F"; i=0
run(){ local t=$1 g=$2; shift 2; mkdir -p "$OUT/$t"
  ( log "[$t g$g] start"
    CUDA_VISIBLE_DEVICES=$g $PY -u "$F/main.py" $BASE "$@" --output "$OUT/$t" --tag "_$t" \
      >"$OUT/$t.log" 2>&1 && log "[$t] done" || log "[$t] FAIL exit=$?" ) & }
run SOURCE  0 --corruption original --algorithm no_adapt
run NOADAPT 1 --corruption all      --algorithm no_adapt
run OURS    2 --corruption all      --algorithm no_adapt $OURSF
wait; log "CODEHIST_DONE $OUT"
for t in SOURCE NOADAPT OURS; do f=$(find "$OUT/$t" -name "codehist_*.pt" | head -1); echo "$t ${f:-MISSING}"; done
