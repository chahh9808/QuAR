#!/usr/bin/env bash
# Table 1's full-precision row on three seeds, so it carries a standard deviation
# like every other cell in that table. Only QuAR needs the seeds: No Adapt holds
# no state, so its 15-corruption mean is identical across seeds, which is what
# the caption already says.
#
# Flags verbatim from run_p13_3seed_fp.sh's fp_vitb jobs: no --quant, ViT-B,
# ImageNet-C severity 5, all 15 corruptions, RobustBench 5k, batch 64, episodic,
# and the full-precision source statistics.
set -uo pipefail
PY="${PY:-python3}"; F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$F/robustbench/data/imagenet_test_image_ids.txt; CMAP=$F/robustbench/data/imagenet_class_to_id_map.json
OUT=$F/runs/fp_3seed_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$F/runs/last_fp_3seed.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
EV="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 3 \
--corruption all --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
--arch vit_base_patch16_224"
OURSF="--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all \
--quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 \
--quar_local_stats_file $F/stats/qac_stats_fp_all_layers_seed0.pt \
--quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
--quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
--quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $F/stats/cls_stats_fp_seed0.pt \
--quar_global_target_ema_alpha 0.05"
cd "$F"
run(){ local g=$1 t=$2 s=$3; shift 3; mkdir -p "$OUT/$t"; log "[$t g$g] start"
  CUDA_VISIBLE_DEVICES=$g $PY -u "$F/main.py" $EV --seed "$s" "$@" \
    --output "$OUT/$t" --tag "_$t" >"$OUT/$t.log" 2>&1 \
    && log "[$t] done" || log "[$t] FAIL exit=$?"; }
( run 0 FP_NOADAPT 0 --algorithm no_adapt ) &
( run 1 FP_OURS_s0 0 $OURSF ) &
( run 2 FP_OURS_s1 1 $OURSF ) &
( run 3 FP_OURS_s2 2 $OURSF ) &
wait; log "FP_3SEED_DONE $OUT"
for t in FP_NOADAPT FP_OURS_s0 FP_OURS_s1 FP_OURS_s2; do
  a=$(find "$OUT/$t" -name "*acc=*" 2>/dev/null | grep -oE "acc=[0-9.]+" | head -1)
  echo "$t  ${a:-NA}"
done
