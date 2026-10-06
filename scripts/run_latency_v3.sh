#!/usr/bin/env bash
# Clean per-image latency + peak memory: ONE measurement at a time on GPU 0 ONLY, fully sequential
# (no other GPU/CPU co-tenant to contend for CPU / dataloader / PCIe / host-mem bandwidth).
# 9 methods x 5 reps = 45 runs; each rep times every batch forward over ~50 batches
# (--latency_profile -> LATPROF median per-image ms; main.py logs PEAK_GPU_MEM_MB). W6A6 ViT-B/16,
# gaussian_noise, bs64. Report = median over the 5 reps (latency AND peak memory).
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
QACB3=$QUAR/stats/qac_stats_w6a6_all_layers_seed0.pt; CLS=$QUAR/stats/cls_stats_w6a6_seed0.pt
OUT=$QUAR/runs/latency_v3_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_latency_v3_dir.txt"
W6A6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"
BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 4 \
  --corruption gaussian_noise --seed 0 --max_eval_batches 50 --latency_profile \
  --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP"
QAC="--quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QACB3 --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag"
CLSF="--quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $CLS --quar_global_target_ema_alpha 0.05"
declare -A FL
FL[NoAdapt]="--algorithm no_adapt"
FL[T3A]="--algorithm t3a --t3a_filter_k 20 --arch vit_base_patch16_224"
FL[FOA]="--algorithm foa --popsize 1 --foa_stats_dir $QUAR/foa_stats"
FL[FOZO]="--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 --fozo_stats_file $QUAR/stats/train_stats_2.pt"
FL[ZOA]="--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 --zoa_layer_balance none --official_zoa_stats_file $QUAR/zoa_stats/w6a6.pt"
FL[NEO]="--algorithm no_adapt --neo_enable"
FL[Ours]="--algorithm no_adapt $QAC $CLSF"
FL[Final]="--algorithm no_adapt $CLSF"
FL[Local]="--algorithm no_adapt $QAC"
METHODS=(NoAdapt T3A FOA FOZO ZOA NEO Ours Final Local)
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
cd "$QUAR"
for m in "${METHODS[@]}"; do
  for rep in 1 2 3 4 5; do
    mkdir -p "$OUT/${m}_r${rep}"
    CUDA_VISIBLE_DEVICES=0 $PY -u main.py $BASE $W6A6 ${FL[$m]} \
      --output "$OUT/${m}_r${rep}" --tag "_${m}_r${rep}" >"$OUT/${m}_r${rep}.log" 2>&1
    v=$(grep -oE "median_per_image_ms=[0-9.]+" "$OUT/${m}_r${rep}.log" | grep -oE "[0-9.]+$")
    log "[$m rep$rep] done  per-img median=${v:-NA} ms"
  done
done
log "ALL DONE: $OUT"
