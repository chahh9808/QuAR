#!/usr/bin/env bash
# ImageNet-C-bar (10 algorithmic corruptions, Mintun et al. 2021): generalization to a
# DIFFERENT family of corruptions than the standard ImageNet-C. 7 methods x 4 bitwidths, 8 GPUs.
#
# Setting: ViT-B | ImageNet-C-bar 10 corruptions | severity level 5 (highest) | RobustBench 5k
#   subset | bs=64 | seed 0 | episodic | no ECE. 1000-class (no mask). Ours quar_local_lambda=0.5,
#   EMA=0.05 (deployed default). C-bar images generated on the 5k subset, severities remapped 1-5.
# Prereq: make_imagenet_c_bar.py generation done + remap_cbar_levels.py run.
# Usage: bash run_cbar.sh
set -uo pipefail
PYTHON="${PYTHON:-python3}"
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root (…/Data/CLS-LOC)}"
DATA_C="${DATA_C:?set DATA_C to the ImageNet-C-bar root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt
CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
FOA_STATS=$QUAR/foa_stats
CORR="caustic_refraction,inverse_sparkles,sparkles,perlin_noise,blue_noise_sample,plasma_noise,checkerboard_cutout,cocentric_sine_waves,single_frequency_greyscale,brownish_noise"
OUTBASE=$QUAR/runs/cbar_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTBASE"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

quant_flags() {
  case "$1" in
    w3a3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $QUAR/checkpoints/w3a3_checkpoint_opt.pt" ;;
    w4a4) echo "--quant --quant_w_bit 4 --quant_a_bit 4 --adalog_checkpoint $QUAR/checkpoints/w4a4_checkpoint_opt.pt" ;;
    w6a6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $QUAR/checkpoints/ptq4vit_w6a6_vitb.pt" ;;
    w8a8) echo "--quant --quant_w_bit 8 --quant_a_bit 8 --ptq_checkpoint $QUAR/checkpoints/ptq4vit_w8a8_vitb.pt" ;;
  esac
}
ours_flags() {
  local bw=$1
  echo "--algorithm no_adapt \
    --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
    --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 \
    --quar_local_stats_file $QUAR/stats/qac_stats_${bw}_all_layers_seed0.pt \
    --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
    --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std \
    --quar_global_alpha 1.0 --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 \
    --quar_global_stats_file $QUAR/stats/cls_stats_${bw}_seed0.pt --quar_global_target_ema_alpha 0.05"
}
method_flags() {
  case "$1" in
    NOADAPT) echo "--algorithm no_adapt" ;;
    T3A)     echo "--algorithm t3a --t3a_filter_k 20 --arch vit_base_patch16_224" ;;
    FOA)     echo "--algorithm foa --popsize 1 --foa_stats_dir $FOA_STATS" ;;
    FOZO)    echo "--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 --fozo_stats_file $QUAR/stats/train_stats_2.pt" ;;
    ZOA)     echo "--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 --zoa_layer_balance none --official_zoa_stats_file $QUAR/zoa_stats/$2.pt" ;;
    NEO)     echo "--algorithm no_adapt --neo_enable" ;;
    OURS)    ours_flags "$2" ;;
  esac
}
run_cell() {
  local gpu=$1 bw=$2 m=$3; local tag="${bw}_${m}"
  mkdir -p "$OUTBASE/$tag"
  log "[$tag GPU=$gpu] starting"
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u $QUAR/main.py \
    --data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval \
    --corruption $CORR --batch_size 64 --workers 8 --seed 0 \
    --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
    $(quant_flags $bw) $(method_flags $m $bw) \
    --output "$OUTBASE/$tag" --tag "_${tag}" \
    > "$OUTBASE/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAILED (exit $?)"
}
cd "$QUAR"
log "Output: $OUTBASE  (10 C-bar corruptions, level 5)"
BWS=(w3a3 w4a4 w6a6 w8a8); METHODS=(NOADAPT T3A FOA FOZO ZOA NEO OURS)
declare -a C0 C1 C2 C3 C4 C5 C6 C7; i=0
for bw in "${BWS[@]}"; do for m in "${METHODS[@]}"; do eval "C$((i%8))+=(\"$bw:$m\")"; i=$((i+1)); done; done
run_chain() { local g=$1; local -n cells="C$g"; for c in "${cells[@]}"; do run_cell "$g" "${c%%:*}" "${c##*:}"; done; }
for g in 0 1 2 3 4 5 6 7; do run_chain "$g" & done
wait
log "All 28 C-bar cells done. Output: $OUTBASE"
echo "$OUTBASE" > "$QUAR/runs/last_cbar_dir.txt"
