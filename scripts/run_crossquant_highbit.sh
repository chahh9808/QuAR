#!/usr/bin/env bash
# Matched-bit-width CROSS-QUANTIZER comparison at high bit: AdaLog vs PTQ4ViT, 7 methods each,
# ImageNet-C 15 corruptions, 5k subset, seed 0, level 5, episodic.
#
# The W6A6 and W8A8 halves of tab:crossquant. Same protocol as run_crossquant_w3a3.sh and
# run_crossquant_w4a4.sh, which cover the low-bit halves. At W6A6 and W8A8 the protocol quantizer
# is PTQ4ViT, so AdaLog is the off-protocol side here and is freshly calibrated:
# checkpoints/w{bw}_checkpoint_opt.pt with stats/{qac,cls}_stats_w{bw}_adalog_*.
#
# That table is assembled from three campaigns and this one supplies the baseline columns only.
# The published ZOA cells come from run_zoa_all_remaining.sh (its CROSSQ group), and the published QuAR cells from the cq group of
# run_cumulative_appendix.sh for the off-protocol quantizer and run_cumulative_maintable.sh for the
# protocol one. Reading this campaign's ZOA or QuAR rows against the paper will look like a
# mismatch; the published cells are the ones named above.
#
# Ours uses the deployed configuration: block8_9_10_11_all, lambda=0.5, EMA=0.05.
#
# Usage
#   DATA=... DATA_C=... bash scripts/run_crossquant_highbit.sh w6a6
#   DATA=... DATA_C=... bash scripts/run_crossquant_highbit.sh w8a8
#   FREE_GPUS="0 1 2 3" to restrict the GPUs used.
set -uo pipefail
BW="${1:?usage: run_crossquant_highbit.sh w6a6|w8a8}"
PYTHON="${PYTHON:-python3}"
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root (…/Data/CLS-LOC)}"
DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt
CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
FOA_STATS=$QUAR/foa_stats
read -r -a FREE_GPUS <<<"${FREE_GPUS:-0 1 2 3 4 5 6 7}"

case "$BW" in
  w6a6) WB=6; AB=6; PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt ;;
  w8a8) WB=8; AB=8; PTQ=$QUAR/checkpoints/ptq4vit_w8a8_vitb.pt ;;
  *) echo "bad bit-width '$BW' (expected w6a6 or w8a8)"; exit 1 ;;
esac

OUTBASE=$QUAR/runs/crossquant_${BW}_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUTBASE"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

quant_flags() {
  case "$1" in
    adalog)  echo "--quant --quant_w_bit $WB --quant_a_bit $AB --adalog_checkpoint $QUAR/checkpoints/${BW}_checkpoint_opt.pt" ;;
    ptq4vit) echo "--quant --quant_w_bit $WB --quant_a_bit $AB --ptq_checkpoint $PTQ" ;;
  esac
}
qac_file() { case "$1" in
  adalog)  echo "$QUAR/stats/qac_stats_${BW}_adalog_all_layers_seed0.pt";;
  ptq4vit) echo "$QUAR/stats/qac_stats_${BW}_all_layers_seed0.pt";;
esac; }
cls_file() { case "$1" in
  adalog)  echo "$QUAR/stats/cls_stats_${BW}_adalog_seed0.pt";;
  ptq4vit) echo "$QUAR/stats/cls_stats_${BW}_seed0.pt";;
esac; }

ours_flags() {  # $1=quantizer
  local q=$1
  echo "--algorithm no_adapt \
    --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
    --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 \
    --quar_local_stats_file $(qac_file $q) \
    --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
    --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std \
    --quar_global_alpha 1.0 --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 \
    --quar_global_stats_file $(cls_file $q) --quar_global_target_ema_alpha 0.05"
}
method_flags() {  # $1=method $2=quantizer
  case "$1" in
    NOADAPT) echo "--algorithm no_adapt" ;;
    T3A)     echo "--algorithm t3a --t3a_filter_k 20 --arch vit_base_patch16_224" ;;
    FOA)     echo "--algorithm foa --popsize 1 --foa_stats_dir $FOA_STATS" ;;
    FOZO)    echo "--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 --fozo_stats_file $QUAR/stats/train_stats_2.pt" ;;
    ZOA)     echo "--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 --zoa_layer_balance none --official_zoa_stats_file $QUAR/zoa_stats/${BW}.pt" ;;
    NEO)     echo "--algorithm no_adapt --neo_enable" ;;
    OURS)    ours_flags "$2" ;;
  esac
}
run_cell() {  # $1=gpu $2=quantizer $3=method
  local gpu=$1 q=$2 m=$3; local tag="${q}_${m}"
  mkdir -p "$OUTBASE/$tag"
  log "[$tag GPU=$gpu] starting"
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u "$QUAR/main.py" \
    --data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --corruption all \
    --batch_size 64 --workers 8 --seed 0 \
    --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
    $(quant_flags $q) $(method_flags $m $q) \
    --output "$OUTBASE/$tag" --tag "_${tag}" \
    > "$OUTBASE/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAILED (exit $?)"
}

cd "$QUAR"
log "Output: $OUTBASE  ($BW cross-quantizer: AdaLog vs PTQ4ViT, 15 corruptions, GPUs ${FREE_GPUS[*]})"
QUANTS=(adalog ptq4vit); METHODS=(NOADAPT T3A FOA FOZO ZOA NEO OURS)
NG=${#FREE_GPUS[@]}
declare -a CH0 CH1 CH2 CH3 CH4 CH5 CH6 CH7; i=0
for q in "${QUANTS[@]}"; do for m in "${METHODS[@]}"; do eval "CH$((i%NG))+=(\"$q:$m\")"; i=$((i+1)); done; done
run_chain() { local idx=$1; local gpu=${FREE_GPUS[$idx]}; local -n cells="CH$idx"; for c in "${cells[@]}"; do run_cell "$gpu" "${c%%:*}" "${c##*:}"; done; }
for idx in $(seq 0 $((NG-1))); do run_chain "$idx" & done
wait
log "All $BW cross-quantizer cells done. Output: $OUTBASE"
echo "$OUTBASE" > "$QUAR/runs/last_crossquant_${BW}_dir.txt"
