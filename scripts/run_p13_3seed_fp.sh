#!/usr/bin/env bash
# (A) Backbone 3-seed: {vitb,deitb,vits} x {NoAdapt,T3A,NEO,ZOA,OURS} x seed{0,1,2}, W6A6 PTQ4ViT.
#     OURS uses quar_local_lambda 0.5 (paper default; note run_p13_compare.sh used 1.0). Reuses p13 ckpts/stats.
# (C) FP(W32A32) reference: vitb No-Adapt + OURS, NO --quant, seed0.
# ImageNet-C sev5, 15-corruption RobustBench 5k, bs64, episodic.
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
P13=$QUAR/checkpoints
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
OUT=$QUAR/runs/backbone3seed_fp_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_backbone3seed_fp_dir.txt"
MB=${MB:--1}   # set MB=3 for a quick smoke test
arch(){ case "$1" in vitb) echo vit_base_patch16_224;; deitb) echo deit_base_patch16_224;; vits) echo vit_small_patch16_224;; esac; }
ev(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 4 \
  --corruption all --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP --max_eval_batches $MB"; }
qbb(){ echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $P13/ptq4vit_w6a6_${1}_p13.pt --arch $(arch $1)"; }
ours(){ local sf=$1 cf=$2; echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all \
  --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $sf \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
  --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
  --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $cf --quar_global_target_ema_alpha 0.05"; }
mflags(){ local m=$1 tag=$2
  case "$m" in
    NoAdapt) echo "--algorithm no_adapt";;
    T3A)     echo "--algorithm t3a --t3a_filter_k 20";;
    NEO)     echo "--algorithm no_adapt --neo_enable";;
    ZOA)     echo "--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 --zoa_layer_balance none --official_zoa_stats_file $QUAR/zoa_stats/${tag}_w6a6.pt";;
    OURS)    ours "$QUAR/stats/qac_stats_w6a6_all_layers_seed0_${tag}.pt" "$QUAR/stats/cls_stats_w6a6_seed0_${tag}.pt";;
  esac; }
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
run(){ local gpu=$1; shift; local name=$1; shift; mkdir -p "$OUT/$name"; log "[$name g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py "$@" --output "$OUT/$name" --tag "_$name" >"$OUT/$name.log" 2>&1 \
    && log "[$name] done" || log "[$name] FAIL $?"; }
cd "$QUAR"
declare -a JN JC
add(){ JN+=("$1"); JC+=("$2"); }
# FP reference (vitb, no quant), seed0
add "fp_vitb_NoAdapt_s0" "$(ev) --seed 0 --arch $(arch vitb) --algorithm no_adapt"
add "fp_vitb_OURS_s0"    "$(ev) --seed 0 --arch $(arch vitb) $(ours $QUAR/stats/qac_stats_fp_all_layers_seed0.pt $QUAR/stats/cls_stats_fp_seed0.pt)"
# backbone 3-seed
for s in 0 1 2; do for tag in vitb deitb vits; do for m in NoAdapt T3A NEO ZOA OURS; do
  add "bb_${tag}_${m}_s${s}" "$(ev) --seed $s $(qbb $tag) $(mflags $m $tag)"
done; done; done
log "total jobs: ${#JN[@]}  MB=$MB  OUT=$OUT"
i=0
for k in "${!JN[@]}"; do
  run $(( i % 8 )) "${JN[$k]}" ${JC[$k]} &
  i=$(( i+1 )); (( i % 8 == 0 )) && wait
done
wait; log "ALL DONE: $OUT"
echo "=== RESULTS (name : mean_acc_over_15) ==="
for k in "${!JN[@]}"; do n="${JN[$k]}"
  a=$(grep -oE "mean acc of corruption: [0-9.]+" "$OUT/$n.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
  printf "%-26s %s\n" "$n" "${a:-NA}"
done
