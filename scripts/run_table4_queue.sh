#!/usr/bin/env bash
# R/Sketch 3-seed (seeds 1,2; seed0 already in tab:t4app). 7 methods x 4 bitwidths = 56 jobs.
# 8-GPU work-queue (per-GPU workers + flock; skip-completed). Ours = B3 scope + alpha=0.5
# (matching the committed alpha=0.5 table). Each job evaluates --corruption rendition,sketch
# (full ImageNet-R 30k + ImageNet-Sketch 50.9k); R and Sketch top-1 are logged separately.
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
RDIR="${DATA_R:?set DATA_R to the imagenet-r root}"; SKDIR="${DATA_SKETCH:?set DATA_SKETCH to the imagenet-sketch root}"
OUT=${OUT:-$QUAR/runs/table4_3seed_$(date +%Y%m%d_%H%M%S)}; mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_table4_3seed_dir.txt"
NGPU=${NGPU:-6}; MB=${MB:--1}; SEEDS=${SEEDS:-"1 2"}
PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
PTQ_W8A8=$QUAR/checkpoints/ptq4vit_w8a8_vitb.pt
BASE="--data $DATA --data_corruption $DATA_C --level 5 --data_rendition $RDIR --data_sketch $SKDIR \
  --corruption rendition,sketch --batch_size 64 --workers 4 --max_eval_batches $MB"
qflags(){ case "$1" in
  w3a3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $QUAR/checkpoints/w3a3_checkpoint_opt.pt";;
  w4a4) echo "--quant --quant_w_bit 4 --quant_a_bit 4 --adalog_checkpoint $QUAR/checkpoints/w4a4_checkpoint_opt.pt";;
  w6a6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6";;
  w8a8) echo "--quant --quant_w_bit 8 --quant_a_bit 8 --ptq_checkpoint $PTQ_W8A8";; esac; }
mflags(){ local m=$1 bw=$2
  case "$m" in
    NoAdapt) echo "--algorithm no_adapt";;
    T3A)     echo "--algorithm t3a --t3a_filter_k 20 --arch vit_base_patch16_224";;
    FOA)     echo "--algorithm foa --popsize 1 --foa_stats_dir $QUAR/foa_stats";;
    FOZO)    echo "--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 --fozo_stats_file $QUAR/stats/train_stats_2.pt";;
    ZOA)     echo "--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 --zoa_layer_balance none --official_zoa_stats_file $QUAR/zoa_stats/${bw}.pt";;
    NEO)     echo "--algorithm no_adapt --neo_enable";;
    OURS)    echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QUAR/stats/qac_stats_${bw}_all_layers_seed0.pt --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $QUAR/stats/cls_stats_${bw}_seed0.pt --quar_global_target_ema_alpha 0.05";;
  esac; }
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
done_job(){ [ "$(grep -c 'Under shift type .* Top-1 Accuracy' "$OUT/$1.log" 2>/dev/null)" -ge 2 ]; }
run(){ local gpu=$1; shift; local name=$1; shift; mkdir -p "$OUT/$name"; log "[$name g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py "$@" --output "$OUT/$name" --tag "_$name" >"$OUT/$name.log" 2>&1 \
    && log "[$name] done" || log "[$name] FAIL $?"; }
cd "$QUAR"
declare -a JN JC
for s in $SEEDS; do for bw in w3a3 w4a4 w6a6 w8a8; do for m in NoAdapt T3A FOA FOZO ZOA NEO OURS; do
  JN+=("t4_${bw}_${m}_s${s}"); JC+=("$BASE --seed $s $(qflags $bw) $(mflags $m $bw)")
done; done; done

QF=$(mktemp); LOCK="$QF.lock"; : > "$LOCK"; np=0
for k in "${!JN[@]}"; do done_job "${JN[$k]}" && continue; echo "$k" >> "$QF"; np=$((np+1)); done
log "pending: $np / ${#JN[@]}  NGPU=$NGPU  OUT=$OUT"
[ "$np" -eq 0 ] && { log "nothing to do"; rm -f "$QF" "$LOCK"; exit 0; }
pop(){ ( flock -x 9; read -r line < "$QF" 2>/dev/null || { echo ""; exit 0; }; sed -i '1d' "$QF"; echo "$line" ) 9>"$LOCK"; }
worker(){ local g=$1 k; while k=$(pop); [ -n "$k" ]; do run "$g" "${JN[$k]}" ${JC[$k]}; done; }
for g in $(seq 0 $((NGPU-1))); do worker "$g" & done
wait; rm -f "$QF" "$LOCK"; log "ALL DONE: $OUT"
echo "=== RESULTS (name : R / Sketch top-1) ==="
for k in "${!JN[@]}"; do n="${JN[$k]}"
  r=$(grep -oE "Under shift type rendition After .* Top-1 Accuracy: [0-9.]+" "$OUT/$n.log" 2>/dev/null | grep -oE "[0-9.]+$" | tail -1)
  sk=$(grep -oE "Under shift type sketch After .* Top-1 Accuracy: [0-9.]+" "$OUT/$n.log" 2>/dev/null | grep -oE "[0-9.]+$" | tail -1)
  printf "%-22s R=%-9s Sk=%-9s\n" "$n" "${r:-NA}" "${sk:-NA}"
done
