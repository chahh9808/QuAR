#!/usr/bin/env bash
# FOA recommended-config (popsize=28) per-image latency, SAME protocol as run_latency_v3.sh:
# GPU 0 ONLY, fully sequential, bs64, gaussian_noise, --latency_profile over ~50 batches,
# median per-image ms. Reps=5; report = median over reps. W6A6 ViT-B/16.
# Also re-measures FOA popsize=1 (paper FOA K=2) as a SANITY check (must reproduce ~13.0 ms)
# so FOA@28 is directly comparable to the existing Table-1 latencies on this machine state.
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
OUT=$QUAR/runs/foa28_latency_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_foa28_latency_dir.txt"
W6A6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"
BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 4 \
  --corruption gaussian_noise --seed 0 --max_eval_batches 50 --latency_profile \
  --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP"
declare -A FL
FL[FOA1]="--algorithm foa --popsize 1  --foa_stats_dir $QUAR/foa_stats"
FL[FOA28]="--algorithm foa --popsize 28 --foa_stats_dir $QUAR/foa_stats"
METHODS=(FOA1 FOA28)
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
cd "$QUAR"
for m in "${METHODS[@]}"; do
  for rep in 1 2 3 4 5; do
    mkdir -p "$OUT/${m}_r${rep}"
    CUDA_VISIBLE_DEVICES=0 $PY -u main.py $BASE $W6A6 ${FL[$m]} \
      --output "$OUT/${m}_r${rep}" --tag "_${m}_r${rep}" >"$OUT/${m}_r${rep}.log" 2>&1
    v=$(grep -oE "median_per_image_ms=[0-9.]+" "$OUT/${m}_r${rep}.log" | grep -oE "[0-9.]+$")
    log "[$m rep$rep] per-img median=${v:-NA} ms"
  done
done
log "=== MEDIAN over 5 reps (per-image ms) ==="
for m in "${METHODS[@]}"; do
  med=$(for rep in 1 2 3 4 5; do grep -oE "median_per_image_ms=[0-9.]+" "$OUT/${m}_r${rep}.log" | grep -oE "[0-9.]+$"; done \
    | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  log "$m = $med ms"
done
log "ALL DONE: $OUT  (ref: FOA1 should reproduce ~13.0 ms; Ours 6.6, NoAdapt/NEO 6.0)"
