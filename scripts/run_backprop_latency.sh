#!/usr/bin/env bash
# Per-image latency of backprop TTA (SAR/TENT/EATA) under the SAME controlled protocol as Table 1:
# single idle GPU 0, ONE measurement at a time, --latency_profile (CUDA events around model(images),
# batches 2..50, median; backprop's full forward+backward+update happens inside model(images) so it
# is captured), gaussian_noise, W6A6 ViT-B/16, bs64, seed0. Report = median over 5 reps of
# median_per_image_ms. No Adapt (6.0) and Ours (6.6) reuse runs/latency_v3 (same protocol).
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
OUT=$QUAR/runs/backprop_latency_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_backprop_latency_dir.txt"
W6A6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"
BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 4 \
  --corruption gaussian_noise --seed 0 --max_eval_batches 50 --latency_profile \
  --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP"
declare -A FL; FL[SAR]="--algorithm sar"; FL[TENT]="--algorithm tent"; FL[EATA]="--algorithm eata"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
cd "$QUAR"
for m in SAR TENT EATA; do
  for rep in 1 2 3 4 5; do
    mkdir -p "$OUT/${m}_r${rep}"
    CUDA_VISIBLE_DEVICES=0 $PY -u main.py $BASE $W6A6 ${FL[$m]} \
      --output "$OUT/${m}_r${rep}" --tag "_${m}_r${rep}" >"$OUT/${m}_r${rep}.log" 2>&1
    v=$(grep -oE "median_per_image_ms=[0-9.]+" "$OUT/${m}_r${rep}.log" | grep -oE "[0-9.]+$")
    log "[$m rep$rep] per-img median=${v:-NA} ms"
  done
done
log "=== median over 5 reps (per-image ms) ==="
for m in SAR TENT EATA; do
  vals=$(for rep in 1 2 3 4 5; do grep -oE "median_per_image_ms=[0-9.]+" "$OUT/${m}_r${rep}.log" | grep -oE "[0-9.]+$"; done | sort -n)
  med=$(echo "$vals" | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  log "$m : reps=[$(echo $vals|tr '\n' ' ')] median=$med ms"
done
log "ALL DONE: $OUT"
