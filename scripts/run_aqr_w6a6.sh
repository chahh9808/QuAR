#!/usr/bin/env bash
# AQR baseline (Adaptive Quantile Recalibration, Mehrbod et al., WACV 2026) vs QuAR.
# W6A6 PTQ4ViT, ImageNet-C 15-corruption level-5 5k seed0 episodic, SAME harness /
# checkpoint / eval subset as run_sens_extra_w6a6.sh (QuAR).  Cells:
#   aqr_bs64  / aqr_bs128 : AQR quantile recalibration (parity scope block8_9_10_11_all)
#   na_bs64   / na_bs128  : No-Adapt (quantized, no correction) reference
# QuAR reference (full method, QAC+CLS): bs64 = 54.07, bs128 = 53.79.
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
AQR_STATS=${AQR_STATS:-$QUAR/stats/aqr_stats_w6a6_block8_9_10_11_all_seed0.pt}
W6A6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"
OUT=$QUAR/runs/aqr_baseline/eval_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/aqr_baseline/last_eval_dir.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
# AQR flags: parity placement (same pre-quantizer sites as QuAR), pure AQR (alpha=1).
aqr(){ echo "--algorithm no_adapt --aqr_enable --aqr_stats_file $AQR_STATS \
  --aqr_scope block8_9_10_11_all --aqr_hook linear_prehook --aqr_alpha 1.0 --aqr_eps 1e-6"; }
na(){ echo "--algorithm no_adapt"; }
basef(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size $1 --workers 8 \
  --corruption all --seed 0 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP"; }
run(){ local gpu=$1 tag=$2 bs=$3; shift 3; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $(basef $bs) $W6A6 "$@" --output "$OUT/$tag" --tag "_$tag" \
    >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
cd "$QUAR"
log "OUT=$OUT | AQR stats=$AQR_STATS"
run 0 aqr_bs64  64  $(aqr) &
run 1 aqr_bs128 128 $(aqr) &
run 2 na_bs64   64  $(na)  &
run 3 na_bs128  128 $(na)  &
wait
log "=== AQR/NA W6A6 (15-corruption mean top-1) ==="
for tag in aqr_bs64 aqr_bs128 na_bs64 na_bs128; do
  m=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/$tag.log" | grep -oE "[0-9.]+$" | awk '{s+=$1;n++} END{if(n>0)printf "%.4f (n=%d)",s/n,n; else print "NA"}')
  log "$tag = $m"
done
log "ALL DONE: $OUT   (QuAR ref: bs64=54.07, bs128=53.79)"
