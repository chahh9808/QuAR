#!/usr/bin/env bash
# NEO-Continual (official EMA variant, alpha=0.1) on the CONTINUAL table protocol (tab:t3):
# 10-round x 15-corruption continual stream (state carried, reset once at start), 5k subset, bs64,
# seeds 0/1/2, W3A3 & W6A6. This is the FAIR NEO baseline for the continual table (base NEO is
# "unsuitable when the shift changes" per the NEO paper; NEO-Continual is the prescribed variant).
# Context (same protocol, existing): Ours(EMA) W3A3 35.35 / W6A6 53.70 ; ZOA 29.37 / 52.31 ; NoAdapt 28.35 / 47.69.
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
ADALOG_W3A3=$QUAR/checkpoints/w3a3_checkpoint_opt.pt
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt
CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
OUT=$QUAR/runs/neo_continual_3seed_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"; echo "$OUT" > "$QUAR/runs/last_neo_cont3_dir.txt"
quant_flags(){ case "$1" in
  w3a3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $ADALOG_W3A3";;
  w6a6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6";;
esac; }
base(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval \
  --batch_size 64 --workers 8 --corruption all --seed $1 --corruption_repeat 10 --continual \
  --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $(quant_flags $2)"; }
# NEO-Continual: official EMA variant, default alpha=0.1 (neo_layers.NEO_Continual)
neo_cont(){ echo "--algorithm no_adapt --neo_enable --neo_ema_alpha 0.1"; }
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
run(){ local gpu=$1 tag=$2 seed=$3 bw=$4; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $(base $seed $bw) $(neo_cont) --output "$OUT/$tag" --tag "_$tag" \
    >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
cd "$QUAR"
i=0
for bw in w3a3 w6a6; do for s in 0 1 2; do
  run $(( i % 6 )) "neoc_${bw}_s${s}" "$s" "$bw" &
  i=$(( i + 1 ))
done; done
wait
log "=== NEO-Continual continual (Avg over 150 segments = repeat-10) ==="
for bw in w3a3 w6a6; do for s in 0 1 2; do
  tag="neoc_${bw}_s${s}"; lg="$OUT/$tag.log"; [ -f "$lg" ] || continue
  v=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$lg" | grep -oE "[0-9.]+$" | awk '{s+=$1;n++} END{if(n>0)printf "%.2f (n=%d)",s/n,n; else print "NA"}')
  log "$tag = $v"
done; done
log "ALL DONE: $OUT  (cf. Ours-EMA W3A3 35.35 / W6A6 53.70)"
