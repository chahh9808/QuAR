#!/usr/bin/env bash
# SPA under the most extreme online label shift of the tab:noniid / fig:noniid campaign.
# ir = 5e5 is ~pure class order: each batch is dominated by a single class.
#
# Why: bare SPA has NO entropy threshold and NO EATA redundancy filter (those live only in
# spa+eta / spa+eata, tta_library/spa/official.py:105,110-118). Its objective is two KL
# self-consistency terms between a clean view and two perturbed views. Nothing in it rejects
# samples on a class-correlated stream, and no SPA run on a label-shifted stream exists in
# this repo. This measures it.
#
# Protocol: byte-for-byte run_noniid_3seed_w6a6.sh, W6A6, FULL ImageNet-C (50k/corruption,
# no 5k subset), bs64, all 15 corruptions, --stream_order sar --sar_imbalance_ratio 500000
# --sar_T 50000, --seed s with --noniid_seed 2023+s, seeds 0/1/2.
#
# Absolute accuracy at this ratio for every published method (fig:noniid ABS + paired DEL):
#   T3A 47.24 | No Adapt 47.55 | FOZO 48.92 | FOA 49.85 | NEO 51.24 | ZOA 52.05 | QuAR 54.72
# SPA elsewhere: episodic 5k 59.14, continual 5k 59.56.
#
# WATCH FOR NaN: official.py:142 keeps only samples where the clean branch is more confident
# than the predictor branch, and KLDivLoss(reduction='batchmean') on an empty selection is
# NaN with no guard. A class-correlated stream is where that is most likely to fire. If it
# does, it is an unguarded edge case in the released code, not by itself a method-level
# result, and must be reported that way.
#
# Cost: 50k x 15 = 750k images per seed; SPA runs ~31 ms/image here, so roughly 8 h per seed.
# One seed per GPU: SPA peaks near 23 GiB at bs64 and does not fit two to a 24 GiB card.
set -uo pipefail
PY="${PY:-python}"
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$QUAR"
D=/path/to/data
DATA="${DATA:-$D/imagenet/Data/CLS-LOC}"; DATA_C="${DATA_C:-$D/imagenet-c}"
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
OUT=$QUAR/runs/spa_noniid_ir500k_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
echo "$OUT" > "$QUAR/runs/last_spa_noniid_dir.txt"; cp "$0" "$OUT/launcher.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
log "SPA non-iid ir=5e5, W6A6, full 50k/corruption, bs64, seeds 0/1/2 on GPUs 0/1/2"
for s in 0 1 2; do
  tag="spa_ir500000_s${s}"; mkdir -p "$OUT/$tag"
  CUDA_VISIBLE_DEVICES=$s $PY -u main.py \
    --data "$DATA" --data_corruption "$DATA_C" --level 5 --batch_size 64 --workers 8 \
    --corruption all --seed "$s" \
    --quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint "$PTQ" \
    --stream_order sar --sar_imbalance_ratio 500000 --sar_T 50000 --noniid_seed $((2023+s)) \
    --algorithm spa --mem_profile --output "$OUT/$tag" --tag "_$tag" \
    > "$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAILED exit=$?" &
done
wait; log "ALL DONE: $OUT"
