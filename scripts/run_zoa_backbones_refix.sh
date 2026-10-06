#!/usr/bin/env bash
# ZOA cells of tab:backbones: ViT-B / DeiT-B / ViT-S at W6A6, 3 seeds, under
# ZOA's paper LayerNorm set, the same configuration as every other ZOA number.
#
# Flags verbatim from run_p13_3seed_fp.sh: per-backbone PTQ4ViT full-net
# checkpoint and arch, per-backbone ZOA source statistics, ImageNet-C severity 5,
# all 15 corruptions, RobustBench 5k, batch 64, episodic.
set -uo pipefail
PY="${PY:-python3}"; F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$F/robustbench/data/imagenet_test_image_ids.txt; CMAP=$F/robustbench/data/imagenet_class_to_id_map.json
P13=$F/checkpoints
OUT=$F/runs/zoa_backbones_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$F/runs/last_zoa_backbones.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
arch(){ case "$1" in vitb) echo vit_base_patch16_224;; deitb) echo deit_base_patch16_224;; vits) echo vit_small_patch16_224;; esac; }
ZOA="--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 \
--zoa_layer_balance none"
cd "$F"
i=0
for tag in vitb deitb vits; do for s in 0 1 2; do
  t="${tag}_s${s}"; g=$((i % 8)); i=$((i+1)); mkdir -p "$OUT/$t"
  ( log "[$t g$g] start"
    CUDA_VISIBLE_DEVICES=$g $PY -u "$F/main.py" \
      --data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 3 \
      --corruption all --seed "$s" --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
      --quant --quant_w_bit 6 --quant_a_bit 6 \
      --ptq_checkpoint "$P13/ptq4vit_w6a6_${tag}_p13.pt" --arch "$(arch $tag)" \
      $ZOA --official_zoa_stats_file "$F/zoa_stats/${tag}_w6a6.pt" \
      --output "$OUT/$t" --tag "_$t" >"$OUT/$t.log" 2>&1 \
      && log "[$t] done" || log "[$t] FAIL exit=$?" ) &
done; done
wait; log "ZOA_BACKBONES_DONE $OUT"
for tag in vitb deitb vits; do
  vs=""
  for s in 0 1 2; do
    a=$(find "$OUT/${tag}_s${s}" -name "*acc=*" 2>/dev/null | grep -oE "acc=[0-9.]+" | head -1 | cut -d= -f2)
    vs="$vs ${a:-NA}"
  done
  echo "$tag: $vs"
done
