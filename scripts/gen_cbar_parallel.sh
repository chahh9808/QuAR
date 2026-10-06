#!/usr/bin/env bash
# Generate ImageNet-C-bar at LEVEL 5 ONLY (highest severity per corruption), one process per
# corruption in parallel. 5k subset, severity remapped to level 5 later by remap_cbar_levels.py.
# 10 corruptions x 5000 imgs = 50k images. Wall-clock ~= slowest single corruption.
set -uo pipefail
PY="${PY:-python3}"
ICB=$QUAR/external_repos/augmentation-corruption/imagenet_c_bar
SRCROOT="${DATA_CBAR_SRC:?set DATA_CBAR_SRC to the imagenet-c-bar-src root}"
OUT="${DATA_CBAR_GEN:?set DATA_CBAR_GEN to the imagenet-c-bar-gen root}"
LOGDIR=$QUAR/runs/cbar_gen_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR" "$OUT/ImageNet-C-Bar"   # pre-create shared dir to avoid mkdir race
echo "$LOGDIR" > $QUAR/runs/last_cbar_genlog_dir.txt
cd "$ICB"
echo "[gen] logdir=$LOGDIR"
for csv in single/*.csv; do
  corr=$(basename "$csv" .csv)
  $PY -u tools/make_imagenet_c_bar.py \
    --imagenet_dir "$SRCROOT" --out_dir "$OUT" \
    --corruption_file "$csv" --num_workers 7 --batch_size 100 --seed 0 \
    > "$LOGDIR/$corr.log" 2>&1 &
  echo "[gen] launched $corr (pid $!)"
done
wait
echo "[gen] ALL 10 corruptions done"
touch "$LOGDIR/_DONE"
