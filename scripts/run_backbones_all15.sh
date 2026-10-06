#!/usr/bin/env bash
# The six FOA/FOZO backbone cells, ONE process per cell covering all
# fifteen corruptions, which is how every other row of tab:backbones was measured.
#
# The per-corruption campaign in runs/backbones_safe/ is protocol-faithful (episodic) but is
# not bit-identical to a single all-fifteen process: once the statistics cache is loaded
# rather than computed the RNG stream differs, and FOA's seed spread came out 0.28 against
# 0.05 in tab:main. Measuring the same way as the rest of the table removes that asymmetry.
#
# Same safety as the per-corruption run: memory pinned to node-1 CPUs, away from the failing
# node-0 DIMMs, one job at a time, guard checked before each cell.
set -uo pipefail
F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python3}"
DATA="${DATA:?set DATA to the ImageNet root (…/Data/CLS-LOC)}"
DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$F/robustbench/data/imagenet_test_image_ids.txt
CMAP=$F/robustbench/data/imagenet_class_to_id_map.json
P13=$F/checkpoints
OUT=$F/runs/backbones_all15; mkdir -p "$OUT"
GPU="${GPU:-0}"
NODE1_CPUS="$(cat /sys/devices/system/node/node1/cpulist)"
GUARD="${GUARD:-}"; HWBASE="${HWBASE:-}"   # optional hardware-health hook: a script that exits non-zero to stop the run
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$OUT/driver.log"; }
arch(){ case "$1" in vitb) echo vit_base_patch16_224;; deitb) echo deit_base_patch16_224;;
                     vits) echo vit_small_patch16_224;; esac; }

say "start: one process per cell, all 15 corruptions, node-1 pinned, gpu $GPU"
for s in 0 1 2; do for meth in FOA FOZO; do for tag in vitb deitb vits; do
  cell="${meth}_${tag}_s${s}"; mkdir -p "$OUT/$cell"
  find "$OUT/$cell" -name "*acc=*" 2>/dev/null | grep -q . && { say "[$cell] done, skipping"; continue; }
  if [ -n "$GUARD" ] && [ -x "$GUARD" ]; then g=$("$GUARD" "$HWBASE" 2>&1) || { say "[$cell] NOT STARTED -- $g"; say "HARDWARE STOP"; exit 9; }; [ -n "$g" ] && say "   $g"; fi
  if [ "$meth" = FOA ]; then
    extra="--algorithm foa --popsize 1 --foa_stats_dir $F/foa_stats_backbones"
  else
    extra="--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 --fozo_stats_file $F/fozo_stats_backbones/${tag}_w6a6.pt"
  fi
  say "[$cell] start"
  CUDA_VISIBLE_DEVICES=$GPU taskset -c "$NODE1_CPUS" $PY -u "$F/main.py" \
    --data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval \
    --batch_size 64 --workers 3 --corruption all --seed "$s" \
    --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
    --quant --quant_w_bit 6 --quant_a_bit 6 \
    --ptq_checkpoint "$P13/ptq4vit_w6a6_${tag}_p13.pt" --arch "$(arch $tag)" \
    $extra --output "$OUT/$cell" --tag "_$cell" > "$OUT/$cell.log" 2>&1
  a=$(find "$OUT/$cell" -name "*acc=*" 2>/dev/null | grep -oE "acc=[0-9.]+" | head -1 | cut -d= -f2)
  say "[$cell] ${a:-INCOMPLETE}"
done; done; done
say "done"
for meth in FOA FOZO; do for tag in vitb deitb vits; do
  r=""; for s in 0 1 2; do
    v=$(find "$OUT/${meth}_${tag}_s${s}" -name "*acc=*" 2>/dev/null | grep -oE "acc=[0-9.]+" | head -1 | cut -d= -f2)
    r="$r $(printf '%8s' "${v:-NA}")"
  done
  printf "%-6s %-6s%s\n" "$meth" "$tag" "$r" | tee -a "$OUT/results.txt"
done; done
