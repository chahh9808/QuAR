#!/usr/bin/env bash
# tab:backbones FOA/FOZO cells, run one corruption at a time on the healthy NUMA node.
#
# Why this exists. The host's node-0 DIMMs (P1-DIMMA1, P1-DIMMA2) are failing: corrected
# ECCs accumulate under load, the kernel has poisoned several pages, and the machine has
# now gone down twice. Three changes make this survivable.
#
#  1. Memory is kept off the bad DIMMs. Every failure so far reported "node: 0", and the
#     box has two nodes, so the job is pinned to node-1 CPUs (18-35,54-71). Linux allocates
#     on the faulting CPU's local node by default, so pages land on node 1. All GPUs sit on
#     node 0, so transfers cross the interconnect; this job is NFS-bound, so that is cheap
#     next to another crash.
#  2. One corruption per process. The paper's protocol is episodic, and a single-corruption
#     run reproduces the corresponding entry of an all-15 run exactly (FOZO/ViT-S
#     gaussian_noise: 16.28 standalone, 16.279998779296875 as the first entry of the all-15
#     run). A crash therefore costs about a minute, not a 25-minute cell.
#  3. One job at a time. A single job completed a full cell here; six concurrent jobs took
#     the machine down in four minutes.
#
# Per-corruption results land in cells/<cell>/<corruption>.acc, and a cell's mean is only
# written once all 15 exist, so a partially finished cell can never be mistaken for a result.
set -uo pipefail
F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python3}"
DATA="${DATA:?set DATA to the ImageNet root (…/Data/CLS-LOC)}"
DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$F/robustbench/data/imagenet_test_image_ids.txt
CMAP=$F/robustbench/data/imagenet_class_to_id_map.json
P13=$F/checkpoints
OUT=$F/runs/backbones_safe; mkdir -p "$OUT/cells"
GPU="${GPU:-0}"
NODE1_CPUS="${NODE1_CPUS:-$(cat /sys/devices/system/node/node1/cpulist)}"
GUARD="${GUARD:-}"; HWBASE="${HWBASE:-}"   # optional hardware-health hook: a script that exits non-zero to stop the run
LOG="$OUT/driver.log"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

CORR=(gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur \
      snow frost fog brightness contrast elastic_transform pixelate jpeg_compression)
arch(){ case "$1" in vitb) echo vit_base_patch16_224;; deitb) echo deit_base_patch16_224;;
                     vits) echo vit_small_patch16_224;; esac; }

say "start: gpu $GPU, memory pinned to node-1 cpus $NODE1_CPUS, one corruption per process"

for s in 0 1 2; do for meth in FOA FOZO; do for tag in vitb deitb vits; do
  cell="${meth}_${tag}_s${s}"; mkdir -p "$OUT/cells/$cell"
  [ -f "$OUT/cells/$cell/MEAN" ] && { say "[$cell] complete, skipping"; continue; }

  if [ "$meth" = FOA ]; then
    extra="--algorithm foa --popsize 1 --foa_stats_dir $F/foa_stats_backbones"
  else
    extra="--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 --fozo_stats_file $F/fozo_stats_backbones/${tag}_w6a6.pt"
  fi

  for c in "${CORR[@]}"; do
    acc_f="$OUT/cells/$cell/${c}.acc"
    [ -s "$acc_f" ] && continue

    # refuse to start anything if the hardware has moved since the baseline
    if [ -n "$GUARD" ] && [ -x "$GUARD" ]; then g=$("$GUARD" "$HWBASE" 2>&1) || { say "[$cell/$c] NOT STARTED -- $g"; say "HARDWARE STOP"; exit 9; }; [ -n "$g" ] && say "   $g"; fi

    d="$OUT/cells/$cell/$c"; mkdir -p "$d"
    CUDA_VISIBLE_DEVICES=$GPU taskset -c "$NODE1_CPUS" $PY -u "$F/main.py" \
      --data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval \
      --batch_size 64 --workers 3 --corruption "$c" --seed "$s" \
      --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
      --quant --quant_w_bit 6 --quant_a_bit 6 \
      --ptq_checkpoint "$P13/ptq4vit_w6a6_${tag}_p13.pt" --arch "$(arch $tag)" \
      $extra --output "$d" --tag "_${cell}_${c}" \
      > "$d/run.log" 2>&1
    rc=$?
    a=$(find "$d" -name "*acc=*" 2>/dev/null | grep -oE "acc=[0-9.]+" | head -1 | cut -d= -f2)
    if [ -n "$a" ]; then
      echo "$a" > "$acc_f"; say "[$cell/$c] $a"
    else
      say "[$cell/$c] FAILED rc=$rc, no accuracy written"
    fi
  done

  n=$(ls "$OUT/cells/$cell"/*.acc 2>/dev/null | wc -l)
  if [ "$n" -eq 15 ]; then
    m=$(cat "$OUT/cells/$cell"/*.acc | awk '{t+=$1; n++} END{printf "%.4f", t/n}')
    echo "$m" > "$OUT/cells/$cell/MEAN"; say "[$cell] COMPLETE mean=$m over 15 corruptions"
  else
    say "[$cell] incomplete: $n/15 corruptions, no mean written"
  fi
done; done; done

say "all cells attempted"
{ echo; printf "%-6s %-6s %8s %8s %8s\n" method backbone seed0 seed1 seed2; } | tee -a "$LOG"
for meth in FOA FOZO; do for tag in vitb deitb vits; do
  r=""; for s in 0 1 2; do
    v=$(cat "$OUT/cells/${meth}_${tag}_s${s}/MEAN" 2>/dev/null); r="$r $(printf '%8s' "${v:-NA}")"
  done
  printf "%-6s %-6s%s\n" "$meth" "$tag" "$r" | tee -a "$OUT/results.txt"
done; done
