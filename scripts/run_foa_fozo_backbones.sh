#!/usr/bin/env bash
# tab:backbones — the two missing rows, FOA and FOZO, on ViT-B / DeiT-B / ViT-S at W6A6.
#
# The table omitted them because both methods need source CLS statistics and only
# ViT-B's had ever been built. Both classes compute those statistics themselves from
# a pass over the source set, so nothing blocks the other two backbones; what was
# missing was a per-backbone cache path and one arch-aware slice in FOZO.
#
# Two things this script does NOT reuse:
#   foa_stats/    holds train_info_*_w6a6_p3.pt built against the tab:main checkpoint
#                 (runs/qac_mechanism_part3_diag_.../ptq4vit_w6a6_calibrated.pt), not the
#                 per-backbone p13_backbones checkpoints this table uses. The filename
#                 encodes arch and bitwidth but not the checkpoint, so reusing it would
#                 silently mix calibrations. Fresh dir instead.
#   train_stats_2.pt  is FOZO's single shared cache with no arch in the name. ViT-S's
#                 stats have a different length, so sharing it would clobber ViT-B's.
#
# Flags for FOA and FOZO are verbatim from run_table1_w6a6.sh (the MIN-2 forward config
# the paper reports); checkpoint/arch handling is verbatim from run_zoa_backbones_refix.sh.
#
# Usage: bash run_foa_fozo_backbones.sh [GPU_LIST]      e.g. "1" or "1,4,5"
set -uo pipefail

GPUS="${1:-1}"; IFS=',' read -ra GPU <<< "$GPUS"; NG=${#GPU[@]}
PY="${PY:-python3}"
F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root (…/Data/CLS-LOC)}"
DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$F/robustbench/data/imagenet_test_image_ids.txt
CMAP=$F/robustbench/data/imagenet_class_to_id_map.json
P13=$F/checkpoints
# Fixed root, not timestamped: a run that dies partway must be resumable, and that
# only works if the restart can see the cells that already finished. The driver log
# is timestamped instead so the history of attempts is kept.
OUT=${OUT_ROOT:-$F/runs/foa_fozo_backbones}; mkdir -p "$OUT"
echo "$OUT" > "$F/runs/last_foa_fozo_backbones.txt"
DRIVERLOG="$OUT/driver_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$F/foa_stats_backbones" "$F/fozo_stats_backbones"

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$DRIVERLOG"; }

# Hardware guard. A corrected single-bit ECC is recorded and tolerated; the campaign
# stops on a repeat on the same DIMM, or on anything uncorrected / memory-failure /
# SIGBUS. Checked before every cell and every 30s inside a cell.
GUARD="${GUARD:-}"; HWBASE="${HWBASE:-}"   # optional hardware-health hook; unset means no check
guard(){ [ -x "$GUARD" ] || return 0; "$GUARD" "$HWBASE" "${1:-}"; }
ABORTED=0
arch(){ case "$1" in vitb) echo vit_base_patch16_224;; deitb) echo deit_base_patch16_224;;
                     vits) echo vit_small_patch16_224;; esac; }

BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval \
--batch_size 64 --workers 3 --corruption all \
--imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
--quant --quant_w_bit 6 --quant_a_bit 6"

# record the exact configuration next to the results
cat > "$OUT/config.txt" <<EOF
table          : tab:backbones (FOA, FOZO rows)
setting        : W6A6 | ImageNet-C sev5 | all 15 corruptions | RobustBench 5k | bs 64 | episodic
backbones      : vitb=vit_base_patch16_224  deitb=deit_base_patch16_224  vits=vit_small_patch16_224
checkpoints    : $P13/ptq4vit_w6a6_<tag>_p13.pt
seeds          : 0 1 2
FOA            : --algorithm foa --popsize 1 --foa_stats_dir $F/foa_stats_backbones
                 (fitness_lambda default 0.4, num_prompts default 3 -> 2 forwards/batch)
FOZO           : --algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1
                 --fozo_stats_file $F/fozo_stats_backbones/<tag>_w6a6.pt
source stats   : full ImageNet val 50k (max_source_stat_batches=-1), computed on the
                 quantized model, once per (method, backbone)
code changes   : tta_library/FOZO.py, ViT-S dimension handling (FOA needed none)
                 1) _get_shift_vector sliced [-768:] -> [-prompt_dim:]  (ViT-S is 384)
                 2) _prepare_quant_padding hardcoded 12 heads -> prompt_dim//64  (ViT-S is 6)
                 both crashed rather than silently misbehaving; verified by smoke test
gpus           : $GPUS
EOF
log "config -> $OUT/config.txt"

run_one(){
  local gpu=$1 meth=$2 tag=$3 s=$4
  local t="${meth}_${tag}_s${s}"; mkdir -p "$OUT/$t"
  # resume: a cell that already wrote its accuracy file is done, do not redo it
  if find "$OUT/$t" -name "*acc=*" 2>/dev/null | grep -q .; then
    log "[$t] skip, already done"; return 0
  fi
  local extra
  if [ "$meth" = "FOA" ]; then
    extra="--algorithm foa --popsize 1 --foa_stats_dir $F/foa_stats_backbones"
  else
    extra="--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 \
--fozo_stats_file $F/fozo_stats_backbones/${tag}_w6a6.pt"
  fi
  # gate on hardware health before spending another cell
  local g; g=$(guard) || { log "[$t] NOT STARTED -- $g"; ABORTED=1; return 9; }
  [ -n "$g" ] && log "   $g"

  log "[$t g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u "$F/main.py" $BASE --seed "$s" \
    --ptq_checkpoint "$P13/ptq4vit_w6a6_${tag}_p13.pt" --arch "$(arch $tag)" \
    $extra --output "$OUT/$t" --tag "_$t" > "$OUT/$t.log" 2>&1 &
  local cpid=$!
  echo "$cpid" > "$OUT/.current_pid"
  # poll: kill this exact PID if the guard trips, never a pattern kill
  while kill -0 "$cpid" 2>/dev/null; do
    sleep 30
    g=$(guard "$OUT/$t.log") || {
      log "[$t] HARDWARE STOP -- $g"
      log "[$t] terminating PID $cpid"
      kill -TERM "$cpid" 2>/dev/null; sleep 10
      kill -0 "$cpid" 2>/dev/null && kill -KILL "$cpid" 2>/dev/null
      ABORTED=1; rm -f "$OUT/.current_pid"; return 9
    }
    [ -n "$g" ] && log "   $g"
  done
  wait "$cpid"; local rc=$?
  rm -f "$OUT/.current_pid"
  # only a cell that wrote its accuracy file counts as a usable result
  if find "$OUT/$t" -name "*acc=*" 2>/dev/null | grep -q .; then
    log "[$t] done rc=$rc"
  else
    log "[$t] INCOMPLETE rc=$rc -- no accuracy file, result discarded"
  fi
}

# Two phases, and the barrier between them is the point. Each (method, backbone) pair
# builds its source-stat cache on first use, so phase 1 runs seed 0 for all six pairs and
# they touch six distinct files. Only once every cache is on disk do seeds 1 and 2 start,
# which then only ever read. Interleaving the seeds instead would let one run torch.load a
# file another is still torch.save-ing.
# Phase 1 also has the property that a complete set of six values exists as early as
# possible, so the table can be filled before the standard deviations are in.
launch(){   # launch "$@" jobs, at most CONC in flight
  local i=0
  for j in "$@"; do
    read -r meth tag s <<< "$j"
    [ "$ABORTED" = 1 ] && { log "campaign aborted, skipping the rest"; return 9; }
    if [ "$CONC" = 1 ]; then
      run_one "${GPU[$((i % NG))]}" "$meth" "$tag" "$s"      # sequential, guard polls inline
    else
      run_one "${GPU[$((i % NG))]}" "$meth" "$tag" "$s" &
      while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 20; done
    fi
    i=$((i+1))
  done
  wait
}

CONC="${CONC:-1}"   # one job at a time by default
P1=(); for meth in FOA FOZO; do for tag in vitb deitb vits; do P1+=("$meth $tag 0"); done; done
P2=(); for s in 1 2; do for meth in FOA FOZO; do for tag in vitb deitb vits; do P2+=("$meth $tag $s"); done; done; done

log "phase 1: ${#P1[@]} runs (seed 0, builds the six stat caches), conc=$CONC, gpu(s) $GPUS"
launch "${P1[@]}" || true
[ "$ABORTED" = 1 ] && { log "STOPPED in phase 1 on a hardware trigger"; exit 9; }
log "phase 1 done; caches:"; ls -la "$F/foa_stats_backbones" "$F/fozo_stats_backbones" | tee -a "$DRIVERLOG"

log "phase 2: ${#P2[@]} runs (seeds 1,2, caches read-only), conc=$CONC"
launch "${P2[@]}" || true
[ "$ABORTED" = 1 ] && { log "STOPPED in phase 2 on a hardware trigger"; exit 9; }
log "FOA_FOZO_BACKBONES_DONE $OUT"

# collect
{ echo; echo "tag  method  seed  acc"; } | tee -a "$DRIVERLOG"
for meth in FOA FOZO; do for tag in vitb deitb vits; do
  vs=""
  for s in 0 1 2; do
    a=$(find "$OUT/${meth}_${tag}_s${s}" -name "*acc=*" 2>/dev/null \
        | grep -oE "acc=[0-9.]+" | head -1 | cut -d= -f2)
    vs="$vs ${a:-NA}"
  done
  echo "$meth $tag:$vs" | tee -a "$OUT/results.txt"
done; done
