#!/usr/bin/env bash
# SPA (Niu et al., ICML 2025) as a candidate row for tab:backprop_acc (Table 8).
#
# Phase 1 (parallel, one run per GPU): accuracy, W6A6 ViT-B, ImageNet-C all-15,
#   sev5, RobustBench 5k, bs=64, 3 seeds, episodic per-corruption reset. This is
#   byte-for-byte the protocol of run_backprop_tta.sh that produced the TENT /
#   EATA / SAR rows (49.77 / 50.05 / 52.79), so the numbers are comparable.
#   Two variants: 'spa' (bare self-bootstrapping) and 'spa+eata' (the strongest
#   upstream ImageNet config).
#
# Phase 2 (STRICTLY SEQUENTIAL, GPU 0, machine otherwise idle): per-image
#   latency under the run_backprop_latency.sh protocol (gaussian_noise, bs64,
#   --latency_profile, batches 2..50, median; 5 reps, median of reps) AND peak
#   CUDA memory in the same runs. Memory is also (re)measured for every other
#   Table 8 row so the column is apples-to-apples. Latency for those rows is
#   already published (No Adapt 6.0 / QuAR 6.6 / TENT 11.3 / EATA 11.4 /
#   SAR 22.2) and is only re-checked here as a contamination guard.
#
# NOTE: SPA needs ~23 GiB peak at bs=64 on this quantized graph and does NOT fit
# on a 24 GiB TITAN RTX without expandable_segments. That allocator setting is
# applied to EVERY run here, including the baselines, so no row is advantaged.
# The reported memory metric is peak *allocated* (actual tensor bytes), which is
# allocator-independent; peak reserved is logged too.
set -uo pipefail

PY="${PY:-python3}"
QUAR_DIR="${QUAR_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root (…/Data/CLS-LOC)}"
DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR_DIR/robustbench/data/imagenet_test_image_ids.txt
CMAP=$QUAR_DIR/robustbench/data/imagenet_class_to_id_map.json
PTQ=$QUAR_DIR/checkpoints/ptq4vit_w6a6_vitb.pt
OUT=$QUAR_DIR/runs/spa_table8_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
echo "$OUT" > "$QUAR_DIR/runs/last_spa_table8_dir.txt"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

W6A6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"
BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval \
  --batch_size 64 --workers 6 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP"

QUAR="--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all \
 --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 \
 --quar_local_stats_file $QUAR_DIR/stats/qac_stats_w6a6_all_layers_seed0.pt \
 --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
 --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
 --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $QUAR_DIR/stats/cls_stats_w6a6_seed0.pt \
 --quar_global_target_ema_alpha 0.05 --recal_cumulative"

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
cd "$QUAR"

# ---------------- Phase 1: accuracy, 3 seeds, 15 corruptions ----------------
acc_run(){ local gpu=$1 variant=$2 seed=$3; local tag="${variant/+/_}_s${seed}"
  mkdir -p "$OUT/$tag"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u main.py $BASE $W6A6 \
    --corruption all --seed "$seed" --algorithm "$variant" --mem_profile \
    --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 \
    && log "[$tag] done" || log "[$tag] FAILED (exit $?)"; }

log "Phase 1: accuracy. W6A6 | all-15 | 5k | bs64 | 3 seeds | episodic"
log "  GPU0-2 = spa seeds 0/1/2 ; GPU3-5 = spa+eata seeds 0/1/2"
g=0
for s in 0 1 2; do acc_run $g spa      $s & g=$((g+1)); done
for s in 0 1 2; do acc_run $g spa+eata $s & g=$((g+1)); done
wait
log "Phase 1 complete."

# ---------------- Phase 2: latency + memory, sequential, GPU 0 ----------------
# Guard against contamination: nothing else of ours may be on the GPUs.
while pgrep -f "main\.py .*--tag _spa" >/dev/null 2>&1; do sleep 20; done
sleep 30
log "Phase 2: latency+memory, sequential on GPU 0 (machine idle)"

prof_run(){ local tag=$1 rep=$2; shift 2; mkdir -p "$OUT/prof_${tag}_r${rep}"
  CUDA_VISIBLE_DEVICES=0 $PY -u main.py $BASE $W6A6 \
    --corruption gaussian_noise --seed 0 --max_eval_batches 50 \
    --latency_profile --mem_profile "$@" \
    --output "$OUT/prof_${tag}_r${rep}" --tag "_prof_${tag}_r${rep}" \
    >"$OUT/prof_${tag}_r${rep}.log" 2>&1
  local v m
  v=$(grep -oE "median_per_image_ms=[0-9.]+" "$OUT/prof_${tag}_r${rep}.log" | grep -oE "[0-9.]+$")
  m=$(grep -oE "peak_allocated_MiB=[0-9.]+" "$OUT/prof_${tag}_r${rep}.log" | grep -oE "[0-9.]+$" | tail -1)
  log "  [$tag rep$rep] per-img=${v:-NA} ms  peak_alloc=${m:-NA} MiB"; }

# 5 reps for the new SPA rows (latency is the headline claim there).
for rep in 1 2 3 4 5; do prof_run SPA      $rep --algorithm spa;      done
for rep in 1 2 3 4 5; do prof_run SPAeata  $rep --algorithm spa+eata; done
# 1 rep for the existing rows: their latency is already published, we need the
# memory column and a sanity check that latency reproduces.
prof_run NoAdapt 1 --algorithm no_adapt
prof_run QuAR    1 $QUAR
prof_run TENT    1 --algorithm tent
prof_run EATA    1 --algorithm eata
prof_run SAR     1 --algorithm sar

# ---------------- summary ----------------
log "===== Phase 1: mean top-1 over 15 corruptions (3 seeds) ====="
for variant in spa spa_eata; do
  vals=()
  for s in 0 1 2; do
    v=$(grep -oE "mean acc of corruption: [0-9.]+" "$OUT/${variant}_s${s}.log" 2>/dev/null \
        | tail -1 | grep -oE "[0-9.]+$")
    [[ -n "$v" ]] && vals+=("$v")
  done
  if [[ ${#vals[@]} -eq 3 ]]; then
    stat=$(printf '%s\n' "${vals[@]}" | awk '{a[NR]=$1;s+=$1} END{m=s/NR; \
      for(i=1;i<=NR;i++)d+=(a[i]-m)^2; printf "%.2f +- %.2f", m, sqrt(d/(NR-1))}')
    log "  $(printf '%-9s' $variant) = $stat   (seeds: ${vals[*]})"
  else
    log "  $(printf '%-9s' $variant) = INCOMPLETE(${#vals[@]}/3)"
  fi
done
log "  tab:backprop_acc reference: NoAdapt 47.69 | SAR 49.77 | TENT 50.05 | EATA 52.79 | QuAR 54.35"

log "===== Phase 2: per-image latency (median of reps) and peak memory ====="
for tag in NoAdapt QuAR TENT EATA SAR SPA SPAeata; do
  lv=$(for r in 1 2 3 4 5; do grep -oE "median_per_image_ms=[0-9.]+" "$OUT/prof_${tag}_r${r}.log" 2>/dev/null \
        | grep -oE "[0-9.]+$"; done | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
  mv=$(for r in 1 2 3 4 5; do grep -oE "peak_allocated_MiB=[0-9.]+" "$OUT/prof_${tag}_r${r}.log" 2>/dev/null \
        | grep -oE "[0-9.]+$" | tail -1; done | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
  log "  $(printf '%-8s' $tag) latency=${lv:-NA} ms   peak_allocated=${mv:-NA} MiB"
done
log "ALL DONE: $OUT"
