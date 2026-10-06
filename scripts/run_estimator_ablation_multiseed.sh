#!/usr/bin/env bash
# Estimator ablation, FULL factorial: 6 configs x {bs64,bs1} x seed{0,1,2} = 36 runs.
# configs (local est, global est): baseline(cur,cur) V1(welf,welf) V2(cur,ema) V3(welf,ema)
#                                   V4(welf,cur) V5(cur,welf)
# W6A6 ImageNet-C 5k, 15 corruptions, episodic. Resumable (skip 15-line logs).
# seed0's original 4 configs are copied in from the prior run so they are skipped.
set -u
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python3}"
RUNDIR="$QUAR/runs/estimator_ablation_ms_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUNDIR"; echo "$RUNDIR" > "$QUAR/runs/last_estimator_ablation_ms_dir.txt"
cd "$QUAR"

PTQ=checkpoints/ptq4vit_w6a6_vitb.pt
BASE0="--data ${DATA:?set DATA to the ImageNet root} \
--data_corruption ${DATA_C:?set DATA_C to the imagenet-c root} \
--level 5 --corruption all --workers 4 --imagenet_5k_eval"
QUANT="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"
LGAR="--algorithm no_adapt \
--quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
--quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 \
--quar_local_stats_file qac_stats_w6a6_all_layers_seed0.pt \
--quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
--recal_cumulative \
--quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std \
--quar_global_alpha 1.0 --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 \
--quar_global_stats_file cls_stats_w6a6_seed0.pt --quar_global_target_ema_alpha 0.05"

est_flags(){ case "$1" in
  baseline) echo "";;
  V1) echo "--quar_local_estimator welford --quar_global_estimator welford";;
  V2) echo "--quar_global_estimator ema";;
  V3) echo "--quar_local_estimator welford --quar_global_estimator ema";;
  V4) echo "--quar_local_estimator welford";;
  V5) echo "--quar_global_estimator welford";;
esac; }

# reuse prior seed0 runs (original 4 configs) so they are skipped
OLD=$QUAR/runs/estimator_ablation_20260703_174104
for cfg in baseline V1 V2 V3; do for bs in 64 1; do
  s="$OLD/${cfg}_bs${bs}.log"
  [ -f "$s" ] && [ "$(grep -c 'Top-1 Accuracy' "$s" 2>/dev/null)" -eq 15 ] && cp "$s" "$RUNDIR/${cfg}_bs${bs}_s0.log"
done; done

is_done(){ local f=$1; [ -f "$f" ] && [ "$(grep -c 'Top-1 Accuracy' "$f" 2>/dev/null)" -eq 15 ]; }
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUNDIR/driver.log"; }
GPUS=(0 1 2 3 4 5 6 7); MAXJOBS=8; gi=0

# queue: bs1 first (slow), then bs64
QUEUE=()
for seed in 0 1 2; do for cfg in baseline V1 V2 V3 V4 V5; do QUEUE+=("$cfg 1 $seed"); done; done
for seed in 0 1 2; do for cfg in baseline V1 V2 V3 V4 V5; do QUEUE+=("$cfg 64 $seed"); done; done

run_one(){
  local cfg=$1 bs=$2 seed=$3 gpu=$4
  local tag="${cfg}_bs${bs}_s${seed}"; local logf="$RUNDIR/$tag.log"
  if is_done "$logf"; then log "[$tag g$gpu] SKIP"; return 0; fi
  mkdir -p "$RUNDIR/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u "$QUAR/main.py" $BASE0 --seed $seed --batch_size $bs $QUANT $LGAR $(est_flags "$cfg") \
    --output "$RUNDIR/$tag" --tag "_$tag" >"$logf" 2>&1 \
    && log "[$tag g$gpu] done" || log "[$tag g$gpu] FAIL exit=$?"
}

log "=== estimator ablation (multiseed) start: ${#QUEUE[@]} jobs (8 seed0 copied+skipped), maxjobs=$MAXJOBS ==="
for item in "${QUEUE[@]}"; do
  set -- $item; cfg=$1; bs=$2; seed=$3
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  gpu=${GPUS[$((gi % ${#GPUS[@]}))]}; gi=$((gi+1))
  run_one "$cfg" "$bs" "$seed" "$gpu" &
  sleep 2
done
wait
log "=== estimator ablation (multiseed) ALL DONE: $RUNDIR ==="
