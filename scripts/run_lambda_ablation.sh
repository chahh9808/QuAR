#!/usr/bin/env bash
# Batch-size-1 lambda/branch ablation (W6A6 ImageNet-C 5k, 15 corr, episodic, cumulative estimator).
# 5 configs x {bs64,bs1} x seed{0,1,2} = 30. full QuAR reused from the estimator ablation (copied, skipped).
#   L    = local-only (QAC)
#   Gm   = global mean-only (CLS, mean_only)
#   Gmv  = global mean+variance (CLS, mean_std) [= Final-only]
#   LGm  = local + global mean-only
#   full = full QuAR (local + global mean+variance) [deployed]
set -u
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python3}"
RUNDIR="$QUAR/runs/lambda_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUNDIR"; echo "$RUNDIR" > "$QUAR/runs/last_lambda_ablation_dir.txt"
cd "$QUAR"

PTQ=checkpoints/ptq4vit_w6a6_vitb.pt
BASE0="--data ${DATA:?set DATA to the ImageNet root} \
--data_corruption ${DATA_C:?set DATA_C to the imagenet-c root} \
--level 5 --corruption all --workers 4 --imagenet_5k_eval"
QUANT="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"
QAC="--quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
--quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 \
--quar_local_stats_file qac_stats_w6a6_all_layers_seed0.pt \
--quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag"
CLS_COMMON="--quar_global_enable --quar_global_position post_ln --quar_global_alpha 1.0 \
--quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 \
--quar_global_stats_file cls_stats_w6a6_seed0.pt --quar_global_target_ema_alpha 0.05"
CLS_MV="$CLS_COMMON --quar_global_mode mean_std"
CLS_M="$CLS_COMMON --quar_global_mode mean_only"

cfg_flags(){ case "$1" in
  L)    echo "--algorithm no_adapt $QAC --recal_cumulative";;
  Gm)   echo "--algorithm no_adapt --recal_cumulative $CLS_M";;
  Gmv)  echo "--algorithm no_adapt --recal_cumulative $CLS_MV";;
  LGm)  echo "--algorithm no_adapt $QAC --recal_cumulative $CLS_M";;
  full) echo "--algorithm no_adapt $QAC --recal_cumulative $CLS_MV";;
esac; }

# reuse full QuAR (= estimator ablation baseline) so it is skipped
EST=$QUAR/runs/estimator_ablation_ms_20260714_203037
for bs in 64 1; do for s in 0 1 2; do
  src="$EST/baseline_bs${bs}_s${s}.log"
  [ -f "$src" ] && [ "$(grep -c 'Top-1 Accuracy' "$src" 2>/dev/null)" -eq 15 ] && cp "$src" "$RUNDIR/full_bs${bs}_s${s}.log"
done; done

is_done(){ local f=$1; [ -f "$f" ] && [ "$(grep -c 'Top-1 Accuracy' "$f" 2>/dev/null)" -eq 15 ]; }
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUNDIR/driver.log"; }
GPUS=(0 1 2 3 4 5 6 7); MAXJOBS=8; gi=0

QUEUE=()
for s in 0 1 2; do for cfg in L Gm Gmv LGm full; do QUEUE+=("$cfg 1 $s"); done; done
for s in 0 1 2; do for cfg in L Gm Gmv LGm full; do QUEUE+=("$cfg 64 $s"); done; done

run_one(){
  local cfg=$1 bs=$2 s=$3 gpu=$4
  local tag="${cfg}_bs${bs}_s${s}"; local logf="$RUNDIR/$tag.log"
  if is_done "$logf"; then log "[$tag g$gpu] SKIP"; return 0; fi
  mkdir -p "$RUNDIR/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u "$QUAR/main.py" $BASE0 --seed $s --batch_size $bs $QUANT $(cfg_flags "$cfg") \
    --output "$RUNDIR/$tag" --tag "_$tag" >"$logf" 2>&1 \
    && log "[$tag g$gpu] done" || log "[$tag g$gpu] FAIL exit=$?"
}

log "=== lambda/branch ablation start: ${#QUEUE[@]} jobs (6 full copied+skipped), maxjobs=$MAXJOBS ==="
for item in "${QUEUE[@]}"; do
  set -- $item; cfg=$1; bs=$2; s=$3
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  gpu=${GPUS[$((gi % ${#GPUS[@]}))]}; gi=$((gi+1))
  run_one "$cfg" "$bs" "$s" "$gpu" &
  sleep 2
done
wait
log "=== lambda/branch ablation ALL DONE: $RUNDIR ==="
