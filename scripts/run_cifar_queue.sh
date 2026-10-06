#!/usr/bin/env bash
# CIFAR-10-C / CIFAR-100-C extension (non-ImageNet benchmark), quantized ViT-B/16 fine-tuned on CIFAR.
# Methods: No Adapt, NEO, ZOA, QuAR (the three competitive rows + reference). W8A8/W6A6 (PTQ4ViT) and W4A4/W3A3 (AdaLog,
# checkpoints from tools/stats/build_adalog_ckpt.py), severity 5, all 15 corruptions, 10k test images each, bs 64, episodic, seeds 0/1/2.
# QuAR = the paper's fixed configuration (lambda 0.5, EMA 0.05, blocks 8-11 x 4 modules, cumulative
# estimator) with CIFAR source statistics from scripts/build_cifar_stats.sh; nothing re-tuned.
# ZOA/NEO flags = scripts/run_zoa_table1_refix.sh / run_full50k_table1.sh. ZOA source stats come from
# the CIFAR TRAIN split (--source_split train) and are pre-built once per (dataset, bit-width) in phase 0.
# Work queue: one worker per GPU pulls the next job (flock); completed jobs (log has 'mean acc') are skipped.
# Usage: OUT=runs/cifar_<ts> GPUS="0 1 2 3 4 5 6 7" SEEDS="0 1 2" BWS="6 8" bash scripts/run_cifar_queue.sh   (BWS="4 3" for the AdaLog pair)
#        DSS=domainnet126 for DomainNet-126 (real -> clipart/painting/sketch, 3 target domains instead of 15 corruptions)
set -uo pipefail
PY="${PY:-python}"; QUAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$QUAR"
D="${DATA_ROOT:-/path/to/data}"
DATA_CIFAR10="${DATA_CIFAR10:-$D/cifar10}";       DATA_CIFAR10_C="${DATA_CIFAR10_C:-$D/cifar10-c}"
DATA_CIFAR100="${DATA_CIFAR100:-$D/cifar100}";     DATA_CIFAR100_C="${DATA_CIFAR100_C:-$D/cifar100-c}"
DATA_DN126="${DATA_DN126:-$D/domainnet126/real_src}"; DATA_DN126_C="${DATA_DN126_C:-$D/domainnet126/targets}"   # source domain (real) / target domains
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; SEEDS="${SEEDS:-0 1 2}"; BWS="${BWS:-6 8}";
OUT="${OUT:-runs/cifar_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"; echo "$OUT" > runs/last_cifar_dir.txt; echo "$OUT" > "runs/last_$(echo $DSS | tr -d " ")_bw$(echo $BWS | tr -d " ")_dir.txt"
 DSS="${DSS:-cifar10 cifar100}"; METHODS="${METHODS:-zoa noadapt neo quar}"
cp "$0" "$OUT/launcher.sh"
declare -A NC=( [cifar10]=10 [cifar100]=100 [domainnet126]=126 )
declare -A CK=( [cifar10]=checkpoints/vit_base_patch16_224_cifar10_nateraw_timm.pth [cifar100]=checkpoints/vit_base_patch16_224_cifar100_pkr7098_timm.pth
                [domainnet126]=checkpoints/vit_base_patch16_224_domainnet126_real_timm.pth )
# DomainNet-126 (AdaContrast lists): source = real (train split), targets = clipart/painting/sketch laid out by
# tools/prep/build_domainnet126_imagefolder.py as data_corruption/dn_<domain>/5/<class>/.
declare -A DDATA=( [cifar10]=$DATA_CIFAR10 [cifar100]=$DATA_CIFAR100 [domainnet126]=$DATA_DN126 )
declare -A DCORR=( [cifar10]=$DATA_CIFAR10_C [cifar100]=$DATA_CIFAR100_C [domainnet126]=$DATA_DN126_C )
declare -A CORRS=( [cifar10]=all [cifar100]=all [domainnet126]=dn_clipart,dn_painting,dn_sketch )
declare -A NCORR=( [cifar10]=15 [cifar100]=15 [domainnet126]=3 )
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/launch.log"; }

base(){ local ds=$1 bw=$2 seed=$3
  echo "--dataset $ds --num_classes ${NC[$ds]} --arch vit_base_patch16_224 --arch_checkpoint ${CK[$ds]} \
   --data ${DDATA[$ds]} --data_corruption ${DCORR[$ds]} --level 5 --corruption ${CORRS[$ds]} --batch_size 64 --workers 8 --seed $seed \
   --quant --quant_w_bit $bw --quant_a_bit $bw $(qckpt $ds $bw) --source_split train"; }
qckpt(){ local ds=$1 bw=$2; case "$bw" in
  8|6) echo "--ptq_checkpoint checkpoints/ptq4vit_w${bw}a${bw}_$ds.pt";;
  4|3) echo "--adalog_checkpoint checkpoints/w${bw}a${bw}_checkpoint_opt_$ds.pt";;
  *) echo BADBW; return 1;; esac; }
mflags(){ local m=$1 ds=$2 bw=$3; case "$m" in
  noadapt) echo "--algorithm no_adapt";;
  neo)     echo "--algorithm no_adapt --neo_enable";;
  zoa)     echo "--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 --zoa_layer_balance none \
             --official_zoa_stats_file $QUAR/zoa_stats/${ds}_w${bw}a${bw}.pt";;
  quar)    echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
             --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QUAR/stats/qac_stats_w${bw}a${bw}_all_layers_${ds}_seed0.pt \
             --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
             --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
             --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $QUAR/stats/cls_stats_w${bw}a${bw}_${ds}_seed0.pt \
             --quar_global_target_ema_alpha 0.05 --recal_cumulative";;
  *) echo BADMETHOD; return 1;; esac; }

run_job(){ local gpu=$1 tag=$2 ds=$3 bw=$4 seed=$5 m=$6 extra=${7:-}
  # complete = one Top-1 line per corruption (15); "mean acc of corruption" is a running mean printed after each corruption
  if [ "$(grep -c "Top-1 Accuracy:" "$OUT/$tag.log" 2>/dev/null)" -ge "${NCORR[$ds]}" ]; then log "[$tag] skip (done)"; return; fi
  log "[$tag] start gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u main.py $(base $ds $bw $seed) $(mflags $m $ds $bw) $extra \
    --output "$OUT/$tag" --tag "_$tag" > "$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }

# ---- phase 0: ZOA source-stat files, one cheap job per (ds, bw) so the seeds never race on the cache ----
if [[ " $METHODS " == *" zoa "* ]]; then
  i=0; gl=($GPUS)
  for ds in $DSS; do for bw in $BWS; do
    f=zoa_stats/${ds}_w${bw}a${bw}.pt
    if [ ! -f "$f" ]; then
      g=${gl[$((i % ${#gl[@]}))]}; i=$((i+1))
      c0=${CORRS[$ds]}; [ "$c0" = all ] && c0=brightness || c0=${c0%%,*}
      ( CUDA_VISIBLE_DEVICES=$g $PY -u main.py $(base $ds $bw 0) $(mflags zoa $ds $bw) --corruption $c0 \
          --output "$OUT/zoastats_${ds}_w${bw}a${bw}" --tag "_zoastats" > "$OUT/zoastats_${ds}_w${bw}a${bw}.log" 2>&1; log "[zoastats $ds w${bw}a${bw}] exit=$? file=$([ -f $f ] && echo ok || echo MISSING)" ) &
    fi
  done; done; wait
fi

# ---- phase 1: the queue (seed 0 first so a first read is available early) ----
Q="$OUT/queue"; : > "$Q"
for seed in $SEEDS; do for ds in $DSS; do for bw in $BWS; do for m in $METHODS; do
  echo "${ds}_w${bw}a${bw}_${m}_s${seed} $ds $bw $seed $m" >> "$Q"
done; done; done; done
log "queue: $(wc -l < $Q) jobs on GPUs [$GPUS] -> $OUT"
worker(){ local gpu=$1
  while true; do
    line=$(flock "$Q.lock" bash -c "head -n1 '$Q' && sed -i '1d' '$Q'"); [ -z "$line" ] && break
    set -- $line; run_job "$gpu" "$1" "$2" "$3" "$4" "$5"
  done; }
for g in $GPUS; do worker "$g" & sleep 2; done; wait
log "ALL_DONE"
