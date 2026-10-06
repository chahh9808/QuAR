#!/usr/bin/env bash
# DeiT-B backbone column with per-backbone input normalization.
#
# DeiT-B expects ImageNet mean/std while the augreg ViT-B/ViT-S expect 0.5; the evaluation loader
# follows each backbone's own timm transform, matching the PTQ calibration loader, so calibration
# and evaluation agree for every backbone.
#
# The PTQ checkpoint is reused: it was calibrated through timm's own transform, so it is the one
# asset that was already correct. Every source-statistic file has to be rebuilt because those went
# through the evaluation loader.
#
# Stage 1  rebuild ZOA / FOA / FOZO source statistics (one cheap single-corruption job each)
# Stage 2  7 methods x 3 seeds at W6A6 on ImageNet-C 5k, all 15 corruptions
# Usage: bash scripts/run_deitb_renorm.sh
set -uo pipefail
PY="${PY:-python}"; QUAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$QUAR"
D=/path/to/data
DATA=$D/imagenet/Data/CLS-LOC; DATA_C=$D/imagenet-c
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_deitb_p13.pt
OUT="${OUT:-runs/deitb_renorm_run_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"; echo "$OUT" > runs/last_deitb_renorm_run_dir.txt
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; SEEDS="${SEEDS:-0 1 2}"; METHODS="${METHODS:-noadapt t3a foa zoa neo fozo quar}"
cp "$0" "$OUT/launcher.sh"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/launch.log"; }

base(){ echo "--arch deit_base_patch16_224 --data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval \
  --corruption all --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
  --batch_size 64 --workers 8 --seed $1 --quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"; }
mflags(){ case "$1" in
  noadapt) echo "--algorithm no_adapt";;
  neo)     echo "--algorithm no_adapt --neo_enable";;
  t3a)     echo "--algorithm t3a --t3a_filter_k 20";;
  foa)     echo "--algorithm foa --popsize 1 --foa_stats_dir $QUAR/foa_stats_backbones";;
  fozo)    echo "--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 --fozo_stats_file $QUAR/fozo_stats_backbones/deitb_w6a6.pt";;
  zoa)     echo "--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 \
             --zoa_layer_balance none --official_zoa_stats_file $QUAR/zoa_stats/deitb_w6a6.pt";;
  quar)    echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all \
             --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 \
             --quar_local_stats_file $QUAR/stats/qac_stats_w6a6_all_layers_seed0_deitb.pt \
             --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
             --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
             --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $QUAR/stats/cls_stats_w6a6_seed0_deitb.pt \
             --quar_global_target_ema_alpha 0.05 --recal_cumulative";;
  *) echo BADMETHOD; return 1;; esac; }

run_job(){ local gpu=$1 tag=$2 seed=$3 m=$4 extra=${5:-}
  local n; n=$(grep -c "Top-1 Accuracy:" "$OUT/$tag.log" 2>/dev/null); n=${n:-0}
  if [ "$n" -ge 15 ]; then log "[$tag] skip (done)"; return; fi
  log "[$tag] start gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u main.py $(base $seed) $(mflags $m) $extra \
    --output "$OUT/$tag" --tag "_$tag" > "$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }

# ---- stage 1: source-statistic caches that went through the evaluation loader ----
statfile(){ case "$1" in
  zoa)  echo "zoa_stats/deitb_w6a6.pt";;
  foa)  echo "foa_stats_backbones/train_info_std_deit_base_patch16_224_w6a6_p3.pt";;
  fozo) echo "fozo_stats_backbones/deitb_w6a6.pt";;
  *) echo "";; esac; }
i=0; gl=($GPUS)
for m in zoa foa fozo; do [[ " $METHODS " == *" $m "* ]] || continue
  f=$(statfile $m); [ -n "$f" ] || continue
  rm -f "$f" "${f}.meta.json"
  [ "$m" = foa ] && rm -f foa_stats_backbones/train_info_mean_deit_base_patch16_224_w6a6_p3.pt
  g=${gl[$((i % ${#gl[@]}))]}; i=$((i+1))
  ( CUDA_VISIBLE_DEVICES=$g $PY -u main.py $(base 0) $(mflags $m) --corruption brightness \
      --output "$OUT/${m}stats" --tag "_${m}stats" > "$OUT/${m}stats.log" 2>&1
    log "[${m}stats] exit=$? file=$([ -f $f ] && echo ok || echo MISSING)" ) &
done; wait

# ---- stage 2: the queue ----
Q="$OUT/queue"; : > "$Q"
for seed in $SEEDS; do for m in $METHODS; do echo "deitb_w6a6_${m}_s${seed} $seed $m" >> "$Q"; done; done
log "queue: $(wc -l < $Q) jobs on GPUs [$GPUS] -> $OUT"
worker(){ local gpu=$1
  while true; do
    line=$(flock "$Q.lock" bash -c "head -n1 '$Q' && sed -i '1d' '$Q'"); [ -z "$line" ] && break
    set -- $line; run_job "$gpu" "$1" "$2" "$3"
  done; }
for g in $GPUS; do worker "$g" & sleep 2; done; wait
log "ALL_DONE"
