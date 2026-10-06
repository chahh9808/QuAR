#!/usr/bin/env bash
# Figure 1(b) code-distribution drift, seeds 1 & 2 (seed0 already in experiment_results/qac_code_diag_B3/full15)
# for ALL methods + Ours-cumulative, to get a 3-seed seed-std (error bars). main.py --codedist_diag,
# W6A6, 15-corr 5k, bs64, nwarm=64 lag=False (the fig-specific codedist cfg). Configs replicate the
# seed0 codedist logs exactly; only --seed changes (and --recal_cumulative for the new ours_cum row).
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
SRC=$QUAR/stats/qac_stats_w6a6_codehist_all.pt
OUT=$QUAR/runs/codedist_seeds12_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_codedist_seeds12_dir.txt"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
W6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"
base(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 8 \
  --corruption all --seed $1 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $W6 \
  --codedist_diag --codedist_src $SRC"; }
OURS="--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
  --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QUAR/stats/qac_stats_w6a6_all_layers_seed0.pt \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 64 \
  --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
  --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $QUAR/stats/cls_stats_w6a6_seed0.pt --quar_global_target_ema_alpha 0.05"
mflags(){ case "$1" in
  noadapt)  echo "--algorithm no_adapt";;
  t3a)      echo "--algorithm t3a --t3a_filter_k 20 --arch vit_base_patch16_224";;
  foa)      echo "--algorithm foa --popsize 1 --foa_stats_dir $QUAR/foa_stats";;
  fozo)     echo "--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 --fozo_stats_file $QUAR/stats/train_stats_2.pt";;
  zoa)      echo "--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 --zoa_layer_balance none --official_zoa_stats_file $QUAR/zoa_stats/w6a6.pt";;
  neo)      echo "--algorithm no_adapt --neo_enable";;
  ours_ema) echo "$OURS";;
  ours_cum) echo "$OURS --recal_cumulative";;
esac; }
runcell(){ local gpu=$1 m=$2 s=$3; local tag="${m}_s${s}"; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $(base $s) $(mflags $m) --output "$OUT/$tag" --tag "_$tag" \
    >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
worker(){ local gpu=$1; while :; do
  local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"); [ -z "$line" ] && break
  runcell "$gpu" $line; done; }
cd "$QUAR"
Q="$OUT/queue"; : >"$Q"
for m in noadapt t3a foa fozo zoa neo ours_ema ours_cum; do for s in 1 2; do echo "$m $s" >>"$Q"; done; done
log "OUT=$OUT | $(wc -l <"$Q") codedist runs (8 methods x seeds 1,2) on GPUs [$GPUS]"
for g in $GPUS; do worker $g & done
wait
log "=== per-(method,seed) code-dist 15-corr mean ==="
for m in noadapt t3a foa fozo zoa neo ours_ema ours_cum; do for s in 1 2; do
  v=$(grep -oE "\[CODEDIST\] [a-z_]+ = [0-9.]+" "$OUT/${m}_s${s}.log" 2>/dev/null | grep -oE "[0-9.]+$" | awk '{x+=$1;k++} END{if(k)printf "%.4f(n=%d)",x/k,k; else print "NA"}')
  log "  ${m}_s${s} = $v"
done; done
log "ALL DONE: $OUT"
