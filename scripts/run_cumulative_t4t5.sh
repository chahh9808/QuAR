#!/usr/bin/env bash
# Cumulative-estimator variant of the QuAR rows in Table 4 (OOD: C-bar/R/Sketch/3DCC) and
# Table 5 (backbones: ViT-B/DeiT-B/ViT-S @ W6A6). Canonical lambda=0.5; recipe = the published OURS
# flags + --recal_cumulative (verified to reproduce EMA on the main table). Episodic. Only the Ours
# rows are re-run (baselines are estimator-independent). For potential table update; not yet adopted.
#
# ATTACHABLE QUEUE: shares a flock'd queue file so a 2nd worker pool can join later.
#   Fresh start (creates queue, workers on GPUs 6,7):   bash run_cumulative_t4t5.sh
#   Attach more workers to the SAME queue (e.g. once NEO-continual frees 0-5):
#       GPUS="0 1 2 3 4 5" QUEUE_DIR=$(cat runs/last_cum_t4t5_dir.txt) bash run_cumulative_t4t5.sh
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
RDIR="${DATA_R:?set DATA_R to the imagenet-r root}"; SKDIR="${DATA_SKETCH:?set DATA_SKETCH to the imagenet-sketch root}"
CBARDIR="${DATA_CBAR:?set DATA_CBAR to the imagenet-c-bar root}"; DCCDIR="${DATA_3DCC:?set DATA_3DCC to the imagenet-3dcc root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
P13=$QUAR/checkpoints
CBAR_CORR="caustic_refraction,inverse_sparkles,sparkles,perlin_noise,blue_noise_sample,plasma_noise,checkerboard_cutout,cocentric_sine_waves,single_frequency_greyscale,brownish_noise"
DCC_CORR="near_focus,far_focus,fog_3d,flash,color_quant,low_light,xy_motion_blur,z_motion_blur,iso_noise,bit_error,h265_abr,h265_crf"
GPUS="${GPUS:-6 7}"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
quant_flags(){ case "$1" in
  w3a3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $QUAR/checkpoints/w3a3_checkpoint_opt.pt";;
  w4a4) echo "--quant --quant_w_bit 4 --quant_a_bit 4 --adalog_checkpoint $QUAR/checkpoints/w4a4_checkpoint_opt.pt";;
  w6a6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $QUAR/checkpoints/ptq4vit_w6a6_vitb.pt";;
  w8a8) echo "--quant --quant_w_bit 8 --quant_a_bit 8 --ptq_checkpoint $QUAR/checkpoints/ptq4vit_w8a8_vitb.pt";;
esac; }
arch(){ case "$1" in vitb) echo vit_base_patch16_224;; deitb) echo deit_base_patch16_224;; vits) echo vit_small_patch16_224;; esac; }
# OURS at lambda=0.5 + cumulative; $1=quar_local_stats $2=cls_stats
ours_cum(){ echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
  --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $1 \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
  --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
  --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $2 --quar_global_target_ema_alpha 0.05 --recal_cumulative"; }

runcell(){ local gpu=$1 kind=$2 bw=$3 seed=$4; local tag="${kind}_${bw}_s${seed}"; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  local cmd=""
  case "$kind" in
    cbar)  cmd="--data $DATA --data_corruption $CBARDIR --level 5 --imagenet_5k_eval --corruption $CBAR_CORR \
             --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP --batch_size 64 --workers 8 --seed $seed \
             $(quant_flags $bw) $(ours_cum $QUAR/stats/qac_stats_${bw}_all_layers_seed0.pt $QUAR/stats/cls_stats_${bw}_seed0.pt)";;
    3dcc)  cmd="--data $DATA --data_corruption $DCCDIR --level 5 --imagenet_5k_eval --corruption $DCC_CORR \
             --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP --batch_size 64 --workers 8 --seed $seed \
             $(quant_flags $bw) $(ours_cum $QUAR/stats/qac_stats_${bw}_all_layers_seed0.pt $QUAR/stats/cls_stats_${bw}_seed0.pt)";;
    rsketch) cmd="--data $DATA --data_corruption $DATA_C --level 5 --data_rendition $RDIR --data_sketch $SKDIR \
             --corruption rendition,sketch --batch_size 64 --workers 8 --seed $seed \
             $(quant_flags $bw) $(ours_cum $QUAR/stats/qac_stats_${bw}_all_layers_seed0.pt $QUAR/stats/cls_stats_${bw}_seed0.pt)";;
    vitb|deitb|vits) cmd="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --corruption all \
             --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP --batch_size 64 --workers 8 --seed $seed \
             --quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $P13/ptq4vit_w6a6_${kind}_p13.pt --arch $(arch $kind) \
             $(ours_cum $QUAR/stats/qac_stats_w6a6_all_layers_seed0_${kind}.pt $QUAR/stats/cls_stats_w6a6_seed0_${kind}.pt)";;
  esac
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $cmd --output "$OUT/$tag" --tag "_$tag" \
    >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }

worker(){ local gpu=$1; while :; do
  local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"); [ -z "$line" ] && break
  runcell "$gpu" $line; done; }

cd "$QUAR"
if [ -n "${QUEUE_DIR:-}" ]; then          # ATTACH mode: reuse existing queue
  OUT="$QUEUE_DIR"; Q="$OUT/queue"
  log "ATTACH: workers on GPUs [$GPUS] -> existing queue $Q ($(wc -l <"$Q" 2>/dev/null) left)"
else                                       # FRESH: build queue
  OUT=$QUAR/runs/cum_t4t5_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT" > "$QUAR/runs/last_cum_t4t5_dir.txt"
  Q="$OUT/queue"; : >"$Q"
  for kind in cbar 3dcc rsketch; do for bw in w3a3 w4a4 w6a6 w8a8; do for s in 0 1 2; do echo "$kind $bw $s" >>"$Q"; done; done; done
  for kind in vitb deitb vits; do for s in 0 1 2; do echo "$kind w6a6 $s" >>"$Q"; done; done
  log "FRESH: OUT=$OUT  queued $(wc -l <"$Q") cells  workers on GPUs [$GPUS]"
fi
for g in $GPUS; do worker $g & done
wait
log "WORKER POOL [$GPUS] drained. Remaining in queue: $(wc -l <"$Q" 2>/dev/null)"
if [ "$(wc -l <"$Q" 2>/dev/null)" = "0" ]; then
  log "=== cumulative T4/T5 results (Ours, lambda=0.5) ==="
  for kind in cbar 3dcc rsketch vitb deitb vits; do for bw in w3a3 w4a4 w6a6 w8a8; do
    [ "$kind" = vitb -o "$kind" = deitb -o "$kind" = vits ] && [ "$bw" != w6a6 ] && continue
    accs=$(for s in 0 1 2; do lg="$OUT/${kind}_${bw}_s${s}.log"; [ -f "$lg" ] && grep -oE "Top-1 Accuracy: [0-9.]+" "$lg" | grep -oE "[0-9.]+$" | awk '{x+=$1;k++} END{if(k)print x/k}'; done | awk '{s+=$1;n++} END{if(n)printf "%.2f(%ds)",s/n,n; else print "NA"}')
    log "  ${kind}_${bw} = $accs"
  done; done
fi
log "ALL DONE (this pool): $OUT"
