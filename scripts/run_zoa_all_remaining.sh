#!/usr/bin/env bash
# The ZOA cells of the paper under its paper's LayerNorm set (the
# configure_model_for_official_zoa default).
#
# Background: the fix_layers patterns carried upstream's "model." prefix while we
# call configure_model on the bare net, so all 50 LayerNorm tensors were adapted
# instead of the paper's 32. Effect is negligible at W6A6 (-0.03) but large at the
# low bit-widths (+1.23 at W4A4, +1.09 at W3A3), so every ZOA cell needs redoing.
# tab:main and tab:noniid are already done and applied.
#
# Tables covered here:
#   T3       continual, 10 rounds x 15 corruptions, W6A6 + W3A3, 3 seeds
#   T4APP    ImageNet-R / Sketch / C-bar / 3DCC, 4 bit-widths, 3 seeds
#   CROSSQ   cross-quantizer, adalog vs ptq4vit at 4 bit-widths, seed 0
#   HELDC    held-out corruptions (gaussian_blur, saturate, spatter, speckle), 3 seeds
#   FULL50K  full 50k per corruption, 4 bit-widths, 3 seeds
#   W4A8     asymmetric deployment format, 3 seeds
#
# Scheduling: a flock-guarded queue with PROCS_PER_GPU workers per GPU rather than
# per-batch barriers, so a slow continual run never idles the other slots. Each
# run peaks near 3 GiB against 24 GiB, so 3 per GPU is comfortable.
set -uo pipefail
PY="${PY:-python3}"; F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
DATA_R="${DATA_R:?set DATA_R to the imagenet-r root}"; DATA_SK="${DATA_SKETCH:?set DATA_SKETCH to the imagenet-sketch root}"
CBAR="${DATA_CBAR:?set DATA_CBAR to the imagenet-c-bar root}"; DCC="${DATA_3DCC:?set DATA_3DCC to the imagenet-3dcc root}"
IDS=$F/robustbench/data/imagenet_test_image_ids.txt; CMAP=$F/robustbench/data/imagenet_class_to_id_map.json
PTQ_W6A6=$F/checkpoints/ptq4vit_w6a6_vitb.pt
PTQ_W8A8=$F/checkpoints/ptq4vit_w8a8_vitb.pt
CBAR_CORR="caustic_refraction,inverse_sparkles,sparkles,perlin_noise,blue_noise_sample,plasma_noise,checkerboard_cutout,cocentric_sine_waves,single_frequency_greyscale,brownish_noise"
DCC_CORR="near_focus,far_focus,fog_3d,flash,color_quant,low_light,xy_motion_blur,z_motion_blur,iso_noise,bit_error,h265_abr,h265_crf"
HELDC_CORR="gaussian_blur,saturate,spatter,speckle_noise"
OUT=$F/runs/zoa_all_remaining_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
echo "$OUT" > "$F/runs/last_zoa_all_remaining.txt"
Q="$OUT/queue"; LK="$OUT/queue.lock"; : >"$Q"; : >"$LK"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }

ZOA="--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 --zoa_layer_balance none"
qf(){ case "$1" in
  w3a3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $F/checkpoints/w3a3_checkpoint_opt.pt";;
  w4a4) echo "--quant --quant_w_bit 4 --quant_a_bit 4 --adalog_checkpoint $F/checkpoints/w4a4_checkpoint_opt.pt";;
  w6a6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6";;
  w8a8) echo "--quant --quant_w_bit 8 --quant_a_bit 8 --ptq_checkpoint $PTQ_W8A8";;
  w4a8) echo "--quant --quant_w_bit 4 --quant_a_bit 8 --adalog_checkpoint $F/checkpoints/w4a8_checkpoint_opt.pt";;
esac; }
add(){ printf '%s\t%s\n' "$1" "$2" >>"$Q"; }   # tag, flags

for s in 0 1 2; do for bw in w6a6 w3a3; do
  add "T3_${bw}_s${s}" "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --corruption all \
--batch_size 64 --workers 3 --seed $s $(qf $bw) --corruption_repeat 10 --continual \
$ZOA --official_zoa_stats_file $F/zoa_stats/${bw}.pt"
done; done

for s in 0 1 2; do for bw in w3a3 w4a4 w6a6 w8a8; do
  add "T4_rs_${bw}_s${s}" "--data $DATA --data_rendition $DATA_R --data_sketch $DATA_SK \
--batch_size 64 --workers 3 --corruption rendition,sketch --seed $s $(qf $bw) \
$ZOA --official_zoa_stats_file $F/zoa_stats/${bw}.pt"
  add "T4_cbar_${bw}_s${s}" "--data $DATA --data_corruption $CBAR --level 5 --imagenet_5k_eval \
--corruption $CBAR_CORR --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
--batch_size 64 --workers 3 --seed $s $(qf $bw) $ZOA --official_zoa_stats_file $F/zoa_stats/${bw}.pt"
  add "T4_3dcc_${bw}_s${s}" "--data $DATA --data_corruption $DCC --level 5 --imagenet_5k_eval \
--corruption $DCC_CORR --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
--batch_size 64 --workers 3 --seed $s $(qf $bw) $ZOA --official_zoa_stats_file $F/zoa_stats/${bw}.pt"
  add "HELDC_${bw}_s${s}" "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval \
--corruption $HELDC_CORR --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
--batch_size 64 --workers 3 --seed $s $(qf $bw) $ZOA --official_zoa_stats_file $F/zoa_stats/${bw}.pt"
  add "FULL50K_${bw}_s${s}" "--data $DATA --data_corruption $DATA_C --level 5 --corruption all \
--batch_size 64 --workers 3 --seed $s $(qf $bw) $ZOA --official_zoa_stats_file $F/zoa_stats/${bw}.pt"
done; done

for s in 0 1 2; do
  add "W4A8_s${s}" "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --corruption all \
--imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP --batch_size 64 --workers 3 --seed $s \
$(qf w4a8) $ZOA --official_zoa_stats_file $F/zoa_stats/w4a8.pt"
done

for bw in w3a3 w4a4 w6a6 w8a8; do
  for q in adalog ptq4vit; do
    case "$q:$bw" in
      adalog:*)     QQ="--quant --quant_w_bit ${bw:1:1} --quant_a_bit ${bw:3:1} --adalog_checkpoint $F/checkpoints/${bw}_checkpoint_opt.pt";;
      ptq4vit:w3a3) QQ="--quant --quant_w_bit 3 --quant_a_bit 3 --ptq_checkpoint $F/checkpoints/ptq4vit_w3a3_vitb.pt";;
      ptq4vit:w4a4) QQ="--quant --quant_w_bit 4 --quant_a_bit 4 --ptq_checkpoint $F/checkpoints/ptq4vit_w4a4_vitb.pt";;
      ptq4vit:w6a6) QQ="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6";;
      ptq4vit:w8a8) QQ="--quant --quant_w_bit 8 --quant_a_bit 8 --ptq_checkpoint $PTQ_W8A8";;
    esac
    add "CROSSQ_${q}_${bw}_s0" "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval \
--corruption all --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
--batch_size 64 --workers 3 --seed 0 $QQ $ZOA --official_zoa_stats_file $F/zoa_stats/${bw}.pt"
  done
done

TOTAL=$(wc -l <"$Q")
log "queued $TOTAL ZOA runs -> $OUT"
GPUS=(0 1 2 3 4 5 6 7); PROCS_PER_GPU=${PROCS_PER_GPU:-3}
pop(){ local line=""; { flock 9
    line=$(head -1 "$Q" 2>/dev/null || true)
    [ -n "$line" ] && sed -i '1d' "$Q"
  } 9>"$LK"; printf '%s' "$line"; }
worker(){ local gpu=$1 slot=$2
  while :; do
    local job; job=$(pop); [ -z "$job" ] && break
    local tag flags; tag=${job%%$'\t'*}; flags=${job#*$'\t'}
    mkdir -p "$OUT/$tag"; log "[$tag g${gpu}.${slot}] start"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u "$F/main.py" $flags --output "$OUT/$tag" --tag "_$tag" \
      >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"
  done; }
cd "$F"
for g in "${GPUS[@]}"; do for p in $(seq 1 "$PROCS_PER_GPU"); do worker "$g" "$p" & done; done
wait
log "ZOA_ALL_REMAINING_DONE $OUT  ($TOTAL runs)"
