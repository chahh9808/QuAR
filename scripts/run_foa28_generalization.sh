#!/usr/bin/env bash
# FOA recommended-config (popsize=28, K=28) on the GENERALIZATION datasets for Table 4 (tab:t4app)
# + appendix tab:gen_full: ImageNet-C-bar, 3DCC, ImageNet-R, ImageNet-Sketch, all 4 bit-widths, 3 seeds.
# Replicates the EXACT protocols of run_cbar.sh / run_3dcc.sh / run_alpha05_rsketch.sh but FOA@28.
#   C-bar/3DCC: 5k subset, --imagenet_5k_eval, level 5. R/Sketch: FULL sets, --corruption rendition,sketch.
# 36 runs over 8 GPUs, SEQUENTIAL per GPU, co-located with other tenants' processes.
# RESUMABLE: fixed output dir + skip-completed. If killed, just re-run; completed (dataset,bw,seed)
# cells (full corruption count present in their log) are skipped, so progress accumulates.
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root (…/Data/CLS-LOC)}"
DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
CBAR="${DATA_CBAR:?set DATA_CBAR to the imagenet-c-bar root}"
DCC="${DATA_3DCC:?set DATA_3DCC to the imagenet-3dcc root}"
RDIR="${DATA_R:?set DATA_R to the imagenet-r root}"; SKDIR="${DATA_SKETCH:?set DATA_SKETCH to the imagenet-sketch root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
FOA_STATS=$QUAR/foa_stats
CBAR_CORR="caustic_refraction,inverse_sparkles,sparkles,perlin_noise,blue_noise_sample,plasma_noise,checkerboard_cutout,cocentric_sine_waves,single_frequency_greyscale,brownish_noise"
DCC_CORR="near_focus,far_focus,fog_3d,flash,color_quant,low_light,xy_motion_blur,z_motion_blur,iso_noise,bit_error,h265_abr,h265_crf"
OUT=$QUAR/runs/foa28_gen; mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_foa28_gen_dir.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
need_for(){ case "$1" in cbar) echo 10;; 3dcc) echo 12;; rsketch) echo 2;; esac; }
qf(){ case "$1" in
  w3a3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $QUAR/checkpoints/w3a3_checkpoint_opt.pt";;
  w4a4) echo "--quant --quant_w_bit 4 --quant_a_bit 4 --adalog_checkpoint $QUAR/checkpoints/w4a4_checkpoint_opt.pt";;
  w6a6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $QUAR/checkpoints/ptq4vit_w6a6_vitb.pt";;
  w8a8) echo "--quant --quant_w_bit 8 --quant_a_bit 8 --ptq_checkpoint $QUAR/checkpoints/ptq4vit_w8a8_vitb.pt";; esac; }
dsbase(){ case "$1" in
  cbar)    echo "--data $DATA --data_corruption $CBAR --level 5 --imagenet_5k_eval --corruption $CBAR_CORR --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP";;
  3dcc)    echo "--data $DATA --data_corruption $DCC --level 5 --imagenet_5k_eval --corruption $DCC_CORR --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP";;
  rsketch) echo "--data $DATA --data_corruption $DATA_C --level 5 --data_rendition $RDIR --data_sketch $SKDIR --corruption rendition,sketch";; esac; }
FOA="--algorithm foa --popsize 28 --foa_stats_dir $FOA_STATS"
run_one(){ local gpu=$1 job=$2
  local ds="${job%%:*}"; local rest="${job#*:}"; local bw="${rest%%:*}"; local seed="${rest##*:}"
  local t="${ds}_${bw}_s${seed}"; local need; need=$(need_for "$ds")
  local have=0; [ -f "$OUT/$t.log" ] && have=$(grep -c "Top-1 Accuracy" "$OUT/$t.log" 2>/dev/null)
  if [ "${have:-0}" -ge "$need" ]; then log "[$t] skip (done $have/$need)"; return; fi
  mkdir -p "$OUT/$t"; log "[$t g$gpu] start (have $have/$need)"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $(dsbase "$ds") $(qf "$bw") $FOA \
    --batch_size 64 --workers 8 --seed "$seed" --output "$OUT/$t" --tag "_$t" \
    >"$OUT/$t.log" 2>&1 && log "[$t] done" || log "[$t] FAIL exit=$?"; }
cd "$QUAR"
JOBS=()
for ds in cbar 3dcc rsketch; do for bw in w3a3 w4a4 w6a6 w8a8; do for s in 0 1 2; do JOBS+=("$ds:$bw:$s"); done; done; done
log "Total ${#JOBS[@]} jobs (FOA@28, resumable) over 8 GPUs, sequential per GPU. OUT=$OUT"
for g in 0 1 2 3 4 5 6 7; do
  ( for ((i=g; i<${#JOBS[@]}; i+=8)); do run_one "$g" "${JOBS[i]}"; done ) &
done
wait
log "=== per-cell completion ==="
for ds in cbar 3dcc rsketch; do need=$(need_for "$ds"); for bw in w3a3 w4a4 w6a6 w8a8; do for s in 0 1 2; do
  t="${ds}_${bw}_s${s}"; have=0; [ -f "$OUT/$t.log" ] && have=$(grep -c "Top-1 Accuracy" "$OUT/$t.log")
  [ "${have:-0}" -ge "$need" ] && st="OK" || st="INCOMPLETE"; log "$t: $have/$need $st"
done; done; done
log "ALL DONE: $OUT"
