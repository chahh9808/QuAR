#!/usr/bin/env bash
# Is the scale component of the recalibration worth more where code resolution was
# actually lost? This is the within-group test, so the module-type, width and
# position differences between the two groups cancel: for each group separately we
# compare mean_only (recenter, no rescale) against mean_std (recenter and rescale),
# and read off what the rescale is worth there.
#
# Prediction if the resolution loss is a real contributing mechanism: the rescale is
# worth more at the unnormalized interfaces (attn.proj, mlp.fc2), which forfeit 28.9%
# of channels past half a bit, than at the post-LayerNorm ones (attn.qkv, mlp.fc1),
# which forfeit none. If the two are comparable, the rescale is doing something other
# than restoring resolution and the mechanism should not be claimed even as a
# secondary one.
#
# The mean_std arms are already measured in runs/lnsplit_20260818_132428
# (post-LN 51.25 +- 0.05, unnormalized 52.49 +- 0.08); this run adds the mean_only
# arms under an identical configuration. Local branch only, no global CLS branch, to
# match tab:factorial and tab:scope. W6A6 PTQ4ViT, ImageNet-C sev5 all 15
# corruptions, 5k subset, bs 64, episodic, 3 seeds, cumulative estimator.
# Reference: tab:factorial local mean_only over all 16 = 50.55, mean_std = 53.42.
set -uo pipefail
PY="${PY:-python3}"; F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$F/robustbench/data/imagenet_test_image_ids.txt; CMAP=$F/robustbench/data/imagenet_class_to_id_map.json
PTQ=$F/checkpoints/ptq4vit_w6a6_vitb.pt
OUT=$F/runs/lnsplit_meanonly_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$F/runs/last_lnsplit_meanonly.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
POSTLN=""; UNNORM=""
for b in 8 9 10 11; do
  POSTLN="$POSTLN blocks.$b.attn.qkv blocks.$b.mlp.fc1"
  UNNORM="$UNNORM blocks.$b.attn.proj blocks.$b.mlp.fc2"
done
BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --corruption all \
--batch_size 64 --workers 3 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ --algorithm no_adapt --recal_cumulative"
LOCAL="--quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_only \
--quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $F/stats/qac_stats_w6a6_all_layers_seed0.pt \
--quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 64"
cd "$F"; i=0
for cfg in MO_ALL16 MO_POSTLN MO_UNNORM; do for s in 0 1 2; do
  case $cfg in
    MO_ALL16)  EX="";;
    MO_POSTLN) EX="--quar_local_exclude_layers $UNNORM";;
    MO_UNNORM) EX="--quar_local_exclude_layers $POSTLN";;
  esac
  t="${cfg}_s${s}"; g=$((i % 8)); i=$((i+1)); mkdir -p "$OUT/$t"
  ( log "[$t g$g] start"
    CUDA_VISIBLE_DEVICES=$g $PY -u "$F/main.py" $BASE --seed "$s" $LOCAL $EX \
      --output "$OUT/$t" --tag "_$t" >"$OUT/$t.log" 2>&1 \
      && log "[$t] done" || log "[$t] FAIL exit=$?" ) &
done; done
wait; log "LNSPLIT_MEANONLY_DONE $OUT"
for cfg in MO_ALL16 MO_POSTLN MO_UNNORM; do
  v=""
  for s in 0 1 2; do
    a=$(find "$OUT/${cfg}_s${s}" -name "*acc=*" 2>/dev/null | grep -oE "acc=[0-9.]+" | head -1 | cut -d= -f2)
    v="$v ${a:-NA}"
  done
  echo "$cfg: $v"
done
