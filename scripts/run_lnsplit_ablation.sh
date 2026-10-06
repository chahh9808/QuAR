#!/usr/bin/env bash
# Does the code-level starvation predict where the accuracy comes from?
#
# The starvation is confined to the interfaces with no LayerNorm in front of them
# (attn.proj, mlp.fc2: 28.9% of channels forfeit >=0.5 bit) and is exactly zero at
# the ones that follow a LayerNorm (attn.qkv after norm1, mlp.fc1 after norm2).
# So recalibrating ONLY the post-LN pair isolates the pure mean-displacement
# mechanism, since there is no resolution to restore there; recalibrating ONLY the
# unnormalized pair mixes mean displacement with resolution restoration.
#
# Confound to report either way: the unnormalized pair covers 15,360 channels
# against the post-LN pair's 6,144, because the post-GELU input to mlp.fc2 is 4x
# wide. A larger gain for the unnormalized pair is therefore not by itself evidence
# for the resolution mechanism.
#
# Config: Local-only (no global CLS branch) to match tab:scope, W6A6 PTQ4ViT,
# ImageNet-C sev5 all 15 corruptions, RobustBench 5k, bs 64, episodic, 3 seeds,
# deployed cumulative estimator. ALL16 is a control that should reproduce
# tab:scope's 53.42 and confirm the config matches.
set -uo pipefail
PY="${PY:-python3}"; F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$F/robustbench/data/imagenet_test_image_ids.txt; CMAP=$F/robustbench/data/imagenet_class_to_id_map.json
PTQ=$F/checkpoints/ptq4vit_w6a6_vitb.pt
OUT=$F/runs/lnsplit_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$F/runs/last_lnsplit.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
POSTLN=""; UNNORM=""
for b in 8 9 10 11; do
  POSTLN="$POSTLN blocks.$b.attn.qkv blocks.$b.mlp.fc1"
  UNNORM="$UNNORM blocks.$b.attn.proj blocks.$b.mlp.fc2"
done
BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --corruption all \
--batch_size 64 --workers 3 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ --algorithm no_adapt --recal_cumulative"
LOCAL="--quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
--quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $F/stats/qac_stats_w6a6_all_layers_seed0.pt \
--quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 64"
cd "$F"; i=0
for cfg in ALL16 POSTLN_ONLY UNNORM_ONLY; do for s in 0 1 2; do
  case $cfg in
    ALL16)       EX="";;
    POSTLN_ONLY) EX="--quar_local_exclude_layers $UNNORM";;   # keep qkv+fc1 only
    UNNORM_ONLY) EX="--quar_local_exclude_layers $POSTLN";;   # keep proj+fc2 only
  esac
  t="${cfg}_s${s}"; g=$((i % 8)); i=$((i+1)); mkdir -p "$OUT/$t"
  ( log "[$t g$g] start"
    CUDA_VISIBLE_DEVICES=$g $PY -u "$F/main.py" $BASE --seed "$s" $LOCAL $EX \
      --output "$OUT/$t" --tag "_$t" >"$OUT/$t.log" 2>&1 \
      && log "[$t] done" || log "[$t] FAIL exit=$?" ) &
done; done
wait; log "LNSPLIT_DONE $OUT"
for cfg in ALL16 POSTLN_ONLY UNNORM_ONLY; do
  v=""
  for s in 0 1 2; do
    a=$(find "$OUT/${cfg}_s${s}" -name "*acc=*" 2>/dev/null | grep -oE "acc=[0-9.]+" | head -1 | cut -d= -f2)
    v="$v ${a:-NA}"
  done
  echo "$cfg: $v"
done
