#!/usr/bin/env bash
# Is lambda* < 1 because the target moments are estimated online, or because (A1) is only
# approximate? Proposition 1 says lambda* -> 1 as the estimation noise delta^2 -> 0, so if the
# noise is the whole story the sweep should peak at lambda = 1 once the estimate is exact.
#
# ORACLE arm: the per-channel target moments are replaced by the ones measured on this very
# stream (qac_oracle_target_w6a6.pt, from the No Adapt pass of
# runs/mismatch16cum_20260814_115630), so delta^2 ~ 0 with respect to the pooled stream.
# ONLINE arm: the deployed cumulative estimator, for reference at the same lambdas.
# Caveat to report either way: the oracle moments are pooled over the 15 corruptions, not
# per-corruption, so a residual mismatch remains and this is a lower bound on how much of the
# shrinkage the noise explains.
#
# Local branch only, W6A6 PTQ4ViT, ImageNet-C sev5 all 15, 5k, bs 64, seed 0, episodic.
set -uo pipefail
PY="${PY:-python3}"; F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$F/robustbench/data/imagenet_test_image_ids.txt; CMAP=$F/robustbench/data/imagenet_class_to_id_map.json
PTQ=$F/checkpoints/ptq4vit_w6a6_vitb.pt
OUT=$F/runs/lambda_oracle_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$F/runs/last_lambda_oracle.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --corruption all \
--batch_size 64 --workers 3 --seed 0 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ --algorithm no_adapt --recal_cumulative \
--quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
--quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_stats_file $F/stats/qac_stats_w6a6_all_layers_seed0.pt \
--quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 64"
cd "$F"; i=0
for arm in ORACLE ONLINE; do for lam in 0.25 0.50 0.75 1.00; do
  EX=""; [ "$arm" = ORACLE ] && EX="--quar_local_oracle_target_file $F/stats/qac_oracle_target_w6a6.pt"
  t="${arm}_lam${lam}"; g=$((i % 8)); i=$((i+1)); mkdir -p "$OUT/$t"
  ( log "[$t g$g] start"
    CUDA_VISIBLE_DEVICES=$g $PY -u "$F/main.py" $BASE --quar_local_lambda $lam $EX \
      --output "$OUT/$t" --tag "_$t" >"$OUT/$t.log" 2>&1 \
      && log "[$t] done" || log "[$t] FAIL exit=$?" ) &
done; done
wait; log "LAMBDA_ORACLE_DONE $OUT"
for arm in ORACLE ONLINE; do printf "%-7s" "$arm"; for lam in 0.25 0.50 0.75 1.00; do
  a=$(find "$OUT/${arm}_lam${lam}" -name "*acc=*" 2>/dev/null | grep -oE "acc=[0-9.]+" | head -1 | cut -d= -f2)
  printf " lam%s=%s" "$lam" "${a:-NA}"; done; echo; done
