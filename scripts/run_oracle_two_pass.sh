#!/usr/bin/env bash
# Two-pass oracle for the local target moments (tab:pooling, "Converged over an extra pass").
# Pass 1 runs the deployed configuration and saves the moments each corruption converged to;
# pass 2 re-runs the same stream with those moments as a fixed target, so the local branch
# never pays the online estimator's transient. Both passes go through
# tools/analysis/two_pass_oracle.py, which patches the local mixin and leaves main.py alone.
# W6A6 PTQ4ViT ViT-B, ImageNet-C sev5, 15 corruptions, episodic, seed 0, lambda 0.5.
#   bs64 and bs1 share one dump taken on the 5k subset at batch 64.
#   Each label-shift stream (SAR imbalance 1e3 and 5e5, full 50k per corruption) dumps its own,
#   since its stream order differs and so do the moments it converges to.
# Resumable: a finished cell is skipped, and a complete dump is not retaken.
set -uo pipefail
PY="${PY:-python3}"
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"
DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
ORACLE=$QUAR/tools/analysis/two_pass_oracle.py
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt
CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
QAC=$QUAR/stats/qac_stats_w6a6_all_layers_seed0.pt
CLS=$QUAR/stats/cls_stats_w6a6_seed0.pt
OUT="${OUT:-$QUAR/runs/oracle2pass_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"
echo "$OUT" > "$QUAR/runs/last_oracle2pass_dir.txt"
GPUS="${GPUS:-0 1 2}"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }

# deployed QuAR: cumulative local stream, lagged; global inclusive. Identical in both passes.
quar(){ echo "--algorithm no_adapt \
  --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
  --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QAC \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
  --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
  --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $CLS \
  --quar_global_target_ema_alpha 0.05 --recal_cumulative"; }
b5k(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size $1 --workers 8 \
  --corruption all --seed 0 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
  --quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"; }
b50k(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --batch_size 64 --workers 8 --corruption all --seed 0 \
  --quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ \
  --stream_order sar --sar_imbalance_ratio $1 --sar_T 50000 --noniid_seed 2023"; }
cell_done(){ local f="$OUT/$1.log"; [ -f "$f" ] && [ "$(grep -c 'Top-1 Accuracy:' "$f")" -ge 15 ]; }
dump_done(){ [ "$(ls "$1"/*.pt 2>/dev/null | wc -l)" -ge 15 ]; }

pass1(){ local gpu=$1 tag=$2 dir=$3; shift 3
  if dump_done "$dir"; then log "[$tag] dump already complete"; return 0; fi
  mkdir -p "$dir" "$OUT/$tag"; log "[$tag g$gpu] pass 1, dumping converged moments"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u "$ORACLE" --dump-dir "$dir" "$@" $(quar) \
    --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 \
    && log "[$tag] pass 1 done ($(ls "$dir"/*.pt | wc -l) dumps)" || { log "[$tag] pass 1 FAIL exit=$?"; return 1; }; }
pass2(){ local gpu=$1 tag=$2 dir=$3; shift 3
  if cell_done "$tag"; then log "[$tag] skip"; return 0; fi
  mkdir -p "$OUT/$tag"; log "[$tag g$gpu] pass 2, fixed target moments"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u "$ORACLE" --oracle-dir "$dir" "$@" $(quar) \
    --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 \
    && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }

cd "$QUAR"
set -- $GPUS; G1=${1:-0}; G2=${2:-$G1}; G3=${3:-$G1}
# batch 64 and batch 1 share the batch-64 dump
pass1 "$G1" dump_bs64 "$OUT/dump_bs64" $(b5k 64) || exit 1
( pass2 "$G1" oracle_bs64 "$OUT/dump_bs64" $(b5k 64); pass2 "$G1" oracle_bs1 "$OUT/dump_bs64" $(b5k 1) ) &
# each label-shift stream dumps and replays its own
for r in 1000 500000; do
  g=$([ "$r" = 1000 ] && echo "$G2" || echo "$G3")
  ( pass1 "$g" "dump_ir$r" "$OUT/dump_ir$r" $(b50k "$r") && pass2 "$g" "oracle_ir$r" "$OUT/dump_ir$r" $(b50k "$r") ) &
done
wait

log "=== tab:pooling, 'Converged over an extra pass' row (W6A6, 15-corruption mean) ==="
$PY - "$OUT" <<'PYEOF'
import re,sys,os
OUT=sys.argv[1]
def m15(tag):
    f=f"{OUT}/{tag}.log"
    if not os.path.exists(f): return "(missing)"
    a=[float(x) for x in re.findall(r'Top-1 Accuracy: ([0-9.]+)', open(f).read())]
    return f"{sum(a)/len(a):.2f}" if len(a)==15 else f"(incomplete: {len(a)}/15)"
print(f"{'bs 64':>12} {'bs 1':>12} {'ir 1e3':>12} {'ir 5e5':>12}")
print(f"{m15('oracle_bs64'):>12} {m15('oracle_bs1'):>12} {m15('oracle_ir1000'):>12} {m15('oracle_ir500000'):>12}")
print("published: 54.43, 54.53, 54.74, 54.66")
PYEOF
log "ALL DONE: $OUT"
