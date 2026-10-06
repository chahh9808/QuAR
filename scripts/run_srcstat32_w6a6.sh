#!/usr/bin/env bash
# Source-statistics provenance check (Appendix D, sensitivity section). Builds the local (qac) and
# global (cls) source statistics from N clean ImageNet TRAIN images with the deployed W6A6 PTQ
# checkpoint, evaluates QuAR (deployed config) on ImageNet-C sev5, 15 corruptions, 5k subset, bs64,
# seed 0, and compares with the deployed 12,800-image validation-split statistics.
# The paper reports N=12800 (train split, 54.14 vs 54.37): `N=12800 GPU=1 bash scripts/run_srcstat32_w6a6.sh`.
# Single GPU, resumable.
set -uo pipefail
PY="${PY:-python}"
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:-/path/to/data/imagenet/Data/CLS-LOC}"
DATA_C="${DATA_C:-/path/to/data/imagenet-c}"
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt
CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
GPU="${GPU:-0}"; N="${N:-32}"; BS=$(( N<64 ? N : 64 )); MB=$(( (N + BS - 1) / BS ))
OUT="${OUT:-$QUAR/runs/srcstat${N}_w6a6_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"
echo "$OUT" > "$QUAR/runs/last_srcstat${N}_dir.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
QAC=$QUAR/stats/qac_stats_w6a6_all_layers_src${N}_seed0.pt
CLS=$QUAR/stats/cls_stats_w6a6_src${N}_seed0.pt
PTQFLAG=""; grep -q '"--ptq_checkpoint"' $QUAR/tools/stats/build_qac_stats.py && PTQFLAG="--ptq_checkpoint $PTQ"
PTQFLAG_CLS=""; grep -q '"--ptq_checkpoint"' $QUAR/tools/stats/build_cls_stats.py && PTQFLAG_CLS="--ptq_checkpoint $PTQ"
cd "$QUAR"
if [ ! -f "$QAC" ]; then log "build local stats from $N train images -> $QAC (ptqflag='$PTQFLAG')"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u tools/stats/build_qac_stats.py --data "$DATA" --quant --quant_w_bit 6 --quant_a_bit 6 $PTQFLAG \
    --quar_local_scope all --source_split train --source_samples $N --batch_size $BS --max_batches $MB --seed 0 \
    --output "$QAC" >"$OUT/build_qac.log" 2>&1 && log "qac done" || { log "qac FAIL exit=$?"; exit 1; }
else log "qac exists, skip"; fi
if [ ! -f "$CLS" ]; then log "build global stats from $N train images -> $CLS (ptqflag='$PTQFLAG_CLS')"
  CUDA_VISIBLE_DEVICES=$GPU $PY -u tools/stats/build_cls_stats.py --data "$DATA" --quant --quant_w_bit 6 --quant_a_bit 6 $PTQFLAG_CLS \
    --source_split train --source_samples $N --batch_size $BS --max_batches $MB --seed 0 \
    --output "$CLS" >"$OUT/build_cls.log" 2>&1 && log "cls done" || { log "cls FAIL exit=$?"; exit 1; }
else log "cls exists, skip"; fi
TAG="quar_src${N}_bs64_s0"; mkdir -p "$OUT/$TAG"
if [ "$(grep -c 'Top-1 Accuracy:' "$OUT/$TAG.log" 2>/dev/null)" -ge 15 ]; then log "[$TAG] done already"; else
log "[$TAG g$GPU] start eval"
CUDA_VISIBLE_DEVICES=$GPU $PY -u main.py --data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 8 \
  --corruption all --seed 0 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
  --quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ \
  --algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
  --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file "$QAC" \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
  --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
  --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file "$CLS" \
  --quar_global_target_ema_alpha 0.05 --recal_cumulative --max_eval_batches -1 \
  --output "$OUT/$TAG" --tag "_$TAG" >"$OUT/$TAG.log" 2>&1 && log "[$TAG] done" || log "[$TAG] FAIL exit=$?"
fi
$PY - "$OUT" "$TAG" "$QUAR" <<'PYEOF'
import re,sys,os
OUT,TAG,Q=sys.argv[1:4]
def m15(f):
    if not os.path.exists(f): return None
    a=[float(x) for x in re.findall(r'Top-1 Accuracy: ([0-9.]+)', open(f).read())]
    return sum(a)/len(a) if len(a)==15 else None
new=m15(f"{OUT}/{TAG}.log")
ref_dir=open(f"{Q}/runs/last_locallag_dir.txt").read().strip()
ref=m15(f"{ref_dir}/cur_bs64_kfull_s0.log")
print(f"W6A6 ImageNet-C 15-corr mean, seed 0: deployed 12,800-img stats = {ref} | {TAG} = {new} | diff = {None if None in (ref,new) else round(new-ref,2)}")
print("RESULT_DONE")
PYEOF
