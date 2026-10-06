#!/usr/bin/env bash
# Does the local branch's lag rule matter under online label shift?
# cur  = deployed (local corrects with the preceding batches only)
# incl = local branch includes the current batch, where a class-dominated batch would enter
#        its own correction. Global branch left as deployed in both arms.
# glag = global branch corrects with the preceding batches (--quar_global_stream_lag), local as
#        deployed: the 'global branch also lagged' label-shift cells of the branch-update table.
# Same protocol as the main non-iid table: W6A6, full ImageNet-C (15 corr, 50k/corr), SAR online
# imbalanced label shift, T=50000, bs64, seed 0. Resumable, GPU-polite.
set -uo pipefail
PY="${PY:-python}"
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:-/path/to/data/imagenet/Data/CLS-LOC}"
DATA_C="${DATA_C:-/path/to/data/imagenet-c}"
W6A6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $QUAR/checkpoints/ptq4vit_w6a6_vitb.pt"
QAC=$QUAR/stats/qac_stats_w6a6_all_layers_seed0.pt; CLS=$QUAR/stats/cls_stats_w6a6_seed0.pt
OUT="${OUT:-$QUAR/runs/noniid_locallag_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"
echo "$OUT" > "$QUAR/runs/last_noniid_locallag_dir.txt"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; MINFREE="${MINFREE:-6000}"; PERGPU="${PERGPU:-1}"
IRS="${IRS:-1 1000 500000}"; SEEDS="${SEEDS:-0}"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
ours(){ local lag=$1; echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all \
  --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QAC \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 $lag --recal_cumulative \
  --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
  --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $CLS --quar_global_target_ema_alpha 0.05"; }
bfull(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --batch_size 64 --workers 8 --corruption all --seed $1"; }
sar(){ echo "--stream_order sar --sar_imbalance_ratio $1 --sar_T 50000 --noniid_seed $2"; }
done_cell(){ local f="$OUT/$1.log"; [ -f "$f" ] && [ "$(grep -c 'Top-1 Accuracy:' "$f")" -ge 15 ]; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 0; }
runcell(){ local gpu=$1 m=$2 ir=$3 s=$4; local tag="${m}_ir${ir}_s${s}"
  if done_cell "$tag"; then log "[$tag] skip"; return 0; fi
  local lag="--quar_local_stream_lag"; [ "$m" = incl ] && lag=""; [ "$m" = glag ] && lag="--quar_local_stream_lag --quar_global_stream_lag"
  mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $(bfull $s) $W6A6 $(ours "$lag") $(sar $ir $((2023+s))) \
    --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
worker(){ local gpu=$1
  while :; do
    while [ "$(free_mib "$gpu")" -lt "$MINFREE" ]; do sleep 60; [ -s "$Q" ] || return 0; done
    local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null")
    [ -z "$line" ] && break; runcell "$gpu" $line
  done; }
cd "$QUAR"; Q="$OUT/queue"; : >"$Q"; : >"$Q.lock"
for s in $SEEDS; do for ir in $IRS; do for m in cur incl glag; do
  done_cell "${m}_ir${ir}_s${s}" || echo "$m $ir $s" >>"$Q"; done; done; done
log "OUT=$OUT | $(wc -l <"$Q") cells (ir[$IRS] x {cur,incl,glag} x seed[$SEEDS]), 1/GPU on [$GPUS]"
for g in $GPUS; do for i in $(seq 1 $PERGPU); do worker "$g" & done; done
wait
log "=== branch update order under online label shift (W6A6, full 50k/corr, 15-corr mean) ==="
$PY - "$OUT" "$IRS" "$SEEDS" <<'PYEOF'
import re,sys,os,statistics as st
OUT,IRS,SEEDS=sys.argv[1],sys.argv[2].split(),sys.argv[3].split()
def m15(tag):
    f=f"{OUT}/{tag}.log"
    if not os.path.exists(f): return None
    a=[float(x) for x in re.findall(r'Top-1 Accuracy: ([0-9.]+)', open(f).read())]
    return sum(a)/len(a) if len(a)==15 else None
print(f"{'imbalance':>12} {'deployed':>12} {'local incl':>12} {'global lag':>12}")
for ir in IRS:
    row={}
    for m in ('cur','incl','glag'):
        vs=[m15(f"{m}_ir{ir}_s{s}") for s in SEEDS]; vs=[v for v in vs if v is not None]
        row[m]=sum(vs)/len(vs) if vs else None
    fmt=lambda v: f"{v:>12.2f}" if v is not None else f"{'--':>12}"
    print(f"{ir:>12} {fmt(row['cur'])} {fmt(row['incl'])} {fmt(row['glag'])}")
PYEOF
log "ALL DONE: $OUT"
