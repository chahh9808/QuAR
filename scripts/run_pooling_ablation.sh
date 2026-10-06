#!/usr/bin/env bash
# How much do the local target moments need to be pooled? (tab:pooling, Appendix D)
# Reproduces the "Current batch only" row: the local branch corrects with the CURRENT
# batch's exact per-channel moments instead of a running estimate, i.e. the code path
# with --quar_local_stream_ema (and its _alpha/_nwarm/_lag) omitted. The global branch
# stays deployed (cumulative, inclusive), so the row isolates the local estimator.
# The "Running, cumulative (deployed)" row of the same table is tab:main's W6A6 cell
# and the label-shift cells of tab:branchlag; nothing is recomputed for it here.
# W6A6 PTQ4ViT ViT-B, ImageNet-C sev5, 15 corruptions, episodic.
#   5k subset: batch 64 and batch 1, full stream, seeds 0/1/2.
#   label shift: full 50k per corruption, SAR online imbalance ratio 1e3 and 5e5, seed 0.
# Resumable (finished cells are skipped) and GPU-polite (waits for free memory).
set -uo pipefail
PY="${PY:-python3}"
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"
DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt
CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
QAC=$QUAR/stats/qac_stats_w6a6_all_layers_seed0.pt
CLS=$QUAR/stats/cls_stats_w6a6_seed0.pt
OUT="${OUT:-$QUAR/runs/pooling_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"
echo "$OUT" > "$QUAR/runs/last_pooling_dir.txt"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; MINFREE="${MINFREE:-6000}"; PERGPU="${PERGPU:-2}"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }

# local branch, per-batch exact moments: no --quar_local_stream_ema / _alpha / _nwarm / _lag
perbatch(){ echo "--algorithm no_adapt \
  --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
  --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QAC \
  --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
  --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $CLS \
  --quar_global_target_ema_alpha 0.05 --recal_cumulative"; }
quant(){ echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"; }
base5k(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size $2 --workers 8 \
  --corruption all --seed $1 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP"; }
basefull(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --batch_size 64 --workers 8 \
  --corruption all --seed $1"; }
sar(){ echo "--stream_order sar --sar_imbalance_ratio $1 --sar_T 50000 --noniid_seed $2"; }

done_cell(){ local f="$OUT/$1.log"; [ -f "$f" ] && [ "$(grep -c 'Top-1 Accuracy:' "$f")" -ge 15 ]; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 0; }

run_cell(){ local gpu=$1 kind=$2 a=$3 s=$4 tag
  case $kind in
    bs) tag="perbatch_bs${a}_s${s}";;
    ir) tag="perbatch_ir${a}_s${s}";;
  esac
  if done_cell "$tag"; then log "[$tag] skip"; return 0; fi
  mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  case $kind in
    bs) CUDA_VISIBLE_DEVICES=$gpu $PY -u "$QUAR/main.py" $(base5k "$s" "$a") $(quant) $(perbatch) \
          --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1;;
    ir) CUDA_VISIBLE_DEVICES=$gpu $PY -u "$QUAR/main.py" $(basefull "$s") $(quant) $(perbatch) $(sar "$a" $((2023+s))) \
          --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1;;
  esac && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }

worker(){ local gpu=$1
  while :; do
    while [ "$(free_mib "$gpu")" -lt "$MINFREE" ]; do sleep 60; [ -s "$Q" ] || return 0; done
    local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null")
    [ -z "$line" ] && break; run_cell "$gpu" $line
  done; }

cd "$QUAR"; Q="$OUT/queue"; : >"$Q"; : >"$Q.lock"
# longest cells first: batch 1 and the two full-50k label-shift streams
for s in 0 1 2; do done_cell "perbatch_bs1_s${s}" || echo "bs 1 $s" >>"$Q"; done
for ir in 1000 500000; do done_cell "perbatch_ir${ir}_s0" || echo "ir $ir 0" >>"$Q"; done
for s in 0 1 2; do done_cell "perbatch_bs64_s${s}" || echo "bs 64 $s" >>"$Q"; done
log "OUT=$OUT | cells: $(wc -l <"$Q") on [$GPUS] x$PERGPU"
for g in $GPUS; do for i in $(seq 1 "$PERGPU"); do worker "$g" & done; done
wait

log "=== tab:pooling, 'Current batch only' row (W6A6, 15-corruption mean) ==="
$PY - "$OUT" <<'PYEOF'
import re,sys,os,statistics as st
OUT=sys.argv[1]
def m15(tag):
    f=f"{OUT}/{tag}.log"
    if not os.path.exists(f): return None
    a=[float(x) for x in re.findall(r'Top-1 Accuracy: ([0-9.]+)', open(f).read())]
    return sum(a)/len(a) if len(a)==15 else None
def agg(tags):
    vs=[v for v in (m15(t) for t in tags) if v is not None]
    if not vs: return "(incomplete)"
    return f"{sum(vs)/len(vs):.2f}" + (f"±{st.pstdev(vs):.2f}" if len(vs)>1 else "")
print(f"{'bs 64':>12} {'bs 1':>12} {'ir 1e3':>12} {'ir 5e5':>12}")
print(f"{agg([f'perbatch_bs64_s{s}' for s in (0,1,2)]):>12} "
      f"{agg([f'perbatch_bs1_s{s}'  for s in (0,1,2)]):>12} "
      f"{agg(['perbatch_ir1000_s0']):>12} {agg(['perbatch_ir500000_s0']):>12}")
print("published: 52.37±0.01, 38.11±0.01, 42.39, 23.89")
PYEOF
log "ALL DONE: $OUT"
