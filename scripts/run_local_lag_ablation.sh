#!/usr/bin/env bash
# Does the lag rule matter at the LOCAL interfaces? cur = deployed (local lagged);
# incl = local branch includes the current batch (global left as deployed, inclusive).
# Together with the global ablation this says whether the asymmetry is justified.
# W6A6, ImageNet-C sev5, 15 corruptions, 5k subset, episodic, QuAR full (local+global).
# bs64 at K in {1,5,full} batches per corruption x 3 seeds (K shows the early-stream cost),
# plus bs1 full at seed 0, where the global batch std is degenerate. Resumable, GPU-polite.
set -uo pipefail
PY="${PY:-python}"
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:-/path/to/data/imagenet/Data/CLS-LOC}"
DATA_C="${DATA_C:-/path/to/data/imagenet-c}"
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt
CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
OUT="${OUT:-$QUAR/runs/locallag_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"
echo "$OUT" > "$QUAR/runs/last_locallag_dir.txt"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; MINFREE="${MINFREE:-6000}"; PERGPU="${PERGPU:-2}"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
quar(){ echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
  --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QUAR/stats/qac_stats_w6a6_all_layers_seed0.pt \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 $LOCALLAG \
  --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
  --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $QUAR/stats/cls_stats_w6a6_seed0.pt \
  --quar_global_target_ema_alpha 0.05 --recal_cumulative"; }
base(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size $2 --workers 8 \
  --corruption all --seed $1 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
  --quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"; }
done_cell(){ local f="$OUT/$1.log"; [ -f "$f" ] && [ "$(grep -c 'Top-1 Accuracy:' "$f")" -ge 15 ]; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 0; }
runcell(){ local gpu=$1 mode=$2 bs=$3 k=$4 s=$5; local kt=${k/-1/full}; local tag="${mode}_bs${bs}_k${kt}_s${s}"
  if done_cell "$tag"; then log "[$tag] skip"; return 0; fi
  local LOCALLAG="--quar_local_stream_lag"; [ "$mode" = incl ] && LOCALLAG=""; export LOCALLAG; local extra=""
  mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $(base $s $bs) $(quar) $extra --max_eval_batches $k \
    --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
worker(){ local gpu=$1
  while :; do
    while [ "$(free_mib "$gpu")" -lt "$MINFREE" ]; do sleep 60; [ -s "$Q" ] || return 0; done
    local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null")
    [ -z "$line" ] && break; runcell "$gpu" $line
  done; }
cd "$QUAR"; Q="$OUT/queue"; : >"$Q"; : >"$Q.lock"
for s in 0 1 2; do for m in cur incl; do for k in 1 5 -1; do
  kt=${k/-1/full}; done_cell "${m}_bs64_k${kt}_s${s}" || echo "$m 64 $k $s" >>"$Q"; done; done; done
for m in cur incl; do done_cell "${m}_bs1_kfull_s0" || echo "$m 1 -1 0" >>"$Q"; done
log "OUT=$OUT | $(wc -l <"$Q") cells | ${PERGPU}/GPU on [$GPUS]"
for g in $GPUS; do for i in $(seq 1 $PERGPU); do worker "$g" & done; done
wait
log "=== local branch: deployed lagged vs including the current batch, W6A6 15-corr mean ==="
$PY - "$OUT" <<'PYEOF'
import re,sys,os,statistics as st
OUT=sys.argv[1]
def m15(tag):
    f=f"{OUT}/{tag}.log"
    if not os.path.exists(f): return None
    a=[float(x) for x in re.findall(r'Top-1 Accuracy: ([0-9.]+)', open(f).read())]
    return sum(a)/len(a) if len(a)==15 else None
print(f"{'setting':>16} {'lagged(dep)':>16} {'inclusive':>16} {'diff':>8}")
for bs,k,seeds in [(64,'1',(0,1,2)),(64,'5',(0,1,2)),(64,'full',(0,1,2)),(1,'full',(0,))]:
    row={}
    for m in ('cur','incl'):
        vs=[m15(f"{m}_bs{bs}_k{k}_s{s}") for s in seeds]; vs=[v for v in vs if v is not None]
        row[m]=(sum(vs)/len(vs), st.pstdev(vs) if len(vs)>1 else 0.0) if vs else None
    if not row['cur'] or not row['incl']: print(f"{f'bs{bs} K={k}':>16} (incomplete)"); continue
    c,l=row['cur'],row['incl']
    print(f"{f'bs{bs} K={k}':>16} {c[0]:>10.2f}±{c[1]:.2f} {l[0]:>10.2f}±{l[1]:.2f} {l[0]-c[0]:>+8.2f}")
PYEOF
log "ALL DONE: $OUT"
