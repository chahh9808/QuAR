#!/usr/bin/env bash
# 3-SEED version of Table 6 (tab:sens, and tab:sens_full in the appendix): lambda / batch-size / source-stat-setsize sweeps, CUMULATIVE.
# Ours = no_adapt + QAC(local, block8_9_10_11_all, alpha=lambda) + CLS(global), W6A6, ImageNet-C sev5
# 15-corr 5k, episodic. Seeds {0,1,2} vary ONLY the eval --seed (data order); source stats are the
# fixed seed0 files, EXACTLY as the scope 3-seed launcher. seed0 reproduces the current single-seed
# tab:sens (lambda 51.16/52.62/53.99/54.37/54.20/53.51; bs 54.31/54.35/54.49/54.34/54.37/54.30;
# ss 52.93/53.62/53.78/54.17/54.27/54.38). Recipe = run_cumulative_table6lam / _bs_sweep / _setsize.
# 3 sweeps x 6 vals x 3 seeds = 54 cells (bs=1/4 are slow; enqueued first).
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
QACB3=$QUAR/stats/qac_stats_w6a6_all_layers_seed0.pt; CLS=$QUAR/stats/cls_stats_w6a6_seed0.pt
OUT=$QUAR/runs/sens_3seed_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_sens_3seed_dir.txt"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
W6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6"
base(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size $1 --workers 8 \
  --corruption all --seed $2 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $W6"; }
qac(){ echo "--quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std --quar_local_eps 1e-6 \
  --quar_local_clip_ratio 3.0 --quar_local_lambda $1 --quar_local_stats_file $2 --quar_local_stream_ema --quar_local_stream_alpha 0.05 \
  --quar_local_stream_nwarm 0 --quar_local_stream_lag --recal_cumulative"; }
clsf(){ echo "--quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
  --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $1 --quar_global_target_ema_alpha 0.05"; }
# cell = "kind val seed"
runcell(){ local gpu=$1 kind=$2 val=$3 s=$4; local tag="${kind}_${val/./p}_s${s}"; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  local cmd
  case "$kind" in
    lam) cmd="$(base 64 $s) --algorithm no_adapt $(qac $val $QACB3) $(clsf $CLS)" ;;
    bs)  cmd="$(base $val $s) --algorithm no_adapt $(qac 0.5 $QACB3) $(clsf $CLS)" ;;
    ss)  cmd="$(base 64 $s) --algorithm no_adapt $(qac 0.5 $QUAR/stats/qac_stats_w6a6_all_layers_seed0_nb${val}.pt) $(clsf $QUAR/stats/cls_stats_w6a6_seed0_nb${val}.pt)" ;;
  esac
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $cmd --output "$OUT/$tag" --tag "_$tag" \
    >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
worker(){ local gpu=$1; while :; do
  local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"); [ -z "$line" ] && break
  runcell "$gpu" $line; done; }
cd "$QUAR"
Q="$OUT/queue"; : >"$Q"
# enqueue slow bs cells first (bs=1,4), then rest of bs, lambda, setsize
for s in 0 1 2; do for bs in 1 4;            do echo "bs $bs $s"  >>"$Q"; done; done
for s in 0 1 2; do for bs in 8 16 64 128;    do echo "bs $bs $s"  >>"$Q"; done; done
for s in 0 1 2; do for lam in 0 0.1 0.3 0.5 0.7 1.0; do echo "lam $lam $s" >>"$Q"; done; done
for s in 0 1 2; do for N in 8 16 32 64 128 256;      do echo "ss $N $s"    >>"$Q"; done; done
log "OUT=$OUT | $(wc -l <"$Q") cells (lam/bs/ss x 6 x 3 seeds, cumulative), 1 worker/GPU on [$GPUS]"
for g in $GPUS; do worker $g & done
wait
log "=== Table 7 3-seed (per-cell 15-corr mean; per-config 3-seed mean+-std) ==="
$PY - "$OUT" <<'PYEOF'
import os,re,sys,math
OUT=sys.argv[1]
def cellmean(tag):
    p=os.path.join(OUT,tag+".log")
    if not os.path.isfile(p): return None
    vs=[float(x) for x in re.findall(r"Top-1 Accuracy: ([0-9.]+)", open(p).read())]
    return sum(vs)/len(vs) if len(vs)==15 else (sum(vs)/len(vs) if vs else None)
def agg(kind,vals):
    print(f"--- {kind} ---")
    for v in vals:
        vt=str(v).replace('.','p')
        ms=[cellmean(f"{kind}_{vt}_s{s}") for s in (0,1,2)]
        ms=[m for m in ms if m is not None]
        if not ms: print(f"  {v}: NA"); continue
        mu=sum(ms)/len(ms); sd=math.sqrt(sum((x-mu)**2 for x in ms)/len(ms)) if len(ms)>1 else 0.0
        print(f"  {v}: mean={mu:.2f} std={sd:.2f} n={len(ms)}  ({', '.join(f'{x:.2f}' for x in ms)})")
agg("lam",[0,0.1,0.3,0.5,0.7,1.0])
agg("bs",[1,4,8,16,64,128])
agg("ss",[8,16,32,64,128,256])
PYEOF
log "ALL DONE: $OUT"
