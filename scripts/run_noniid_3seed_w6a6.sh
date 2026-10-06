#!/usr/bin/env bash
# MAIN-TEXT non-iid robustness table, 3-SEED. Full ImageNet-C (15 corr, FULL 50k/corr), faithful SAR
# online imbalanced label shift, W6A6 ViT-B, bs64, T=50000 (one online pass). ir in {1(iid),1000(~50%),
# 500000(~pure class order)}. Methods = full Table-1 set (No Adapt, T3A, FOA, FOZO, NEO, ZOA) + QuAR
# (cumulative, our DEPLOYED default). 3 seeds: --seed s, --noniid_seed 2023+s (varies stream order so
# even deterministic methods get genuine cross-run variance). No Adapt is stateless -> one cell.
# Deployed QuAR config (block8_9_10_11_all, lambda 0.5, 16 modules).
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
PTQ=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
QAC=$QUAR/stats/qac_stats_w6a6_all_layers_seed0.pt; CLS=$QUAR/stats/cls_stats_w6a6_seed0.pt
FOA_STATS=$QUAR/foa_stats
W6A6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"
# IRS/SEEDS/METHODS/GPUS are overridable so extra imbalance ratios can be added without
# touching the published set. Defaults reproduce the original campaign exactly.
IRS="${IRS:-1 1000 500000}"
SEEDS="${SEEDS:-0 1 2}"
METHODS="${METHODS:-t3a foa fozo neo zoa lgar_cum}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
RUN_NOADAPT="${RUN_NOADAPT:-1}"   # No Adapt is stateless, so it needs one cell only
# OUT is overridable so a second set of workers can join an in-flight campaign's queue
# (flock-guarded) when more GPUs free up, and so a killed run resumes in place.
OUT="${OUT:-$QUAR/runs/noniid_3seed_w6a6_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_noniid_3seed_dir.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
# base for a given seed s
bfull(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --batch_size 64 --workers 8 --corruption all --seed $1"; }
# QuAR cumulative (deployed) -- alpha unused under --recal_cumulative but kept for the stream-EMA flags
ours_cum(){ echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all \
  --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QAC \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag --recal_cumulative \
  --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
  --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $CLS --quar_global_target_ema_alpha 0.05"; }
declare -A CFG
CFG[noadapt]="--algorithm no_adapt"
CFG[neo]="--algorithm no_adapt --neo_enable"
CFG[t3a]="--algorithm t3a --t3a_filter_k 20 --arch vit_base_patch16_224"
CFG[foa]="--algorithm foa --popsize 1 --foa_stats_dir $FOA_STATS"
CFG[fozo]="--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 --fozo_stats_file $QUAR/stats/train_stats_2.pt"
CFG[zoa]="--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 \
  --zoa_layer_balance none --official_zoa_stats_file $QUAR/zoa_stats/w6a6.pt"
CFG[lgar_cum]="$(ours_cum)"
sar(){ echo "--stream_order sar --sar_imbalance_ratio $1 --sar_T 50000 --noniid_seed $2"; }
# runcmd: gpu method ir seed
runcmd(){ local gpu=$1 m=$2 ir=$3 s=$4; local tag="${m}_ir${ir}_s${s}"
  if [ "$(grep -c 'Top-1 Accuracy:' "$OUT/$tag.log" 2>/dev/null || echo 0)" -ge 15 ]; then log "[$tag] already done"; return 0; fi
  mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $(bfull $s) $W6A6 ${CFG[$m]} $(sar $ir $((2023+s))) \
    --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
cd "$QUAR"
log "OUT=$OUT | 3-seed non-iid: 15-corr full-50k bs64 T=50k, ir{1,1000,500000} x {t3a,foa,fozo,neo,zoa,lgar_cum} x seed{0,1,2} + noadapt(1)"
Q=$OUT/queue; [ -f "$Q" ] || : >"$Q"
if [ "${JOIN:-0}" = 1 ]; then
  log "JOIN mode: pulling from the existing queue at $Q, not enqueuing"
else
[ "$RUN_NOADAPT" = 1 ] && echo "noadapt 1 0" >>"$Q"   # No Adapt: stateless -> single cell
for s in $SEEDS; do for ir in $IRS; do for m in $METHODS; do echo "$m $ir $s" >>"$Q"; done; done; done
fi
worker(){ local gpu=$1; while :; do
  local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"); [ -z "$line" ] && break
  runcmd "$gpu" $line; done; }
for g in $GPUS; do worker $g & done
wait
log "=== 3-seed non-iid results: per-(method,ir) 15-corr mean over seeds ==="
IRS="$IRS" SEEDS="$SEEDS" METHODS="$METHODS" $PY - "$OUT" <<'PYEOF'
import re,sys,os,glob
import statistics as st
OUT=sys.argv[1]
def m15(tag):
    f=f"{OUT}/{tag}.log"
    if not os.path.exists(f): return None
    a=[float(x) for x in re.findall(r'Top-1 Accuracy: ([0-9.]+)', open(f).read())]
    return sum(a)/len(a) if len(a)==15 else None
print(f"{'method':<10}{'ir':>8}   seed-means -> mean+/-std")
METHODS=os.environ.get('METHODS','t3a foa fozo neo zoa lgar_cum').split()
for m in ['noadapt']+METHODS:
    IRS=[float(x) if '.' in x else int(x) for x in os.environ.get('IRS','1 1000 500000').split()]
    irs=[1] if m=='noadapt' else IRS
    for ir in irs:
        SEEDS=[int(x) for x in os.environ.get('SEEDS','0 1 2').split()]
        vs=[m15(f"{m}_ir{ir}_s{s}") for s in SEEDS]
        vs=[v for v in vs if v is not None]
        if not vs: print(f"{m:<10}{ir:>8}   (none)"); continue
        mu=sum(vs)/len(vs); sd=st.pstdev(vs) if len(vs)>1 else 0.0
        print(f"{m:<10}{ir:>8}   {'/'.join(f'{v:.2f}' for v in vs):<20} -> {mu:.2f}+/-{sd:.2f}")
PYEOF
log "ALL DONE: $OUT"
