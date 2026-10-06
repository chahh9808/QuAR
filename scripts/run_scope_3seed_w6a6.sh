#!/usr/bin/env bash
# 3-SEED version of Table 3 (tab:scope): recalibration-scope ablation, Local-only (NO cls_shift),
# lambda=0.5, cumulative, ImageNet-C sev5 15-corr 5k, bs64, episodic, W6A6+W4A4. Seeds {0,1,2} in ONE
# dir for clean 3-seed mean+-std (seed0 also re-run = reproduces the current single-seed tab:scope).
# Recipe = run_cumulative_table3.sh localf + --seed s. 12 scopes x 2 bw x 3 seeds = 72 cells.
# Each cell is a cheap 5k 15-corr Local-only run (~10-15 min). Run AFTER the SAR 3-seed run (plan A).
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
ADALOG_W4A4=$QUAR/checkpoints/w4a4_checkpoint_opt.pt
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
OUT=$QUAR/runs/scope_3seed_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_scope_3seed_dir.txt"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
qf(){ case "$1" in w6a6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6";;
                   w4a4) echo "--quant --quant_w_bit 4 --quant_a_bit 4 --adalog_checkpoint $ADALOG_W4A4";; esac; }
base(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 8 \
  --corruption all --seed $2 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $(qf $1)"; }
localf(){ local bw=$1 scope=$2; echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope $scope \
  --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 \
  --quar_local_stats_file $QUAR/stats/qac_stats_${bw}_all_layers_seed0.pt \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag --recal_cumulative"; }
runcell(){ local gpu=$1 bw=$2 scope=$3 s=$4; local tag="${bw}_${scope}_s${s}"; mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $(base $bw $s) $(localf $bw $scope) --output "$OUT/$tag" --tag "_$tag" \
    >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
worker(){ local gpu=$1; while :; do
  local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"); [ -z "$line" ] && break
  runcell "$gpu" $line; done; }
cd "$QUAR"
SCOPES="block8_9_10_11_attn block8_9_10_11_valid block8_9_10_11_all block0_to_3_all block4_to_7_all \
        block11_all block10_11_all block9_10_11_all block7_to_11_all block6_to_11_all block5_to_11_all all"
Q="$OUT/queue"; : >"$Q"
# pack 2 cells per GPU (cheap 5k cells, ~2.5GB each): enqueue all, run 2 workers/GPU
for s in 0 1 2; do for bw in w6a6 w4a4; do for sc in $SCOPES; do echo "$bw $sc $s" >>"$Q"; done; done; done
log "OUT=$OUT | $(wc -l <"$Q") cells (12 scope x 2 bw x 3 seed, Local-only cumulative), 2 workers/GPU on [$GPUS]"
for g in $GPUS; do for i in 1 2; do worker $g & done; done
wait
log "=== Table 3 3-seed (Local-only cumulative, mean top-1 over 15 corr; per-scope 3-seed mean+-std) ==="
$PY - "$OUT" "$SCOPES" <<'PYEOF'
import re,sys,os
import statistics as st
OUT,SCOPES=sys.argv[1],sys.argv[2].split()
def m15(tag):
    f=f"{OUT}/{tag}.log"
    if not os.path.exists(f): return None
    a=[float(x) for x in re.findall(r'Top-1 Accuracy: ([0-9.]+)', open(f).read())]
    return sum(a)/len(a) if len(a)==15 else None
for bw in ['w6a6','w4a4']:
    print(f"--- {bw} ---")
    for sc in SCOPES:
        vs=[m15(f"{bw}_{sc}_s{s}") for s in (0,1,2)]; vs=[v for v in vs if v is not None]
        if not vs: print(f"  {sc:<22} (none)"); continue
        mu=sum(vs)/len(vs); sd=st.pstdev(vs) if len(vs)>1 else 0.0
        print(f"  {sc:<22} {'/'.join(f'{v:.2f}' for v in vs):<20} -> {mu:.2f}+/-{sd:.2f}")
PYEOF
log "ALL DONE: $OUT"
