#!/usr/bin/env bash
# Table 3 (tab:scope) recalibration-scope ablation extended to W3A3 and W8A8.
# Same recipe as scripts/run_scope_3seed_w6a6.sh: Local-only (no global CLS branch), lambda=0.5,
# cumulative estimator, ImageNet-C sev5 all 15 corruptions, RobustBench 5k, bs64, episodic, seeds 0/1/2.
# 12 scopes x 2 bit-widths x 3 seeds = 72 cells.
# GPU policy: never touch another job's card. A worker takes a cell only when its GPU has at least
# MINFREE MiB free, so cards busy with other runs are skipped and picked up automatically once freed.
# Resumable: a cell whose log already has 15 "Top-1 Accuracy:" lines is skipped.
set -uo pipefail
PY="${PY:-python}"
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:-/path/to/data/imagenet/Data/CLS-LOC}"
DATA_C="${DATA_C:-/path/to/data/imagenet-c}"
PTQ_W8A8=$QUAR/checkpoints/ptq4vit_w8a8_vitb.pt
ADALOG_W3A3=$QUAR/checkpoints/w3a3_checkpoint_opt.pt
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt
CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
OUT="${OUT:-$QUAR/runs/scope_3seed_bw38_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"
echo "$OUT" > "$QUAR/runs/last_scope_bw38_dir.txt"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
MINFREE="${MINFREE:-6000}"     # MiB free required before a worker starts a cell on that card
PERGPU="${PERGPU:-2}"          # concurrent cells per card (each cell peaks ~2.5 GB)
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
qf(){ case "$1" in
  w8a8) echo "--quant --quant_w_bit 8 --quant_a_bit 8 --ptq_checkpoint $PTQ_W8A8";;
  w3a3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $ADALOG_W3A3";; esac; }
base(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 8 \
  --corruption all --seed $2 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $(qf $1)"; }
localf(){ local bw=$1 scope=$2; echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope $scope \
  --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 \
  --quar_local_stats_file $QUAR/stats/qac_stats_${bw}_all_layers_seed0.pt \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag --recal_cumulative"; }
done_cell(){ local f="$OUT/$1.log"; [ -f "$f" ] && [ "$(grep -c 'Top-1 Accuracy:' "$f")" -ge 15 ]; }
free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 0; }
runcell(){ local gpu=$1 bw=$2 scope=$3 s=$4; local tag="${bw}_${scope}_s${s}"
  if done_cell "$tag"; then log "[$tag] skip (already complete)"; return 0; fi
  mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $(base $bw $s) $(localf $bw $scope) --output "$OUT/$tag" --tag "_$tag" \
    >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
worker(){ local gpu=$1
  while :; do
    # wait for this card to have room; re-checked before every cell so busy cards are simply skipped
    while [ "$(free_mib "$gpu")" -lt "$MINFREE" ]; do sleep 60; [ -s "$Q" ] || return 0; done
    local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null")
    [ -z "$line" ] && break
    runcell "$gpu" $line
  done; }
cd "$QUAR"
SCOPES="block8_9_10_11_attn block8_9_10_11_valid block8_9_10_11_all block0_to_3_all block4_to_7_all \
        block11_all block10_11_all block9_10_11_all block7_to_11_all block6_to_11_all block5_to_11_all all"
Q="$OUT/queue"; : >"$Q"; : >"$Q.lock"
for s in 0 1 2; do for bw in w8a8 w3a3; do for sc in $SCOPES; do
  done_cell "${bw}_${sc}_s${s}" || echo "$bw $sc $s" >>"$Q"; done; done; done
log "OUT=$OUT | $(wc -l <"$Q") cells queued (12 scopes x 2 bw x 3 seeds), ${PERGPU}/GPU on [$GPUS], MINFREE=${MINFREE}MiB"
for g in $GPUS; do for i in $(seq 1 $PERGPU); do worker "$g" & done; done
wait
log "=== Table 3 extension, Local-only cumulative, 15-corruption mean top-1, 3-seed ==="
$PY - "$OUT" "$SCOPES" <<'PYEOF'
import re,sys,os,statistics as st
OUT,SCOPES=sys.argv[1],sys.argv[2].split()
def m15(tag):
    f=f"{OUT}/{tag}.log"
    if not os.path.exists(f): return None
    a=[float(x) for x in re.findall(r'Top-1 Accuracy: ([0-9.]+)', open(f).read())]
    return sum(a)/len(a) if len(a)==15 else None
for bw in ['w3a3','w8a8']:
    print(f"--- {bw} ---")
    for sc in SCOPES:
        vs=[m15(f"{bw}_{sc}_s{s}") for s in (0,1,2)]; vs=[v for v in vs if v is not None]
        if not vs: print(f"  {sc:<24} (none)"); continue
        mu=sum(vs)/len(vs); sd=st.pstdev(vs) if len(vs)>1 else 0.0
        print(f"  {sc:<24} {'/'.join(f'{v:.2f}' for v in vs):<22} -> {mu:.2f}+/-{sd:.2f}")
PYEOF
log "ALL DONE: $OUT"
