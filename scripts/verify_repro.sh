#!/usr/bin/env bash
# Phase 3: does the kept launcher set still reproduce the paper?
#
# Cheap and conclusive. The appendix prints per-corruption accuracy (tab:t2pc for the W6A6
# component ablation, tab:t1percorr across bit-widths), so one corruption is enough to compare
# exact values; there is no need to spend 25 minutes per cell on all fifteen.
#
# tab:t2pc caption fixes the setting: W6A6 PTQ4ViT ViT-B, 5k subset, bs 64, seed 0, episodic
# reset, cumulative running target moments, lagged. The flags below are that setting, taken
# from run_cumulative_maintable.sh, which the archive index names as the Table 1 launcher.
#
# Memory is pinned to node-1 CPUs: this host's node-0 DIMMs produce errors under load.
set -uo pipefail
F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python3}"
DATA="${DATA:?set DATA to the ImageNet root (…/Data/CLS-LOC)}"
DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$F/robustbench/data/imagenet_test_image_ids.txt
CMAP=$F/robustbench/data/imagenet_class_to_id_map.json
PTQ=$F/checkpoints/ptq4vit_w6a6_vitb.pt
QAC=$F/stats/qac_stats_w6a6_all_layers_seed0.pt
CLS=$F/stats/cls_stats_w6a6_seed0.pt
CORR="${CORR:-gaussian_noise}"
OUT="${OUT:-$F/runs/verify_repro_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"
NODE1="$(cat /sys/devices/system/node/node1/cpulist 2>/dev/null || echo 0-71)"
GPU="${GPU:-0}"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/log"; }

BASE="--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval \
--corruption $CORR --seed 0 --batch_size 64 --workers 3 \
--imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ"

LOCAL="--quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all --quar_local_stats mean_std \
--quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QAC \
--quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag"
GLOBAL="--quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std \
--quar_global_alpha 1.0 --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 \
--quar_global_stats_file $CLS --quar_global_target_ema_alpha 0.05"

run(){   # name, extra flags
  local n=$1; shift
  local d="$OUT/$n"; mkdir -p "$d"
  say "[$n] start"
  CUDA_VISIBLE_DEVICES=$GPU taskset -c "$NODE1" $PY -u "$F/main.py" $BASE \
    --algorithm no_adapt --recal_cumulative "$@" \
    --output "$d" --tag "_$n" > "$d/run.log" 2>&1
  local a mods
  a=$(find "$d" -name "*acc=*" 2>/dev/null | grep -oE "acc=[0-9.]+" | head -1 | cut -d= -f2)
  mods=$(grep -oE "modules=[0-9]+" "$d/run.log" 2>/dev/null | head -1)
  say "[$n] acc=${a:-FAIL} ${mods}"
  echo "$n ${a:-FAIL} ${mods:-}" >> "$OUT/results.tsv"
}

: > "$OUT/results.tsv"
run NoAdapt
run Final-only $GLOBAL
run Local-only $LOCAL
run QuAR       $LOCAL $GLOBAL

say "comparing against tab:t2pc / tab:main"
$PY - "$OUT/results.tsv" "$CORR" <<'PY' | tee -a "$OUT/log"
import sys
# tab:t2pc, gaussian_noise column (W6A6, 5k, bs64, seed0, episodic, cumulative lagged)
target = {'gaussian_noise': {'NoAdapt': None, 'Final-only': 45.40, 'Local-only': 46.48, 'QuAR': 46.80}}
corr = sys.argv[2]
tg = target.get(corr, {})
print("\n  %-12s %10s %10s %8s %s" % ("config","measured","paper","delta","verdict"))
ok = True
for ln in open(sys.argv[1]):
    p = ln.split()
    if len(p) < 2: continue
    name, val = p[0], p[1]
    t = tg.get(name)
    if val == 'FAIL':
        print("  %-12s %10s %10s %8s %s" % (name,"FAIL","-","-","RUN FAILED")); ok=False; continue
    v = float(val)
    if t is None:
        print("  %-12s %10.2f %10s %8s %s" % (name,v,"n/a","-","no paper value for this cell")); continue
    d = v - t
    verdict = "MATCH" if abs(d) < 0.005 else ("close" if abs(d) < 0.15 else "MISMATCH")
    if verdict == "MISMATCH": ok = False
    print("  %-12s %10.2f %10.2f %+8.2f %s" % (name,v,t,d,verdict))
print("\n  %s" % ("reproduction confirmed" if ok else "NOT reproduced -- investigate before archiving anything else"))
PY
