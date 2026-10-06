#!/usr/bin/env bash
# Phase 3, widened: the same one-corruption check at the other three bit-widths.
# Targets are tab:t1percorr's gaussian_noise column, so a single corruption per bit-width
# settles it. W3A3/W4A4 use AdaLog checkpoints, so this also exercises that path.
set -uo pipefail
F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python3}"
DATA="${DATA:?set DATA to the ImageNet root (…/Data/CLS-LOC)}"
DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$F/robustbench/data/imagenet_test_image_ids.txt
CMAP=$F/robustbench/data/imagenet_class_to_id_map.json
OUT="${OUT:-$F/runs/verify_bw_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"
NODE1="$(cat /sys/devices/system/node/node1/cpulist 2>/dev/null || echo 0-71)"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/log"; }
qflags(){ case "$1" in
  w3a3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $F/checkpoints/w3a3_checkpoint_opt.pt";;
  w4a4) echo "--quant --quant_w_bit 4 --quant_a_bit 4 --adalog_checkpoint $F/checkpoints/w4a4_checkpoint_opt.pt";;
  w8a8) echo "--quant --quant_w_bit 8 --quant_a_bit 8 --ptq_checkpoint $F/checkpoints/ptq4vit_w8a8_vitb.pt";;
esac; }
: > "$OUT/results.tsv"
i=0
for bw in w3a3 w4a4 w8a8; do
  for cfg in NoAdapt QuAR; do
    d="$OUT/${bw}_${cfg}"; mkdir -p "$d"; g=$((i%8)); i=$((i+1))
    extra=""
    [ "$cfg" = QuAR ] && extra="--quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all \
--quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 \
--quar_local_stats_file $F/stats/qac_stats_${bw}_all_layers_seed0.pt --quar_local_stream_ema --quar_local_stream_alpha 0.05 \
--quar_local_stream_nwarm 0 --quar_local_stream_lag --quar_global_enable --quar_global_position post_ln \
--quar_global_mode mean_std --quar_global_alpha 1.0 --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 \
--quar_global_stats_file $F/stats/cls_stats_${bw}_seed0.pt --quar_global_target_ema_alpha 0.05"
    say "[$bw/$cfg gpu$g] start"
    ( CUDA_VISIBLE_DEVICES=$g taskset -c "$NODE1" $PY -u "$F/main.py" \
        --data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval \
        --corruption gaussian_noise --seed 0 --batch_size 64 --workers 3 \
        --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
        $(qflags $bw) --algorithm no_adapt --recal_cumulative $extra \
        --output "$d" --tag "_${bw}_${cfg}" > "$d/run.log" 2>&1
      a=$(find "$d" -name "*acc=*" 2>/dev/null | grep -oE "acc=[0-9.]+" | head -1 | cut -d= -f2)
      echo -e "$bw\t$cfg\t${a:-FAIL}" >> "$OUT/results.tsv"
      say "[$bw/$cfg] acc=${a:-FAIL}" ) &
  done
done
wait
say "결과 대조"
$PY - "$OUT/results.tsv" <<'PY' | tee -a "$OUT/log"
import sys
# tab:t1percorr, gaussian_noise column
T={('w3a3','NoAdapt'):14.52,('w3a3','QuAR'):20.94,
   ('w4a4','NoAdapt'):45.30,('w4a4','QuAR'):47.06,
   ('w8a8','NoAdapt'):56.34,('w8a8','QuAR'):57.70}
print("\n  %-6s %-9s %10s %9s %8s %s" % ("bit","config","measured","paper","delta","verdict"))
ok=True
for ln in sorted(open(sys.argv[1]).read().strip().split('\n')):
    p=ln.split('\t')
    if len(p)<3: continue
    bw,cfg,v=p[0],p[1],p[2]
    t=T.get((bw,cfg))
    if v=='FAIL': print("  %-6s %-9s %10s %9s %8s RUN FAILED"%(bw,cfg,"FAIL","-","-")); ok=False; continue
    d=float(v)-t
    verdict="MATCH" if abs(d)<0.005 else ("close" if abs(d)<0.15 else "MISMATCH")
    if verdict=="MISMATCH": ok=False
    print("  %-6s %-9s %10.2f %9.2f %+8.2f %s"%(bw,cfg,float(v),t,d,verdict))
print("\n  %s"%("all bit-widths reproduce" if ok else "MISMATCH -- investigate"))
PY
