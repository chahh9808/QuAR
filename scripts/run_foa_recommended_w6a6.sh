#!/usr/bin/env bash
# FOA at its recommended configuration: FOA at its DEFAULT CMA-ES population
# size popsize=28 (Niu et al., ICML 2024 default) instead of the minimum-forward popsize=1 used in
# Table 1. popsize=28 -> ~29 forwards/batch (es.ask() 28 candidates + best). W6A6 ViT-B, ImageNet-C
# all-15, sev5, 5k subset, bs64, episodic. Seeds 0/1/2 in parallel on GPUs 0/1/2.
# Compare to: paper FOA@popsize1 = 50.39+-0.05 (2 forwards), Ours = 54.17 (1 forward).
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
FOA_STATS=$QUAR/foa_stats
OUT=$QUAR/runs/foa_recommended_w6a6_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_foa_recommended_dir.txt"
W6A6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
run(){ local gpu=$1 seed=$2; local t="foa28_s${seed}"; mkdir -p "$OUT/$t"; log "[$t g$gpu] start (popsize=28)"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py --data $DATA --data_corruption $DATA_C --level 5 \
    --imagenet_5k_eval --batch_size 64 --workers 8 --corruption all --seed $seed \
    --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $W6A6 \
    --algorithm foa --popsize 28 --foa_stats_dir $FOA_STATS \
    --output "$OUT/$t" --tag "_$t" >"$OUT/$t.log" 2>&1 && log "[$t] done" || log "[$t] FAIL exit=$?"; }
cd "$QUAR"
gpu=0; for s in 0 1 2; do run $gpu "$s" & gpu=$((gpu+1)); done
wait
log "=== FOA@popsize28 W6A6, per-seed 15-corruption mean ==="
ms=()
for s in 0 1 2; do
  m=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$OUT/foa28_s${s}.log" | grep -oE "[0-9.]+$" | awk '{x+=$1;n++} END{if(n>0)printf "%.4f",x/n; else print "NA"}')
  n=$(grep -c "Top-1 Accuracy" "$OUT/foa28_s${s}.log")
  log "seed $s = $m (n=$n corruptions)"; ms+=("$m")
done
log "=== mean +- std over 3 seeds ==="
printf '%s\n' "${ms[@]}" | awk '{a[NR]=$1;s+=$1} END{m=s/NR; for(i=1;i<=NR;i++)v+=(a[i]-m)^2; sd=sqrt(v/NR); printf "FOA@popsize28 W6A6 = %.2f +- %.2f  (vs FOA@1 50.39, Ours 54.17)\n",m,sd}'
log "ALL DONE: $OUT"
