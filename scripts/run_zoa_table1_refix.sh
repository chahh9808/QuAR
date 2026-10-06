#!/usr/bin/env bash
# Table 1 ZOA re-measurement under the corrected fix_layers, 4 bit-widths x 3 seeds.
#
# Why: configure_model_for_official_zoa replicated ZOA's fix_layers list but with
# upstream's 'model.' prefix, which never matched our bare-net parameter names, so
# all 50 LayerNorm tensors (38,400 params) were adapted instead of the paper's 32
# (24,576, blocks 1-8 x {norm1,norm2}). Seed-0 effect of the correction:
#   W6A6  51.38 -> 51.35  (-0.03, inside the reported +/-0.09)
#   W4A4  49.59 -> 50.78  (+1.19)
#   W3A3  30.81 -> 31.40  (+0.59)
# The low-bit shifts are far outside the reported seed spread, and at W4A4 ZOA
# overtakes NEO as the strongest baseline, so the reported cells and the margins
# quoted in the abstract/intro have to be recomputed from these runs.
#
# Every flag other than the corrected default is verbatim from run_table1_*.sh.
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ_W8A8=$QUAR/checkpoints/ptq4vit_w8a8_vitb.pt
PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
OUT=$QUAR/runs/zoa_table1_refix_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
echo "$OUT">"$QUAR/runs/last_zoa_table1_refix_dir.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
ZOA="--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 --official_zoa_lr_eps 0.0002 \
  --zoa_layer_balance none"
qflags(){ case "$1" in
  8) echo "--quant --quant_w_bit 8 --quant_a_bit 8 --ptq_checkpoint $PTQ_W8A8";;
  6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6";;
  4) echo "--quant --quant_w_bit 4 --quant_a_bit 4 --adalog_checkpoint $QUAR/checkpoints/w4a4_checkpoint_opt.pt";;
  3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $QUAR/checkpoints/w3a3_checkpoint_opt.pt";;
esac; }
run(){ local gpu=$1 bw=$2 s=$3 tag="W${2}A${2}_s${3}"; mkdir -p "$OUT/$tag"
  log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py \
    --data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --corruption all \
    --batch_size 64 --workers 4 --seed "$s" --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP \
    $(qflags "$bw") $ZOA --official_zoa_stats_file "$QUAR/zoa_stats/w${bw}a${bw}.pt" \
    --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 \
    && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
cd "$QUAR"
GPUS=(1 2 3 4 5 6 7)
Q="$OUT/queue"; : >"$Q"
for bw in 8 6 4 3; do for s in 0 1 2; do echo "$bw $s" >>"$Q"; done; done
log "$(wc -l <"$Q") runs on GPUs [${GPUS[*]}]"
i=0
while read -r bw s; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  run "$g" "$bw" "$s" &
  i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
done <"$Q"
wait; log "ALL DONE: $OUT"
for bw in 3 4 6 8; do
  vals=""
  for s in 0 1 2; do
    a=$(find "$OUT/W${bw}A${bw}_s${s}" -name "*acc=*" 2>/dev/null | grep -oE "acc=[0-9.]+" | head -1 | cut -d= -f2)
    vals="$vals ${a:-NA}"
  done
  m=$(echo $vals | awk '{n=0;x=0;for(i=1;i<=NF;i++) if($i!="NA"){x+=$i;n++}} END{if(n)printf "%.2f (n=%d)",x/n,n; else print "NA"}')
  echo "W${bw}A${bw}: $vals  mean $m"
done
