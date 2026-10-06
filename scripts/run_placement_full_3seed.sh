#!/usr/bin/env bash
# tab:placement at all four bit-widths, 3 seeds: post-quant (output_only) and block-output (residual stream)
# rows; the pre-quant row is Table 1's Local-only. Shares GPUs politely: one of our jobs per GPU, only on
# GPUs with >= MINFREE MiB free. Resumable: finished cells (15 corruptions in log) are skipped.
set -u
PY="${PY:-python}"; QUAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$QUAR"
DATA=/path/to/data/imagenet/Data/CLS-LOC; DATA_C=/path/to/data/imagenet-c
IDS=robustbench/data/imagenet_test_image_ids.txt; CMAP=robustbench/data/imagenet_class_to_id_map.json
PTQ_W6A6=checkpoints/ptq4vit_w6a6_vitb.pt
PTQ_W8A8=checkpoints/ptq4vit_w8a8_vitb.pt
ADA_W3A3=checkpoints/w3a3_checkpoint_opt.pt; ADA_W4A4=checkpoints/w4a4_checkpoint_opt.pt
OUT="${OUT:-runs/placement_full_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"; echo "$OUT" > runs/last_placement_full_dir.txt
MINFREE="${MINFREE:-3500}"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }
qf(){ case "$1" in w3a3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $ADA_W3A3";; w4a4) echo "--quant --quant_w_bit 4 --quant_a_bit 4 --adalog_checkpoint $ADA_W4A4";;
                   w6a6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6";; w8a8) echo "--quant --quant_w_bit 8 --quant_a_bit 8 --ptq_checkpoint $PTQ_W8A8";; esac; }
base(){ echo "--data $DATA --data_corruption $DATA_C --level 5 --imagenet_5k_eval --batch_size 64 --workers 6 --corruption all --seed $2 --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $(qf $1) --algorithm no_adapt"; }
common(){ echo "--quar_local_enable --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag --recal_cumulative"; }
post(){ echo "--quar_local_scope block8_9_10_11_all --quar_local_stats_file stats/qac_stats_$1_all_layers_seed0.pt --quar_local_mode folded_quantizer --quar_local_fold_type output_only"; }
resid(){ echo "--quar_local_scope resid_block8_9_10_11 --quar_local_stats_file stats/resid_stats_$1_blocks_seed0.pt --quar_local_mode explicit"; }
statscmd(){ case "$1" in w3a3) echo "$PY -u tools/stats/build_resid_stats.py --data $DATA --adalog_checkpoint $ADA_W3A3 --w 3 --a 3 --output stats/resid_stats_w3a3_blocks_seed0.pt --seed 0 --max_batches 200";;
                       w8a8) echo "$PY -u tools/stats/build_resid_stats.py --data $DATA --ptq_checkpoint $PTQ_W8A8 --output stats/resid_stats_w8a8_blocks_seed0.pt --seed 0 --max_batches 200";; esac; }
jobcmd(){ local kind=$1 bw=$2 s=$3; case $kind in
  stats) statscmd $bw;;
  post)  echo "$PY -u main.py $(base $bw $s) $(common) $(post $bw) --output $OUT/post_${bw}_s$s --tag _post_${bw}_s$s";;
  resid) echo "$PY -u main.py $(base $bw $s) $(common) $(resid $bw) --output $OUT/resid_${bw}_s$s --tag _resid_${bw}_s$s";; esac; }
done_already(){ local kind=$1 bw=$2 s=$3; case $kind in
  stats) [ -f stats/resid_stats_${bw}_blocks_seed0.pt ];;
  *) [ -f "$OUT/${kind}_${bw}_s$s.log" ] && [ "$(grep -c 'mean acc of corruption' "$OUT/${kind}_${bw}_s$s.log")" -ge 15 ];; esac; }
ready(){ local kind=$1 bw=$2; [ $kind != resid ] || [ -f stats/resid_stats_${bw}_blocks_seed0.pt ]; }
pick_gpu(){ nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' -v m=$MINFREE '{free=$3-$2; if(free>=m) print free, $1}' | sort -rn | while read free g; do
  lock="$OUT/gpu$g.pid"; if [ -f "$lock" ] && kill -0 "$(cat $lock)" 2>/dev/null; then continue; fi; rm -f "$lock"; echo $g; break; done; }
QUEUE=("stats w3a3 0" "stats w8a8 0" "post w4a4 0" "post w8a8 0" "post w3a3 1" "post w6a6 1" "post w4a4 1" "post w8a8 1" "post w3a3 2" "post w6a6 2" "post w4a4 2" "post w8a8 2"
       "resid w3a3 0" "resid w8a8 0" "resid w3a3 1" "resid w8a8 1" "resid w3a3 2" "resid w8a8 2")
log "OUT=$OUT | ${#QUEUE[@]} jobs | MINFREE=$MINFREE MiB, one job per GPU"
pending=("${QUEUE[@]}")
while [ ${#pending[@]} -gt 0 ]; do
  launched=0; rest=()
  for job in "${pending[@]}"; do set -- $job; kind=$1; bw=$2; s=$3
    if done_already $kind $bw $s; then log "[skip] $kind $bw s$s already done"; continue; fi
    if [ $launched -eq 0 ] && ready $kind $bw; then g=$(pick_gpu)
      if [ -n "$g" ]; then tag="${kind}_${bw}_s$s"; mkdir -p "$OUT/$tag"; CMD="CUDA_VISIBLE_DEVICES=$g $(jobcmd $kind $bw $s)"; echo "$CMD" > "$OUT/cmd_$tag.txt"
        bash -c "$CMD" > "$OUT/$tag.log" 2>&1 & echo $! > "$OUT/gpu$g.pid"; log "[start] $tag on gpu $g pid $!"; launched=1; sleep 20; continue; fi; fi
    rest+=("$job"); done
  pending=("${rest[@]}"); [ ${#pending[@]} -gt 0 ] && [ $launched -eq 0 ] && sleep 30
done
wait; log "ALL DONE"
log "=== results (15-corruption mean top-1) ==="
for kind in post resid; do for bw in w3a3 w4a4 w6a6 w8a8; do for s in 0 1 2; do f="$OUT/${kind}_${bw}_s$s.log"; [ -f "$f" ] && log "$kind $bw s$s: $(grep 'mean acc of corruption' $f | tail -1 | grep -o '[0-9.]*$') ($(grep -c 'mean acc of corruption' $f)/15)"; done; done; done
