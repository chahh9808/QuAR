#!/usr/bin/env bash
# Cumulative-estimator runs for the remaining appendix Ours/Local/Final cells (NEEDS-NEW set).
# Recipe = published per-cell flags + --recal_cumulative (global; affects QAC stream + CLS). lambda=0.5,
# seed0, 5k, bs64, episodic. Baselines are estimator-independent (not re-run). 32 cells:
#   prec  : Final/Local x {fp,w8a8,w6a6,w4a4,w3a3}            (tab:precision, tab:main-ablation, tab:t2pc Final, tab:mismatch, tab:placement)
#   fc2   : 10 depth scopes (valid=w/o-fc2 + all=with-fc2), W6A6 Local-only  (tab:fc2depth)
#   cq    : off-protocol Ours: adalog@{w6a6,w8a8} + ptq4vit@{w3a3,w4a4}      (tab:crossquant)
#   heldc : Ours x {w3a3,w4a4,w6a6,w8a8}, held-out 4 corruptions            (tab:heldc_full)
#   noshift: Ours x {w3a3,w4a4,w6a6,w8a8}, clean (--corruption original)    (tab:noshift)
set -uo pipefail
PY="${PY:-python3}"; QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root}"; DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt; CMAP=$QUAR/robustbench/data/imagenet_class_to_id_map.json
PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
PTQ_W8A8=$QUAR/checkpoints/ptq4vit_w8a8_vitb.pt
ADALOG_W4A4=$QUAR/checkpoints/w4a4_checkpoint_opt.pt; ADALOG_W3A3=$QUAR/checkpoints/w3a3_checkpoint_opt.pt
HELDC="gaussian_blur,saturate,spatter,speckle_noise"
OUT=$QUAR/runs/cum_appendix_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"; echo "$OUT">"$QUAR/runs/last_cum_appendix_dir.txt"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
# Seed 0 is what the appendix cells were run at. tab:main's Final-only and Local-only rows are
# 3-seed, so seeds 1 and 2 come from re-running this with SEED=1 and SEED=2. Non-zero seeds get
# their own output tag, leaving the seed-0 paths exactly as they were.
SEED="${SEED:-0}"
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
qf_std(){ case "$1" in fp) echo "";; w8a8) echo "--quant --quant_w_bit 8 --quant_a_bit 8 --ptq_checkpoint $PTQ_W8A8";;
  w6a6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6";;
  w4a4) echo "--quant --quant_w_bit 4 --quant_a_bit 4 --adalog_checkpoint $ADALOG_W4A4";;
  w3a3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $ADALOG_W3A3";; esac; }
qacs(){ [ "$1" = fp ] && echo "$QUAR/stats/qac_stats_fp_all_layers_seed0.pt" || echo "$QUAR/stats/qac_stats_$1_all_layers_seed0.pt"; }
clss(){ [ "$1" = fp ] && echo "$QUAR/stats/cls_stats_fp_seed0.pt" || echo "$QUAR/stats/cls_stats_$1_seed0.pt"; }
QAC(){ echo "--quar_local_enable --quar_local_mode explicit --quar_local_scope $1 --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $2 --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag"; }
CLS(){ echo "--quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $1 --quar_global_target_ema_alpha 0.05"; }
CUM="--recal_cumulative"
runcell(){ local gpu=$1 grp=$2 a=$3 b=${4:-}; local tag base meth corr="all" q
  case "$grp" in
    prec)    tag="prec_${a}_${b}"; q="$(qf_std $a)"
             [ "$b" = FINAL ] && meth="--algorithm no_adapt $(CLS $(clss $a))" || meth="--algorithm no_adapt $(QAC block8_9_10_11_all $(qacs $a))";;
    fc2)     tag="fc2_${a}"; q="$(qf_std w6a6)"; meth="--algorithm no_adapt $(QAC $a $(qacs w6a6))";;
    cq)      tag="cq_${a}_${b}"  # a=bw b=quant
             case "${a}_${b}" in
               w6a6_adalog)  q="--quant --quant_w_bit 6 --quant_a_bit 6 --adalog_checkpoint $QUAR/checkpoints/w6a6_checkpoint_opt.pt"; qfile="$QUAR/stats/qac_stats_w6a6_adalog_all_layers_seed0.pt"; cfile="$QUAR/stats/cls_stats_w6a6_adalog_seed0.pt";;
               w8a8_adalog)  q="--quant --quant_w_bit 8 --quant_a_bit 8 --adalog_checkpoint $QUAR/checkpoints/w8a8_checkpoint_opt.pt"; qfile="$QUAR/stats/qac_stats_w8a8_adalog_all_layers_seed0.pt"; cfile="$QUAR/stats/cls_stats_w8a8_adalog_seed0.pt";;
               w3a3_ptq4vit) q="--quant --quant_w_bit 3 --quant_a_bit 3 --ptq_checkpoint $QUAR/checkpoints/ptq4vit_w3a3_vitb.pt"; qfile="$QUAR/stats/qac_stats_w3a3_ptq4vit_all_layers_seed0.pt"; cfile="$QUAR/stats/cls_stats_w3a3_ptq4vit_seed0.pt";;
               w4a4_ptq4vit) q="--quant --quant_w_bit 4 --quant_a_bit 4 --ptq_checkpoint $QUAR/checkpoints/ptq4vit_w4a4_vitb.pt"; qfile="$QUAR/stats/qac_stats_w4a4_ptq4vit_all_layers_seed0.pt"; cfile="$QUAR/stats/cls_stats_w4a4_ptq4vit_seed0.pt";;
             esac
             meth="--algorithm no_adapt $(QAC block8_9_10_11_all $qfile) $(CLS $cfile)";;
    heldc)   tag="heldc_${a}"; q="$(qf_std $a)"; corr="$HELDC"; meth="--algorithm no_adapt $(QAC block8_9_10_11_all $(qacs $a)) $(CLS $(clss $a))";;
    noshift) tag="noshift_${a}"; q="$(qf_std $a)"; corr="original"; meth="--algorithm no_adapt $(QAC block8_9_10_11_all $(qacs $a)) $(CLS $(clss $a))";;
  esac
  [ "$SEED" != 0 ] && tag="${tag}_s${SEED}"
  mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py --data $DATA --data_corruption $DATA_C --level 5 \
    --imagenet_5k_eval --batch_size 64 --workers 8 --corruption $corr --seed "$SEED" \
    --imagenet_5k_indices_file $IDS --imagenet_class_map_file $CMAP $q $meth $CUM \
    --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }
worker(){ local gpu=$1; while :; do
  local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"); [ -z "$line" ] && break
  runcell "$gpu" $line; done; }
cd "$QUAR"
Q="$OUT/queue"; : >"$Q"
for bw in fp w8a8 w6a6 w4a4 w3a3; do for v in FINAL LOCAL; do echo "prec $bw $v" >>"$Q"; done; done
for sc in block11_valid block11_all qba_valid block10_11_all block9_10_11_valid block9_10_11_all block8_9_10_11_valid block8_9_10_11_all all_blocks_valid_no_fc2 all; do echo "fc2 $sc" >>"$Q"; done
echo "cq w6a6 adalog" >>"$Q"; echo "cq w8a8 adalog" >>"$Q"; echo "cq w3a3 ptq4vit" >>"$Q"; echo "cq w4a4 ptq4vit" >>"$Q"
for bw in w3a3 w4a4 w6a6 w8a8; do echo "heldc $bw" >>"$Q"; done
for bw in w3a3 w4a4 w6a6 w8a8; do echo "noshift $bw" >>"$Q"; done
log "OUT=$OUT | $(wc -l <"$Q") cumulative appendix cells on GPUs [$GPUS]"
for g in $GPUS; do worker $g & done
wait
log "=== cumulative appendix results (15-corr/held-out mean top-1) ==="
for f in "$OUT"/*.log; do [ -f "$f" ] || continue; t=$(basename "$f" .log)
  v=$(grep -oE "Top-1 Accuracy: [0-9.]+" "$f" | grep -oE "[0-9.]+$" | awk '{s+=$1;n++} END{if(n)printf "%.2f(n=%d)",s/n,n; else print "NA"}')
  log "  $t = $v"
done
log "ALL DONE: $OUT"
