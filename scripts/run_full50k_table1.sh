#!/usr/bin/env bash
# =============================================================================
# Full-ImageNet-C replication of the paper's Table 1 grid at seed 0.
#   9 configs x 4 bit-widths = 36 cells.
#   configs: No Adapt, T3A, FOA, ZOA, NEO, FOZO, Final-only, Local-only, QuAR
#   bits   : W3A3, W4A4, W6A6, W8A8
# Each cell = main.py on the FULL ImageNet-C validation set (all 50000 img/corr),
# all 15 corruptions, severity 5, seed 0, batch_size 64, episodic reset between
# corruptions.  PLAIN full eval (NO SAR, NO 5k subset).
#
# FULL-50k mechanism: we simply OMIT --imagenet_5k_eval (and its indices/class-map
# flags).  Confirmed in dataset/selectedRotateImageFolder.py: the 5k RobustBench
# subset is applied ONLY inside `if use_imagenet_5k:`; without the flag control
# falls through to the full ImageFolder loader (50k/corr) and reset behaviour is
# unchanged = episodic (main.py ~L1016).
#
# Canonical per-method/per-bit configs are generalised from run_noniid_3seed_w6a6.sh
# (CFG[] + ours_cum()), with SAR flags removed.  ZOA lr, FOZO stats, FOA stats and
# the Final/Local ablation branches follow the --algorithm zoa launchers
# (run_table1_w*.sh, run_seed_sweep_t1t2.sh) and run_cumulative_appendix.sh.
# See the "CONFIG NOTES" block at the bottom.
#
# Structure mirrors run_noniid_3seed_w6a6.sh: flock work-queue, one worker/GPU.
# Resumable: OUT can be overridden to an existing dir; completed cells (log with
# 15 "Top-1 Accuracy" lines) are skipped.
#
# DOES NOT LAUNCH AUTOMATICALLY on source; run with:  bash run_full50k_table1.sh
# Optional env: GPUS="0 1 2 3"   OUT=runs/full50k_table1_<stamp>   (to resume)
# =============================================================================
set -uo pipefail
PY="${PY:-python3}"
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:?set DATA to the ImageNet root (…/Data/CLS-LOC)}"
DATA_C="${DATA_C:?set DATA_C to the ImageNet-C root}"
# ---- per-bit source checkpoints ----
PTQ_W6A6=$QUAR/checkpoints/ptq4vit_w6a6_vitb.pt
PTQ_W8A8=$QUAR/checkpoints/ptq4vit_w8a8_vitb.pt
ADALOG_W3A3=$QUAR/checkpoints/w3a3_checkpoint_opt.pt
ADALOG_W4A4=$QUAR/checkpoints/w4a4_checkpoint_opt.pt
# ---- shared source stats ----
FOA_STATS=$QUAR/foa_stats          # FOA picks train_info_{mean,std}_<arch>_w{bit}_p3.pt internally
FOZO_STATS=$QUAR/stats/train_stats_2.pt  # single generic FOZO stats file for ALL bit-widths (see notes)

OUT="${OUT:-$QUAR/runs/full50k_table1_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"; echo "$OUT" > "$QUAR/runs/last_full50k_table1_dir.txt"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

# ---- base flags (seed 0, FULL 50k => NO --imagenet_5k_eval; episodic) ----
BASE="--data $DATA --data_corruption $DATA_C --level 5 --corruption all --seed 0 --batch_size 64 --workers 8"

# ---- per-bit quant flags ----
qf(){ case "$1" in
  w3a3) echo "--quant --quant_w_bit 3 --quant_a_bit 3 --adalog_checkpoint $ADALOG_W3A3";;
  w4a4) echo "--quant --quant_w_bit 4 --quant_a_bit 4 --adalog_checkpoint $ADALOG_W4A4";;
  w6a6) echo "--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint $PTQ_W6A6";;
  w8a8) echo "--quant --quant_w_bit 8 --quant_a_bit 8 --ptq_checkpoint $PTQ_W8A8";;
  *) echo "BADBW"; return 1;;
esac; }

# ---- QuAR branch builders (per bit); canonical = cumulative (--recal_cumulative) ----
# QAC (local) branch: scope block8_9_10_11_all, alpha 0.5  (== ours_cum() QAC part)
qac_branch(){ local bw=$1; echo "--quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all \
  --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 \
  --quar_local_stats_file $QUAR/stats/qac_stats_${bw}_all_layers_seed0.pt \
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag"; }
# CLS (global) branch: post_ln mean_std, alpha 1.0  (== ours_cum() CLS part)
cls_branch(){ local bw=$1; echo "--quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std \
  --quar_global_alpha 1.0 --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 \
  --quar_global_stats_file $QUAR/stats/cls_stats_${bw}_seed0.pt --quar_global_target_ema_alpha 0.05"; }

# ---- method flag builder: <method> <bw> ----
mflags(){ local m=$1 bw=$2; case "$m" in
  noadapt) echo "--algorithm no_adapt";;
  neo)     echo "--algorithm no_adapt --neo_enable";;
  t3a)     echo "--algorithm t3a --t3a_filter_k 20 --arch vit_base_patch16_224";;
  foa)     echo "--algorithm foa --popsize 1 --foa_stats_dir $FOA_STATS";;
  zoa)     echo "--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 \
             --official_zoa_lr_eps 0.0002 --zoa_layer_balance none \
             --official_zoa_stats_file $QUAR/zoa_stats/${bw}.pt";;
  fozo)    echo "--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 --fozo_stats_file $FOZO_STATS";;
  lgar)    echo "--algorithm no_adapt $(qac_branch "$bw") --recal_cumulative $(cls_branch "$bw")";;
  local)   echo "--algorithm no_adapt $(qac_branch "$bw") --recal_cumulative";;
  final)   echo "--algorithm no_adapt $(cls_branch "$bw") --recal_cumulative";;
  *) echo "BADMETHOD"; return 1;;
esac; }

METHODS="noadapt t3a foa zoa neo fozo final local lgar"
BITS="w3a3 w4a4 w6a6 w8a8"

# ---- skip-completed: log exists AND has exactly 15 "Top-1 Accuracy" lines ----
is_done(){ local f=$1; [ -f "$f" ] && [ "$(grep -c 'Top-1 Accuracy' "$f" 2>/dev/null)" -eq 15 ]; }

# ---- run one cell ----
runcmd(){ local gpu=$1 m=$2 bw=$3; local tag="${m}_${bw}"
  if is_done "$OUT/$tag.log"; then log "[$tag g$gpu] SKIP (already 15 corruptions)"; return 0; fi
  mkdir -p "$OUT/$tag"; log "[$tag g$gpu] start"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u $QUAR/main.py $BASE $(qf "$bw") $(mflags "$m" "$bw") \
    --output "$OUT/$tag" --tag "_$tag" >"$OUT/$tag.log" 2>&1 \
    && log "[$tag] done" || log "[$tag] FAIL exit=$?"; }

cd "$QUAR"

# ---- build queue: slowest cells first (zoa/foa @ w3a3/w4a4 = 2-forward + low-bit), then the rest ----
Q="$OUT/queue"; : >"$Q"
for m in zoa foa; do for bw in w3a3 w4a4; do echo "$m $bw" >>"$Q"; done; done
for m in $METHODS; do for bw in $BITS; do
  case "$m $bw" in "zoa w3a3"|"zoa w4a4"|"foa w3a3"|"foa w4a4") continue;; esac
  echo "$m $bw" >>"$Q"
done; done

log "OUT=$OUT | full-50k Table1 grid: $(wc -l <"$Q") cells over GPUs [$GPUS]"

# ---- flock work-queue: one worker per GPU ----
worker(){ local gpu=$1; while :; do
  local line; line=$(flock "$Q.lock" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null")
  [ -z "$line" ] && break
  runcmd "$gpu" $line
done; }
for g in $GPUS; do worker "$g" & done
wait

# ---- aggregate: 9x4 table of 15-corruption mean Top-1 (rows=methods, cols=bit-widths) ----
log "=== full-50k Table1: 15-corruption mean Top-1 (rows=methods, cols=bit-widths) ==="
$PY - "$OUT" <<'PYEOF'
import re, sys, os
OUT = sys.argv[1]
METHODS = ["noadapt","t3a","foa","zoa","neo","fozo","final","local","lgar"]
BITS = ["w3a3","w4a4","w6a6","w8a8"]
def m15(tag):
    f = os.path.join(OUT, tag + ".log")
    if not os.path.exists(f):
        return None
    a = [float(x) for x in re.findall(r'Top-1 Accuracy: ([0-9.]+)', open(f).read())]
    return sum(a) / len(a) if len(a) == 15 else None
hdr = f"{'method':<10}" + "".join(f"{b:>10}" for b in BITS)
print(hdr); print("-" * len(hdr))
for m in METHODS:
    row = f"{m:<10}"
    for b in BITS:
        v = m15(f"{m}_{b}")
        row += f"{v:>10.2f}" if v is not None else f"{'--':>10}"
    print(row)
PYEOF
log "ALL DONE: $OUT"

# =============================================================================
# CONFIG NOTES
# -----------------------------------------------------------------------------
# * ZOA lr: the --algorithm zoa path uses --official_zoa_lr_eps 0.0002 for
#   ALL 4 bits (run_table1_w{3a3,4a4,6a6,8a8}.sh, run_seed_sweep_t1t2.sh). The
#   Stats: zoa_stats/w{bit}.pt.
# * FOZO stats: ONE generic file train_stats_2.pt for every bit-width. Every FOZO
#   launcher (incl. run_table1_w{3a3,4a4,6a6,8a8}.sh that produced tab:main's FOZO
#   row) uses --fozo_stats_file $QUAR/stats/train_stats_2.pt. There is NO per-bit FOZO
#   stats file (train_info_*_w{bit}_p3.pt live in foa_stats/ and belong to FOA).
# * FOA: --foa_stats_dir foa_stats only; foa.py builds the per-bit filename from
#   arch(vit_base_patch16_224)+w/a bits+num_prompts(3), matching the _p3 files.
#   popsize=1 as in run_noniid_3seed_w6a6.sh (popsize=28 is FOA's recommended setting, tab:foa28).
# * QuAR / Local-only / Final-only all use --recal_cumulative (canonical QuAR is
#   cumulative; --recal_cumulative also drives the CLS branch, cls_shift.py:245),
#   so all three form one consistent cumulative family.
# =============================================================================
