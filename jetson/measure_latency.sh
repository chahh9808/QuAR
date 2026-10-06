#!/usr/bin/env bash
# Per-image forward latency of every Table 1 method, measured on the deployment device.
#
# What this measures, and what it does not. The quantized graph here is SIMULATED: weights
# and activations are fake-quantized in PyTorch, and no integer kernel is involved. So the
# absolute milliseconds are not an INT8 deployment figure and must not be reported as one.
# What survives the simulation is the RATIO between methods, which is what the paper's
# latency argument actually uses: one extra forward pass costs about one forward either way,
# and the cost of 16 per-channel affine hooks relative to a forward is exactly the thing a
# device with a different compute-to-bandwidth balance can change. That ratio is the result.
#
# Protocol, matching Appendix app:cost so the two platforms are comparable:
#   idle device, one measurement at a time, per-image forward time,
#   median over batches, median over REPS repetitions, warmup batches discarded.
# On Jetson two extra things are required or the numbers scatter: a fixed power mode and
# locked clocks. Both are attempted below and reported either way.
#
# QuAR runs its deployed configuration, scope block8_9_10_11_all with lambda 0.5, the 16
# interfaces the paper describes.
set -uo pipefail
# Repository root. Derived from this script's own location so the checkout can live anywhere
# on the device; override with F=... if you keep the code and the data apart.
F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-$F/jetson/env/bin/python}"
DATA="${DATA:-${DATA:?set DATA to the ImageNet root}}"
DATA_C="${DATA_C:-${DATA_C:?set DATA_C to the ImageNet-C root}}"
IDS=$F/robustbench/data/imagenet_test_image_ids.txt
CMAP=$F/robustbench/data/imagenet_class_to_id_map.json
PTQ="${PTQ:-$F/checkpoints/ptq4vit_w6a6_vitb.pt}"
QAC_STATS="${QAC_STATS:-$F/stats/qac_stats_w6a6_all_layers_seed0.pt}"
CLS_STATS="${CLS_STATS:-$F/stats/cls_stats_w6a6_seed0.pt}"
BS="${BS:-64}"                 # retried at batch 16 if it does not fit
QUAR_CHUNK="${QUAR_CHUNK:-0}"  # --quar_local_chunk: slices the local correction in place.
                               # Bit-identical; removes the fc2 interface's peak-memory cost.
                               # 0 reproduces the committed behaviour.
NBATCH="${NBATCH:-50}"         # measured batches per repetition
REPS="${REPS:-5}"              # repetitions, median over them
WARMUP="${WARMUP:-10}"         # extra batches on top of the two main.py already treats as startup
OUT="${OUT:-$F/runs/jetson_latency_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/driver.log"; }

# ---------- device state ----------
say "output: $OUT"
say "arch $(uname -m); $( [ -f /etc/nv_tegra_release ] && head -1 /etc/nv_tegra_release || echo 'no L4T file' )"
if command -v nvpmodel >/dev/null 2>&1; then
  if sudo -n true 2>/dev/null; then
    sudo nvpmodel -m "${NVPMODEL:-0}" >/dev/null 2>&1 && say "power mode set to ${NVPMODEL:-0} (max)" \
      || say "WARNING: nvpmodel failed; latency will vary with DVFS"
  else
    say "WARNING: no passwordless sudo, cannot set power mode. Run 'sudo nvpmodel -m 0' first."
  fi
  say "power mode now: $(nvpmodel -q 2>/dev/null | grep -i 'NV Power Mode' | head -1)"
fi
if command -v jetson_clocks >/dev/null 2>&1 || [ -x /usr/bin/jetson_clocks ]; then
  sudo -n jetson_clocks >/dev/null 2>&1 && say "clocks locked to max" \
    || say "WARNING: jetson_clocks not applied; expect run-to-run scatter"
fi
$PY - <<'PY' 2>&1 | tee -a "$OUT/env.txt"
import torch, timm, sys, platform
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| avail", torch.cuda.is_available(), "| timm", timm.__version__, "|", platform.machine())
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties(0)
    print("device", p.name, "| total mem %.2f GiB" % (p.total_memory/1024**3))
PY
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { say "REFUSE: torch cannot see the GPU"; exit 1; }

# ---------- method flags: verbatim from the launchers that produced the paper ----------
flags_for(){ case "$1" in
  NoAdapt) echo "--algorithm no_adapt";;
  T3A)     echo "--algorithm t3a --t3a_filter_k 20";;
  FOA)     echo "--algorithm foa --popsize 1 --foa_stats_dir $F/foa_stats";;
  ZOA)     echo "--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 \
--official_zoa_lr_eps 0.0002 --zoa_layer_balance none --official_zoa_stats_file $F/zoa_stats/w6a6.pt";;
  NEO)     echo "--algorithm no_adapt --neo_enable";;
  FOZO)    echo "--algorithm fozo --zo_eps 0.5 --lr 0.08 --n_spsa 1 --fozo_stats_file $F/stats/train_stats_2.pt";;
  TENT)    echo "--algorithm tent";;
  EATA)    echo "--algorithm eata";;
  # EATA without its sample selection, for the cost question only. EATA skips the backward
  # pass entirely when no sample clears its entropy and diversity filters
  # (sar_official/eata.py "if x[ids1][ids2].size(0) != 0"), so at batch 1 a rejected sample
  # costs one forward and nothing else and the measured overhead is data dependent: only
  # 18% of samples cleared the filters there, so the median landed on the rejected-sample
  # cost, 1.329x, against 2.015x when every sample adapts. ln(1000) is the maximum entropy
  # of a 1000-way softmax, so every sample clears the first filter; cosine similarity
  # cannot exceed 1, so 2.0 clears the second. This also inflates the entropy reweighting
  # coeff = exp(e_margin - H), so report latency and memory from this row, not accuracy.
  # At batch 16 selection no longer distorts the median (median 1613.9 ms against mean
  # 1635.3, and the median sits on the adapting mode), so the plain EATA row is fine there.
  EATA_nofilter) echo "--algorithm eata --eata_e_margin 6.907755 --eata_d_margin 2.0";;
  SAR)     echo "--algorithm sar";;
  SPA)     echo "--algorithm spa";;
  QuAR)    echo "--algorithm no_adapt --quar_local_enable --quar_local_mode explicit --quar_local_scope block8_9_10_11_all \
--quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 0.5 --quar_local_stats_file $QAC_STATS \
--quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \
--quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std --quar_global_alpha 1.0 \
--quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 --quar_global_stats_file $CLS_STATS \
--quar_global_target_ema_alpha 0.05 --quar_local_chunk $QUAR_CHUNK";;
esac; }
METHODS="${METHODS:-NoAdapt T3A FOA ZOA NEO FOZO QuAR TENT EATA SAR SPA}"

run_one(){   # method, batch, nbatch, tag -> prints "median_per_image_ms peak_mb" or "FAIL"
  local m=$1 bs=$2 nb=$3 tag=$4
  local d="$OUT/${tag}"
  mkdir -p "$d"
  CUDA_VISIBLE_DEVICES=0 $PY -u "$F/main.py" \
    --data "$DATA" --data_corruption "$DATA_C" --level 5 --imagenet_5k_eval \
    --corruption gaussian_noise --seed 0 --batch_size "$bs" --workers 2 \
    --imagenet_5k_indices_file "$IDS" --imagenet_class_map_file "$CMAP" \
    --quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint "$PTQ" \
    --arch vit_base_patch16_224 --latency_profile --mem_profile \
    --max_eval_batches "$nb" $(flags_for "$m") \
    --output "$d" --tag "_$tag" > "$d/run.log" 2>&1
  local lat peak
  lat=$(grep -oE "median_per_image_ms=[0-9.]+" "$d/run.log" | tail -1 | cut -d= -f2)
  peak=$(grep -oE "PEAK_GPU_MEM_MB=[0-9.]+" "$d/run.log" | tail -1 | cut -d= -f2)
  if [ -n "$lat" ]; then echo "OK $lat ${peak:-nan}"; return 0; fi
  # An OOM is a result here, not an error: it is the measured form of "this method does not
  # fit the device". Anything else is a broken run and must not be reported as OOM.
  # On Tegra the allocator does not report an OOM the way it does on a discrete GPU. The Tegra
  # allocator refuses first ("NvMapMemHandleAlloc failed: error 12", errno ENOMEM), and then
  # torch's own OOM path tries to read device memory through NVML, which Tegra does not fully
  # implement, so the process dies on an internal assert in CUDACachingAllocator instead of
  # raising "CUDA out of memory". Both signatures mean the same thing as an OOM here, and
  # without them a method that simply does not fit is reported as a broken run, which is the
  # opposite of the measurement this table is trying to make.
  if grep -qiE "out of memory|CUDA error: out of memory|Killed|NvMapMemHandleAlloc failed: error 12|NvMapMemAllocInternalTagged failed: error 12|NVML_SUCCESS == r INTERNAL ASSERT FAILED" "$d/run.log" 2>/dev/null; then
    echo "OOM"; else echo "FAIL"; fi
  return 0
}

# ---------- measure at the target batch, and record what does not fit ----------
# The batch is FIXED, not probed to the lowest common denominator: the paper's tables are at
# batch 64, and a method that cannot run there has told us something. For those, the ladder
# below finds the largest batch it CAN run, so the row reads "OOM at 64, runs at 8" rather
# than just failing. That is the comparison the deployment argument wants.
LADDER="${LADDER:-16}"        # the paper reports batch 64 and, for the backprop-based methods, batch 16
say "target batch $BS; methods that OOM there are retried at: $LADDER"
: > "$OUT/raw.tsv"
: > "$OUT/status.tsv"

for m in $METHODS; do
  # --- does it run at the target batch? ---
  probe=$(run_one "$m" "$BS" 6 "probe_${m}_bs${BS}")
  st=${probe%% *}
  if [ "$st" = FAIL ]; then
    say "[$m] broken at bs=$BS (not an OOM); see $OUT/probe_${m}_bs${BS}/run.log"
    printf "%s\tBROKEN\t%s\t\t\n" "$m" "$BS" >> "$OUT/status.tsv"; continue
  fi
  use_bs=$BS; note=""
  if [ "$st" = OOM ]; then
    say "[$m] OOM at bs=$BS, walking down the ladder"
    use_bs=""
    for cand in $LADDER; do
      r=$(run_one "$m" "$cand" 6 "probe_${m}_bs${cand}")
      case "${r%% *}" in
        OK)   use_bs=$cand; note="OOM at $BS"; say "   runs at bs=$cand"; break;;
        OOM)  say "   OOM at bs=$cand";;
        FAIL) say "   broken at bs=$cand"; break;;
      esac
    done
    if [ -z "$use_bs" ]; then
      say "[$m] does not fit this device at any batch down to 1"
      printf "%s\tOOM_ALL\t%s\t\t\n" "$m" "$BS" >> "$OUT/status.tsv"; continue
    fi
  fi

  # --- measure it ---
  for r in $(seq 1 "$REPS"); do
    out=$(run_one "$m" "$use_bs" $((WARMUP+NBATCH)) "m_${m}_r${r}" || true)
    case "${out%% *}" in
      OK)  set -- $out; lat=$2; peak=$3
           printf "%s\t%s\t%s\t%s\t%s\n" "$m" "$r" "$lat" "$peak" "$use_bs" >> "$OUT/raw.tsv"
           say "[$m rep$r bs=$use_bs] ${lat} ms/img, peak ${peak} MB";;
      OOM) say "[$m rep$r] OOM at bs=$use_bs after it had fitted; device state may have drifted";;
      *)   say "[$m rep$r] FAILED, see $OUT/m_${m}_r${r}/run.log";;
    esac
  done
  printf "%s\tOK\t%s\t%s\n" "$m" "$use_bs" "$note" >> "$OUT/status.tsv"
done

# ---------- report ----------
say "done; summarising"
$PY - "$OUT/raw.tsv" "$OUT/status.tsv" "$BS" <<'PY' | tee -a "$OUT/results.txt"
import sys, statistics as st, collections
raw, statf, target = sys.argv[1], sys.argv[2], sys.argv[3]
rows=collections.defaultdict(list); peaks=collections.defaultdict(list); bss={}
for ln in open(raw):
    p=ln.rstrip('\n').split('\t')
    if len(p) < 3: continue
    m=p[0]
    try: rows[m].append(float(p[2]))
    except ValueError: continue
    if len(p)>3:
        try: peaks[m].append(float(p[3]))
        except ValueError: pass
    if len(p)>4: bss[m]=p[4]
status={}
for ln in open(statf):
    p=ln.rstrip('\n').split('\t')
    if p and p[0]: status[p[0]]=(p[1] if len(p)>1 else '', p[3] if len(p)>3 else '')

ORDER=["NoAdapt","T3A","NEO","QuAR","FOA","ZOA","FOZO","TENT","EATA","SAR","SPA"]
base=st.median(rows['NoAdapt']) if rows.get('NoAdapt') else None
print("\n  Per-image forward latency on this device, SIMULATED-quantization W6A6 ViT-B")
print("  target batch %s; median over reps of the per-batch median" % target)
print("  absolute ms are not an INT8 deployment figure; the ratios are the result\n")
print("  %-8s %-14s %13s %10s %11s" % ("method","status","ms/img","x NoAdapt","peak MB"))
for m in ORDER:
    stt = status.get(m, ('',''))[0]
    if stt == 'OOM_ALL':
        print("  %-8s %-14s %13s %10s %11s" % (m,"does not fit","-","-","-")); continue
    if stt == 'BROKEN':
        print("  %-8s %-14s %13s %10s %11s" % (m,"run failed","-","-","-")); continue
    if not rows.get(m):
        print("  %-8s %-14s %13s %10s %11s" % (m,"no data","-","-","-")); continue
    v=st.median(rows[m]); sd=st.stdev(rows[m]) if len(rows[m])>1 else 0.0
    used=bss.get(m,target)
    lab = "ok" if used==target else ("ok at bs=%s" % used)
    ratio=("%.2fx" % (v/base)) if (base and used==target) else ("%.2fx*" % (v/base) if base else "-")
    pk=("%.0f" % st.median(peaks[m])) if peaks.get(m) else "-"
    print("  %-8s %-14s %9.3f+-%.3f %10s %11s" % (m,lab,v,sd,ratio,pk))
print("\n  * ratio across different batch sizes; not directly comparable")
print("\n  server reference (TITAN RTX, Appendix app:cost / tab:backprop_acc), batch 64:")
print("    NoAdapt 6.0  T3A 8.2  NEO 6.0  QuAR 6.6  FOA 13.0  ZOA 12.3  FOZO 12.3")
print("    TENT 11.3 (9159 MB)  EATA 11.4 (9160)  SAR 22.2 (9160)  SPA 31.2 (23450)")
PY
say "results in $OUT/results.txt and $OUT/raw.tsv"
