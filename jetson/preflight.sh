#!/usr/bin/env bash
# GO / NO-GO for this device, in a few minutes, before anyone spends a day on it.
#
# Written because Jetson Nano and Orin Nano are not the same proposition. Orin Nano runs
# JetPack 6 with a recent Python and a CUDA PyTorch, and the measurement stack works. Jetson
# Nano's last supported release is JetPack 4.6, which ships Python 3.6 and tops out near
# torch 1.10; timm 1.0.26 declares Requires-Python >=3.8 and will not install, and the older
# timm that would install has different ViT-B state-dict keys, so the checkpoints stop
# matching. This script reports which situation you are in rather than guessing.
set -uo pipefail
# Repository root. Derived from this script's own location so the checkout can live anywhere
# on the device; override with F=... if you keep the code and the data apart.
F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-$F/jetson/env/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
PTQ="${PTQ:-$F/checkpoints/ptq4vit_w6a6_vitb.pt}"
FAIL=0
ok(){   echo "   PASS  $*"; }
bad(){  echo "   FAIL  $*"; FAIL=1; }
warn(){ echo "   WARN  $*"; }

echo "== device =="
echo "   arch      $(uname -m)"
[ -f /etc/nv_tegra_release ] && echo "   L4T       $(head -1 /etc/nv_tegra_release)" || warn "no /etc/nv_tegra_release"
[ -f /etc/nv_boot_control.conf ] && echo "   board     $(grep -i 'TNSPEC\|COMPATIBLE_SPEC' /etc/nv_boot_control.conf 2>/dev/null | head -1)"
if [ -f /etc/nv_tegra_release ]; then
  free -g 2>/dev/null | awk 'NR==2{printf "   memory    %s GB total, %s GB available (CPU and GPU share this on Tegra)\n",$2,$7}'
else
  free -g 2>/dev/null | awk 'NR==2{printf "   memory    %s GB total, %s GB available (host)\n",$2,$7}'
fi

echo; echo "== python and packages =="
"$PY" - <<'PY'
import sys
v=sys.version_info
print("   python    %d.%d.%d  (%s)" % (v.major, v.minor, v.micro, sys.executable))
print("   VERDICT_PY", "ok" if (v.major,v.minor) >= (3,8) else "too_old_for_timm_1.0.26")
PY
PYOK=$("$PY" -c 'import sys;print("ok" if sys.version_info[:2]>=(3,8) else "old")')
[ "$PYOK" = ok ] && ok "python >= 3.8, timm 1.0.26 can install" \
                 || bad "python < 3.8; timm 1.0.26 requires >=3.8 and an older timm changes the ViT-B keys"

"$PY" - <<'PY'
try:
    import torch
    print("   torch     %s | cuda %s | available %s" % (torch.__version__, torch.version.cuda, torch.cuda.is_available()))
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("   gpu       %s | sm_%d%d | %.2f GiB" % (p.name, p.major, p.minor, p.total_memory/1024**3))
        print("   VERDICT_CUDA ok")
    else:
        print("   VERDICT_CUDA no_gpu")
except Exception as e:
    print("   torch     MISSING (%s)" % e); print("   VERDICT_CUDA no_torch")
PY
CUOK=$("$PY" -c "
try:
    import torch; print('ok' if torch.cuda.is_available() else 'no')
except Exception: print('no')")
[ "$CUOK" = ok ] && ok "torch sees the GPU" || bad "no CUDA torch; latency here would be CPU time"

for pkg in timm cma torchvision; do
  "$PY" - "$pkg" <<'PY'
import sys, importlib
n=sys.argv[1]
try:
    m=importlib.import_module(n)
    print("   %-10s %s" % (n, getattr(m,'__version__','?')))
except Exception as e:
    print("   %-10s MISSING (%s)" % (n, type(e).__name__))
PY
done
TIMM=$("$PY" -c "
try:
    import timm; print(timm.__version__)
except Exception: print('none')")
[ "$TIMM" = "1.0.26" ] && ok "timm matches the server (1.0.26)" \
  || { [ "$TIMM" = none ] && bad "timm not installed" || warn "timm $TIMM != 1.0.26; ViT-B keys may not match the checkpoints"; }

echo; echo "== end-to-end: does a real run start and produce a latency? =="
# Loading the checkpoint with a bare torch.load does NOT work and is not a useful test: it is a
# pickled model whose unpickling needs quant_library's patch of timm's Attention
# (attention_forward) to already be installed. main.py applies that before loading, so the
# honest check is a short run through main.py itself. It needs at least three batches:
# main.py times batches 0 and 1 as startup and only records LATPROF samples from i>=2.
if [ ! -f "$PTQ" ]; then
  bad "checkpoint not found at $PTQ (copy it to the device, or set PTQ=)"
elif [ "$CUOK" != ok ] || [ "$TIMM" = none ]; then
  warn "skipping the end-to-end run until torch and timm are in place"
else
  D=$(mktemp -d); say_log="$D/run.log"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u "$F/main.py" \
    --data "${DATA:-${DATA:?set DATA to the ImageNet root}}" \
    --data_corruption "${DATA_C:-${DATA_C:?set DATA_C to the ImageNet-C root}}" \
    --level 5 --imagenet_5k_eval --corruption gaussian_noise --seed 0 \
    --batch_size "${PROBE_BS:-8}" --workers 2 \
    --imagenet_5k_indices_file "$F/robustbench/data/imagenet_test_image_ids.txt" \
    --imagenet_class_map_file "$F/robustbench/data/imagenet_class_to_id_map.json" \
    --quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint "$PTQ" \
    --arch vit_base_patch16_224 --algorithm no_adapt \
    --latency_profile --mem_profile --max_eval_batches 6 \
    --output "$D" --tag _preflight > "$say_log" 2>&1
  if grep -q "LATPROF" "$say_log"; then
    ok "end-to-end run works: $(grep -oE 'median_per_image_ms=[0-9.]+' "$say_log" | tail -1)"
    grep -oE "PEAK_GPU_MEM_MB=[0-9.]+" "$say_log" | tail -1 | sed 's/^/   /'
  else
    bad "the run did not produce a latency; last lines of its log:"
    tail -6 "$say_log" | sed 's/^/         /'
  fi
  rm -rf "$D"
fi

echo; echo "== verdict =="
if [ "$FAIL" -eq 0 ]; then
  echo "   GO. Run: bash $F/jetson/measure_latency.sh"
  echo "   Lock the device first: sudo nvpmodel -m 0 && sudo jetson_clocks"
else
  echo "   NO-GO. Fix the FAIL lines above before measuring."
  echo "   On a Jetson Nano the python and timm failures are not fixable: JetPack 4.6 is the"
  echo "   last release for that board and ships python 3.6. Measuring ViT-B there would need"
  echo "   an older timm whose ViT-B differs from the checkpoints, so the numbers would not"
  echo "   describe the model in the paper."
fi
exit $FAIL
