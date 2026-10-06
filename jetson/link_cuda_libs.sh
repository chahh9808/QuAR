#!/usr/bin/env bash
# Make the venv's pip-installed CUDA 13 libraries visible to the torch wheel.
#
# Why this is needed. The jetson-ai-lab sbsa/cu130 torch wheel is built against a CUDA
# toolkit that is expected to be on the device, but JetPack 7.2 installs only the driver
# (libcuda.so.1); /usr/local/cuda does not exist and there is no libcudart. Installing the
# CUDA toolkit through apt needs root, so the runtime libraries are installed as ordinary
# wheels inside jetson/env instead.
#
# torch can preload most of them itself, but libcudss.so.0 -- a DT_NEEDED of
# libtorch_cuda.so on this build -- is not in torch's preload table, so `import torch`
# still fails. Rather than depend on LD_LIBRARY_PATH being exported by whoever runs the
# scripts, symlink every wheel-provided library into torch/lib, which libtorch_cuda.so
# already searches: its RUNPATH is $ORIGIN. The scripts then work unmodified.
#
# Idempotent: safe to re-run after any pip install that changes the CUDA wheels.
set -uo pipefail
F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-$F/jetson/env/bin/python}"
[ -x "$PY" ] || { echo "[link] ERROR: no python at $PY" >&2; exit 1; }

SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
TORCHLIB="$SITE/torch/lib"
[ -d "$TORCHLIB" ] || { echo "[link] ERROR: no torch/lib at $TORCHLIB" >&2; exit 1; }

# The CUDA 13 wheels put their libraries under nvidia/cu13/lib and nvidia/cudnn/lib, and the
# ARM BLAS/LAPACK backend that libtorch_cpu.so links against (NVPL) ships in its own tree, so
# collect every directory under site-packages that actually holds an NVIDIA shared object
# rather than hardcoding the layout, which has changed once already between CUDA 12 and 13.
mapfile -t LIBDIRS < <(
  {
    find "$SITE/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null
    find "$SITE" -maxdepth 4 \( -name 'libnvpl*.so*' -o -name 'libcupti*.so*' \) \
         -printf '%h\n' 2>/dev/null
  } | sort -u
)
[ "${#LIBDIRS[@]}" -gt 0 ] || { echo "[link] ERROR: no NVIDIA library dirs under $SITE" >&2; exit 1; }
echo "[link] source dirs: ${#LIBDIRS[@]}"

n=0
for d in "${LIBDIRS[@]}"; do
  # Only the shared objects. Skip static archives and headers, which are not loaded and
  # would just clutter torch/lib.
  while IFS= read -r so; do
    b="$(basename "$so")"
    # Never shadow a library torch ships itself.
    if [ -e "$TORCHLIB/$b" ] && [ ! -L "$TORCHLIB/$b" ]; then
      echo "[link] skip $b (torch ships its own)"; continue
    fi
    ln -sfn "$so" "$TORCHLIB/$b" && n=$((n+1))
  done < <(find "$d" -maxdepth 1 -name '*.so*' -type f 2>/dev/null)
done
echo "[link] linked $n libraries into $TORCHLIB"

echo "[link] verifying unresolved DT_NEEDED of libtorch_cuda.so"
miss=$(cd "$TORCHLIB" && LD_LIBRARY_PATH="$TORCHLIB" ldd libtorch_cuda.so 2>/dev/null | awk '/not found/{print $1}')
if [ -n "$miss" ]; then echo "[link] STILL MISSING:"; echo "$miss" | sed 's/^/   /'; exit 1; fi
echo "[link] all DT_NEEDED resolved"
