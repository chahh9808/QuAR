#!/usr/bin/env bash
# Build the measurement environment on a Jetson (aarch64) inside the project folder.
#
# READ THIS FIRST. On Jetson you cannot get a CUDA-capable PyTorch from conda-forge or
# from PyPI: both ship aarch64 wheels that are CPU-only. CUDA support comes either from
# the JetPack-provided torch already on the device, or from NVIDIA's Jetson wheel index.
# This script prefers the first, because it is the one that is guaranteed to match the
# device's CUDA, cuDNN and driver.
#
# What it does
#   1. Finds the JetPack / L4T release and reports it.
#   2. Looks for a system torch with CUDA. If it finds one, the env is created with
#      access to it and only the pure-Python packages are installed on top.
#   3. If it does not, it stops and tells you what to pass, rather than installing a
#      CPU-only torch that would silently measure the wrong thing.
#
# Versions to match the server environment that produced the paper's numbers:
#   python 3.9.25   torch 2.8.0+cu128   torchvision 0.23.0   timm 1.0.26   numpy 2.0.2
# timm is pinned because the model definitions and state-dict keys must match the
# checkpoints. torch is NOT pinned on device: use whatever JetPack provides, and record it.
set -uo pipefail
# Repository root. Derived from this script's own location so the checkout can live anywhere
# on the device; override with F=... if you keep the code and the data apart.
F="${F:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENVDIR="${ENVDIR:-$F/jetson/env}"        # inside the project, not $HOME
TORCH_INDEX="${TORCH_INDEX:-}"           # optional: NVIDIA Jetson wheel index URL

say(){ echo "[setup] $*"; }
die(){ echo "[setup] ERROR: $*" >&2; exit 1; }

say "arch: $(uname -m)"
[ "$(uname -m)" = "aarch64" ] || say "WARNING: not aarch64, this script targets Jetson"

if [ -f /etc/nv_tegra_release ]; then
  say "L4T: $(head -1 /etc/nv_tegra_release)"
else
  say "no /etc/nv_tegra_release; is this really a Jetson?"
fi
command -v nvpmodel >/dev/null && say "power modes: $(nvpmodel -q 2>/dev/null | tr '\n' ' ' | cut -c1-90)"

# ---- 1. is there a system torch with CUDA? ----
SYS_PY="$(command -v python3)"
SYS_OK=no
if [ -n "$SYS_PY" ]; then
  if "$SYS_PY" - <<'PY' 2>/dev/null
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then SYS_OK=yes; fi
fi
if [ "$SYS_OK" = yes ]; then
  say "system python has a CUDA torch: $("$SYS_PY" -c 'import torch;print(torch.__version__, torch.version.cuda)')"
else
  say "system python has no CUDA-capable torch"
fi

# ---- 2. create the env ----
if command -v conda >/dev/null 2>&1; then
  say "creating conda env at $ENVDIR (python 3.9 to match the server)"
  conda create -y -p "$ENVDIR" python=3.9 || die "conda create failed"
  PY="$ENVDIR/bin/python"
  if [ "$SYS_OK" = yes ]; then
    # expose the JetPack torch to the conda env; conda-forge has no CUDA torch for Tegra
    SITE="$("$SYS_PY" -c 'import site;print(site.getsitepackages()[0])')"
    TARGET="$("$PY" -c 'import site;print(site.getsitepackages()[0])')"
    echo "$SITE" > "$TARGET/00_jetpack_system.pth"
    say "linked JetPack site-packages into the env: $SITE"
  fi
else
  say "conda not found, using venv"
  if [ "$SYS_OK" = yes ]; then
    "$SYS_PY" -m venv --system-site-packages "$ENVDIR" || die "venv failed"
    say "venv created with --system-site-packages so the JetPack torch is visible"
  else
    "$SYS_PY" -m venv "$ENVDIR" || die "venv failed"
  fi
  PY="$ENVDIR/bin/python"
fi

# ---- 3. torch, only if the system did not provide one ----
if [ "$SYS_OK" != yes ]; then
  if [ -z "$TORCH_INDEX" ]; then
    die "No CUDA torch on the device and no TORCH_INDEX given.
     Installing a PyPI aarch64 torch would give you a CPU-only build, and the latency
     numbers would be meaningless. Find the wheel index for your JetPack release in
     NVIDIA's Jetson documentation or on the Jetson AI Lab wheel server, then re-run:
       TORCH_INDEX=<url> bash $0
     Alternatively install JetPack's python3-torch system package and re-run with no args."
  fi
  say "installing torch from $TORCH_INDEX"
  "$PY" -m pip install --extra-index-url "$TORCH_INDEX" torch torchvision || die "torch install failed"
fi

# ---- 4. pure-python deps ----
say "installing pure-python deps"
"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install "timm==1.0.26" "cma" "tqdm" "Pillow" || die "dep install failed"
"$PY" -m pip install "numpy==2.0.2" 2>/dev/null || say "numpy 2.0.2 unavailable, keeping whatever torch requires"

# ---- 5. verify, loudly ----
say "verifying"
"$PY" - <<'PY'
import sys
def line(k,v): print("   %-14s %s" % (k,v))
import torch, timm
line("python", sys.version.split()[0]); line("torch", torch.__version__)
line("torch.cuda", torch.version.cuda); line("cuda avail", torch.cuda.is_available())
if torch.cuda.is_available(): line("device", torch.cuda.get_device_name(0))
line("timm", timm.__version__)
try:
    import torchvision; line("torchvision", torchvision.__version__)
except Exception as e: line("torchvision", "MISSING (%s)" % e)
try:
    import cma; line("cma", cma.__version__)
except Exception as e: line("cma", "MISSING -- FOA will not run (%s)" % e)
if not torch.cuda.is_available():
    print("\n   REFUSE: torch cannot see the GPU. Latency measured now would be CPU time.")
    sys.exit(1)
if timm.__version__ != "1.0.26":
    print("\n   WARNING: timm %s != 1.0.26; model definitions may not match the checkpoints" % timm.__version__)
PY
rc=$?
[ $rc -eq 0 ] && say "env ready at $ENVDIR" || die "verification failed"
echo
say "next: bash $F/jetson/measure_latency.sh"
