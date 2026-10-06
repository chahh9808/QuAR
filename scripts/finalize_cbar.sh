#!/usr/bin/env bash
# Wait for C-bar generation -> remap severities to levels -> verify 5000 imgs/corruption ->
# launch the C-bar sweep. Exits early (nonzero) if anything is short, so we don't burn GPU
# time on a broken setup.
set -uo pipefail
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python3}"
GENLOG=$(cat "$QUAR/runs/last_cbar_genlog_dir.txt")
GEN="${DATA_CBAR_GEN:?set DATA_CBAR_GEN to the imagenet-c-bar-gen root}"/ImageNet-C-Bar
CORRS="caustic_refraction inverse_sparkles sparkles perlin_noise blue_noise_sample plasma_noise checkerboard_cutout cocentric_sine_waves single_frequency_greyscale brownish_noise"

echo "[finalize] waiting for generation _DONE flag ..."
waited=0
until [ -f "$GENLOG/_DONE" ]; do
  sleep 15; waited=$((waited+15))
  if [ "$waited" -ge 3600 ]; then echo "[finalize] TIMEOUT waiting for generation"; exit 2; fi
done
echo "[finalize] generation DONE after ~${waited}s"

echo "[finalize] verifying 5000 images per corruption (highest-severity dir) ..."
short=0
for c in $CORRS; do
  d=$(ls -d "$GEN/$c"/*/ 2>/dev/null | head -1)
  n=$(find "$d" -name '*.JPEG' 2>/dev/null | wc -l)
  printf "  %-28s %s imgs\n" "$c" "$n"
  [ "$n" -lt 5000 ] && short=$((short+1))
done
if [ "$short" -gt 0 ]; then echo "[finalize] $short corruption(s) short of 5000 — aborting"; exit 3; fi

echo "[finalize] remapping float severities -> integer levels ..."
$PY "$QUAR/tools/tables/remap_cbar_levels.py" || { echo "[finalize] remap failed"; exit 4; }

echo "[finalize] launching C-bar sweep (run_cbar.sh) ..."
cd "$QUAR"
bash scripts/run_cbar.sh > runs/cbar_launch.log 2>&1
echo "[finalize] C-bar sweep launcher exited $?"
touch "$GENLOG/_SWEEP_DONE"
echo "[finalize] ALL DONE"
