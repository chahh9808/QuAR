#!/usr/bin/env bash
# Full 12-corruption ImageNet-3DCC: wait for the 3 big video-codec tarballs (aria2c), extract
# their sev-5 5k subsets, verify all 12, re-run the full 12-corruption sweep, regenerate report.
# 9 corruptions already extracted under imagenet-3dcc/.
set -uo pipefail
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python3}"
TARDIR="${DATA_3DCC_TAR:?set DATA_3DCC_TAR to the imagenet-3dcc-tar root}"
DST="${DATA_3DCC:?set DATA_3DCC to the imagenet-3dcc root}"
IDS=$QUAR/robustbench/data/imagenet_test_image_ids.txt
BIG="bit_error h265_abr h265_crf"
ALL12="near_focus far_focus fog_3d flash color_quant low_light xy_motion_blur z_motion_blur iso_noise bit_error h265_abr h265_crf"
log() { echo "[orch12 $(date '+%H:%M:%S')] $*"; }

# 1) wait for each big download (aria2c removes <file>.aria2 control file on completion)
for c in $BIG; do
  log "waiting for $c download (aria2c control file to clear) ..."
  w=0
  while [ -f "$TARDIR/$c.tar.gz.aria2" ] || [ ! -f "$TARDIR/$c.tar.gz" ]; do
    sleep 60; w=$((w+60)); [ "$w" -ge 54000 ] && { log "$c download TIMEOUT (15h)"; exit 2; }
  done
  log "$c download complete ($(du -h "$TARDIR/$c.tar.gz" 2>/dev/null | cut -f1))"
done

# 2) extract sev-5 5k subset for each big corruption (tar scans the whole archive; ~10-15min each)
for c in $BIG; do
  n=$(find "$DST/$c/5" -name '*.JPEG' 2>/dev/null | wc -l)
  if [ "$n" -lt 5000 ]; then
    flist="$TARDIR/$c.sev5.list"; sed "s#^#$c/5/#" "$IDS" > "$flist"
    log "$c: extracting sev-5 5k subset ..."
    tar xzf "$TARDIR/$c.tar.gz" -C "$DST" --files-from="$flist" 2>"$TARDIR/$c.extract.err" || log "$c: tar warnings (see $c.extract.err)"
  fi
  log "$c: extracted $(find "$DST/$c/5" -name '*.JPEG' 2>/dev/null | wc -l)/5000"
done

# 3) verify all 12 corruptions have 5000 sev-5 images
short=0
for c in $ALL12; do
  n=$(find "$DST/$c/5" -name '*.JPEG' 2>/dev/null | wc -l)
  printf "  %-16s %s\n" "$c" "$n"
  [ "$n" -lt 5000 ] && short=$((short+1))
done
[ "$short" -gt 0 ] && { log "$short corruption(s) short of 5000 -- aborting before GPU run"; exit 3; }
touch "$DST/_EXTRACT_DONE_12"

# 4) run the full 12-corruption sweep
log "launching full 12-corruption 3DCC sweep ..."
cd "$QUAR"
bash scripts/run_3dcc.sh > runs/3dcc_launch_12.log 2>&1
log "sweep launcher exited $?"

# 5) regenerate report (overwrites the 9-corruption version with the full 12)
$PY tools/tables/generate_3dcc.py
log "12-corruption report generated"
touch "$QUAR/runs/3dcc_12_ALL_DONE"
log "ALL DONE"
