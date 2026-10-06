#!/usr/bin/env bash
# End-to-end 3DCC orchestration: wait for extraction + C-bar sweep (free GPUs), verify the
# extracted 5k subsets, run the 3DCC sweep, generate the report. Aborts before using GPU time
# if any corruption is short.
set -uo pipefail
QUAR="${QUAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python3}"
DST="${DATA_3DCC:?set DATA_3DCC to the imagenet-3dcc root}"
CORRS="near_focus far_focus fog_3d flash color_quant low_light xy_motion_blur z_motion_blur iso_noise bit_error h265_abr h265_crf"
log() { echo "[orch $(date '+%H:%M:%S')] $*"; }

# 1) wait for extraction to finish
log "waiting for 3DCC extraction (_EXTRACT_DONE) ..."
w=0; until [ -f "$DST/_EXTRACT_DONE" ]; do sleep 30; w=$((w+30)); [ "$w" -ge 10800 ] && { log "TIMEOUT extraction"; exit 2; }; done
log "extraction done after ~${w}s"

# 2) verify all 12 corruptions have 5000 sev-5 images
short=0
for c in $CORRS; do
  n=$(find "$DST/$c/5" -name '*.JPEG' 2>/dev/null | wc -l)
  printf "  %-16s %s\n" "$c" "$n"
  [ "$n" -lt 5000 ] && short=$((short+1))
done
[ "$short" -gt 0 ] && { log "$short corruption(s) short of 5000 -- aborting before GPU run"; exit 3; }

# 3) wait for the C-bar sweep to free the GPUs (all 28 cbar cells done)
log "waiting for C-bar sweep to finish (free GPUs) ..."
RDC=$(ls -1dt "$QUAR"/runs/cbar_2* 2>/dev/null | head -1)
w=0
while true; do
  done=0
  for d in "$RDC"/*/; do c=$(find "$d" -name corruption_metrics.csv 2>/dev/null|head -1); [ -n "$c" ] && r=$(($(wc -l <"$c")-1)) || r=0; [ "$r" -ge 10 ] && done=$((done+1)); done
  [ "$done" -ge 28 ] && { log "C-bar sweep complete"; break; }
  sleep 30; w=$((w+30)); [ "$w" -ge 5400 ] && { log "C-bar wait TIMEOUT ($done/28) -- proceeding anyway"; break; }
done

# 4) run the 3DCC sweep
log "launching 3DCC sweep ..."
cd "$QUAR"
bash scripts/run_3dcc.sh > runs/3dcc_launch.log 2>&1
log "3DCC sweep launcher exited $?"

# 5) generate the report
$PY tools/tables/generate_3dcc.py
log "report generated"
touch "$QUAR/runs/3dcc_ALL_DONE"
log "ALL DONE"
