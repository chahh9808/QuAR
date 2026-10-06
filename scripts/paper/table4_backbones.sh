#!/usr/bin/env bash
# Table 4 (tab:backbones), other backbones at W6A6.
#
# ViT-B, DeiT-B and ViT-S, 3 seeds, same deployed configuration on all three.
#
# Campaigns, in the order they must run. Each is the launcher whose logs the verifier reproduces
# the published cell from; the table in README.md (Reproducing the paper) lists them.
#   scripts/run_p13_3seed_fp.sh                 # the No Adapt / T3A / NEO rows
#   scripts/run_backbones_all15.sh              # the FOA / FOZO rows
#   scripts/run_zoa_backbones_refix.sh
#   scripts/run_cumulative_t4t5.sh
# Usage
#   bash scripts/paper/table5_backbones.sh            # run every campaign, then print the paper values
#   PRINT_ONLY=1 bash scripts/paper/table5_backbones.sh    # skip the runs, just print the paper values
#
# The campaigns take hours to days of GPU time. VERIFY_ONLY re-derives every published number from
# the logs in seconds and is what you want unless you are regenerating the results.
set -uo pipefail
QUAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$QUAR"
VERIFY_ONLY="${PRINT_ONLY:-${VERIFY_ONLY:-0}}"

run(){ echo "[$(date '+%H:%M:%S')] === $1"; [ "$VERIFY_ONLY" = 1 ] && { echo "   (skipped, VERIFY_ONLY)"; return 0; }
        bash "$1" || { echo "   FAILED: $1"; return 1; }; }

run scripts/run_p13_3seed_fp.sh
run scripts/run_backbones_all15.sh
run scripts/run_zoa_backbones_refix.sh
run scripts/run_cumulative_t4t5.sh

echo
echo "note: No Adapt, T3A and NEO come from the backbone3seed_fp campaign, already on disk."

cat <<'PAPER'

=== the paper's published values, for comparison with the means above ===

Table 4, other backbones at W6A6, 3-seed mean:
                      ViT-B   DeiT-B  ViT-S
    No Adapt         47.68   37.84   32.14
    T3A              47.79   36.68   32.25
    FOA              50.38   41.04   34.35
    ZOA              51.37   41.27   36.12
    NEO              51.35   43.48   36.13
    FOZO             50.51   42.23   32.36
    QuAR             54.27   42.42   38.35
PAPER
