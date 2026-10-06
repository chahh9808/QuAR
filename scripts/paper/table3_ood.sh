#!/usr/bin/env bash
# Table 3 (tab:t4app), generalization to other OOD benchmarks.
#
# Assembled from three campaigns and that is the thing to know about it: the baselines come from
# the per-dataset seed dirs, ZOA from the fix_layers re-measurement, QuAR from the cumulative run.
# Running only the OOD aggregator reproduces neither of those two rows.
#
# Campaigns, in the order they must run. Each is the launcher whose logs the verifier reproduces
# the published cell from; the table in README.md (Reproducing the paper) lists them.
#   scripts/run_cbar.sh
#   scripts/run_3dcc.sh
#   scripts/run_table4_queue.sh
#   scripts/run_cumulative_t4t5.sh
#   scripts/run_zoa_all_remaining.sh
# Usage
#   bash scripts/paper/table4_ood.sh            # run every campaign, then print the paper values
#   PRINT_ONLY=1 bash scripts/paper/table4_ood.sh    # skip the runs, just print the paper values
#
# The campaigns take hours to days of GPU time. VERIFY_ONLY re-derives every published number from
# the logs in seconds and is what you want unless you are regenerating the results.
set -uo pipefail
QUAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$QUAR"
VERIFY_ONLY="${PRINT_ONLY:-${VERIFY_ONLY:-0}}"

run(){ echo "[$(date '+%H:%M:%S')] === $1"; [ "$VERIFY_ONLY" = 1 ] && { echo "   (skipped, VERIFY_ONLY)"; return 0; }
        bash "$1" || { echo "   FAILED: $1"; return 1; }; }

run scripts/run_cbar.sh
run scripts/run_3dcc.sh
run scripts/run_table4_queue.sh
run scripts/run_cumulative_t4t5.sh
run scripts/run_zoa_all_remaining.sh

echo
echo "note: C-bar needs the dataset generated first; see jetson/../README.md, section Data."

cat <<'PAPER'

=== the paper's published values, for comparison with the means above ===

Table 3, OOD benchmarks, 3-seed mean (C-bar / R / Sketch / 3DCC):
    W3A3   No Adapt 36.63 42.48 25.01 34.21    QuAR 44.92 43.90 32.10 37.50
    W6A6   No Adapt 59.07 55.89 40.13 47.68    QuAR 65.45 56.60 45.65 51.20
PAPER
