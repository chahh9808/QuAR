#!/usr/bin/env bash
# Table 6 (tab:sens), sensitivity sweeps.
#
# Recalibration strength lambda, batch size, and source-statistic set size. All three at W6A6, 3 seeds.
#
# Campaigns, in the order they must run. Each is the launcher whose logs the verifier reproduces
# the published cell from; the table in README.md (Reproducing the paper) lists them.
#   scripts/run_sens_3seed_w6a6.sh
#
# All three sweeps live in that one campaign, and it also produces appendix tab:sens_full, which
# is this table with the standard deviations kept.
# Usage
#   bash scripts/paper/table6_sensitivity.sh            # run every campaign, then print the paper values
#   PRINT_ONLY=1 bash scripts/paper/table6_sensitivity.sh    # skip the runs, just print the paper values
#
# The campaigns take hours to days of GPU time. VERIFY_ONLY re-derives every published number from
# the logs in seconds and is what you want unless you are regenerating the results.
set -uo pipefail
QUAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$QUAR"
VERIFY_ONLY="${PRINT_ONLY:-${VERIFY_ONLY:-0}}"

run(){ echo "[$(date '+%H:%M:%S')] === $1"; [ "$VERIFY_ONLY" = 1 ] && { echo "   (skipped, VERIFY_ONLY)"; return 0; }
        bash "$1" || { echo "   FAILED: $1"; return 1; }; }

run scripts/run_sens_3seed_w6a6.sh

echo
echo "note: The 200-source-batch point is the deployed default and has no sweep run of its own."

cat <<'PAPER'

=== the paper's published values, for comparison with the means above ===

Table 6, sensitivity at W6A6, 3-seed mean:
    lambda      0 51.20   0.1 52.61   0.3 54.00   0.5 54.35   0.7 54.20   1.0 53.52
    batch       1 54.27   4 54.39   8 54.47   16 54.35   64 54.35   128 54.29
    src batches 16 53.65  32 53.88  64 54.18  128 54.29  200 54.35  256 54.33
PAPER
