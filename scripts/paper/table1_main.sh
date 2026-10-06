#!/usr/bin/env bash
# Table 1 (tab:main), the main ImageNet-C comparison.
#
# Seven methods at four bit-widths, 3 seeds, plus the ablation rows and the full-precision row.
# This is the table every other result is read against.
#
# Campaigns, in the order they must run. Each is the launcher whose logs the verifier reproduces
# the published cell from; the table in README.md (Reproducing the paper) lists them.
#   scripts/run_cumulative_maintable.sh
#   scripts/run_fp_3seed_table1.sh
#   scripts/run_p13_3seed_fp.sh
#   scripts/run_zoa_table1_refix.sh
#   scripts/run_cumulative_appendix.sh          # the Final-only and Local-only rows, seed 0
#   SEED=1 scripts/run_cumulative_appendix.sh   # ... and seeds 1 and 2
#   SEED=2 scripts/run_cumulative_appendix.sh
# Usage
#   bash scripts/paper/table1_main.sh            # run every campaign, then print the paper values
#   PRINT_ONLY=1 bash scripts/paper/table1_main.sh    # skip the runs, just print the paper values
#
# The campaigns take hours to days of GPU time. PRINT_ONLY skips them and prints only the
# published values, which is what you want unless you are regenerating the results.
#
# The QuAR row comes from run_cumulative_maintable.sh; the two ablation rows from
# run_cumulative_appendix.sh, whose seed-0 cells reproduce the published means exactly.
set -uo pipefail
QUAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$QUAR"
VERIFY_ONLY="${PRINT_ONLY:-${VERIFY_ONLY:-0}}"

run(){ echo "[$(date '+%H:%M:%S')] === $1"; [ "$VERIFY_ONLY" = 1 ] && { echo "   (skipped, VERIFY_ONLY)"; return 0; }
        bash "$1" || { echo "   FAILED: $1"; return 1; }; }

run scripts/run_cumulative_maintable.sh
run scripts/run_fp_3seed_table1.sh
run scripts/run_p13_3seed_fp.sh
run scripts/run_zoa_table1_refix.sh
run scripts/run_cumulative_appendix.sh
SEED=1 run scripts/run_cumulative_appendix.sh
SEED=2 run scripts/run_cumulative_appendix.sh


cat <<'PAPER'

=== the paper's published values, for comparison with the means above ===

Table 1, ImageNet-C mean top-1 (%), 3-seed:
                      W3A3   W4A4   W6A6   W8A8
    No Adapt         28.35  47.30  47.69  54.36
    T3A              28.49  47.42  47.79  54.16
    FOA              31.52  50.25  50.39  57.85
    ZOA              31.45  50.87  51.37  58.16
    NEO              31.91  50.29  51.35  58.09
    FOZO             31.98  50.09  49.78  57.74
    Final-only       31.43  50.21  51.20  58.10
    Local-only       35.30  52.87  53.42  59.11
    QuAR             35.98  53.58  54.35  60.44
PAPER
