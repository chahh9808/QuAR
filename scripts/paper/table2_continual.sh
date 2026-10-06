#!/usr/bin/env bash
# Table 2 (tab:t3), long-term continual adaptation.
#
# A 10-round by 15-corruption continual stream at W3A3 and W6A6. The QuAR row deliberately uses
# EMA alpha=0.1 to match NEO-Continual's momentum; that matched rate is the comparison.
#
# Campaigns, in the order they must run. Each is the launcher whose logs the verifier reproduces
# the published cell from; the table in README.md (Reproducing the paper) lists them.
#   scripts/run_continual_ema_alpha01.sh
#   scripts/run_neo_continual_3seed.sh
#   scripts/run_zoa_all_remaining.sh
# Usage
#   bash scripts/paper/table2_continual.sh            # run every campaign, then print the paper values
#   PRINT_ONLY=1 bash scripts/paper/table2_continual.sh    # skip the runs, just print the paper values
#
# The campaigns take hours to days of GPU time. VERIFY_ONLY re-derives every published number from
# the logs in seconds and is what you want unless you are regenerating the results.
set -uo pipefail
QUAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$QUAR"
VERIFY_ONLY="${PRINT_ONLY:-${VERIFY_ONLY:-0}}"

run(){ echo "[$(date '+%H:%M:%S')] === $1"; [ "$VERIFY_ONLY" = 1 ] && { echo "   (skipped, VERIFY_ONLY)"; return 0; }
        bash "$1" || { echo "   FAILED: $1"; return 1; }; }

run scripts/run_continual_ema_alpha01.sh
run scripts/run_neo_continual_3seed.sh
run scripts/run_zoa_all_remaining.sh

echo
echo "note: No Adapt is unadapted accuracy, constant across rounds, and comes from Table 1."

cat <<'PAPER'

=== the paper's published values, for comparison with the means above ===

Table 2, 10-round continual, round-1 and average:
                      W3A3 R1 / Avg      W6A6 R1 / Avg
    No Adapt         28.35 / 28.35      47.69 / 47.69
    ZOA              31.27 / 31.94      51.59 / 52.73
    NEO-Continual    31.69 / 31.67      51.16 / 51.15
    QuAR             35.66 / 35.65      53.94 / 53.91
PAPER
