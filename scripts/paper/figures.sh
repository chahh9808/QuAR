#!/usr/bin/env bash
# Figures (main text and appendix).
#
# fig:method is drawn in the document and has no campaign. fig:problem
# is the code-distribution measurement and fig:lambda the strength sweep (appendix). fig:teaser replots the
# W1 code-distribution campaign, and fig:noniid (appendix, next to tab:noniid) replots tab:noniid, whose 22
# cells were recomputed from the non-IID campaign. fig:dgain is produced by a tool rather than a campaign.
#
# Campaigns, in the order they must run. Each is the launcher whose logs the verifier reproduces
# the published cell from; the table in README.md (Reproducing the paper) lists them.
#   scripts/run_codehist_figure.sh            # fig:problem
#   scripts/run_codedist_seeds12.sh           # fig:problem
#   scripts/run_head_codedist.sh              # fig:problem
#   scripts/run_head_codedist_baselines.sh    # fig:problem
#   scripts/run_w1_codedist_full.sh           # fig:problem and fig:teaser
#   scripts/run_lambda_ablation.sh            # fig:lambda
#   scripts/run_lambda_oracle.sh              # fig:lambda
#   scripts/run_noniid_3seed_w6a6.sh          # fig:noniid, the same numbers as tab:noniid
#   tools/analysis/measure_downstream_gain.py # fig:dgain
# Usage
#   bash scripts/paper/figures.sh                 # run every campaign
#   PRINT_ONLY=1 bash scripts/paper/figures.sh    # list them without running anything
#
# The campaigns take hours to days of GPU time.
set -uo pipefail
QUAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$QUAR"
VERIFY_ONLY="${PRINT_ONLY:-${VERIFY_ONLY:-0}}"

run(){ echo "[$(date '+%H:%M:%S')] === $1"; [ "$VERIFY_ONLY" = 1 ] && { echo "   (skipped, VERIFY_ONLY)"; return 0; }
        bash "$1" || { echo "   FAILED: $1"; return 1; }; }

run scripts/run_codehist_figure.sh
run scripts/run_codedist_seeds12.sh
run scripts/run_head_codedist.sh
run scripts/run_head_codedist_baselines.sh
run scripts/run_w1_codedist_full.sh
run scripts/run_lambda_ablation.sh
run scripts/run_lambda_oracle.sh
run scripts/run_noniid_3seed_w6a6.sh

echo
echo "note: these campaigns produce the measurements; the plots are drawn by the paper repository."
echo "      fig:dgain comes from tools/analysis/measure_downstream_gain.py, not from a campaign."

echo
echo "no automatic verifier for the figures; compare the plots by eye."
