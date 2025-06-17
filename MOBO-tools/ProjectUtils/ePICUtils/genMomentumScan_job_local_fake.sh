#!/usr/bin/env bash

# run_scan.sh
#
# A wrapper script that:
# 1) Checks for six arguments: [p] [min eta] [max eta] [N particles] [particle type] [job id]
# 2) Compiles createScan.cpp (using root-config) if the binary is missing or out of date
# 3) Runs the resulting executable with the provided arguments
#
# Usage:
#   ./run_scan.sh [p] [min eta] [max eta] [N particles] [particle type] [job id]
# Example:
#   ./run_scan.sh 25 0.5 2.0 100 pi+ job42

# 1. Verify argument count
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 [p] [min eta] [max eta] [N particles] [particle type] [job id]"
    exit 1
fi

# 2. Determine script directory and source/binary paths
BIN="${AIDE_HOME}/fake_reco"


# 4. Run the executable with all six arguments
echo " "$BIN" "${AIDE_WORKDIR}/log/sim_files/scan_$6_$5_p_$1_eta_$2_$3.root $4" "
"$BIN" "${AIDE_WORKDIR}/log/sim_files/scan_$6_$5_p_$1_eta_$2_$3.root" "$4"

# 5. Just rename the output to recon_scan for whatever stuff...
cp "${AIDE_WORKDIR}/log/sim_files/scan_$6_$5_p_$1_eta_$2_$3.root" "${AIDE_WORKDIR}/log/sim_files/recon_scan_$6_$5_p_$1_eta_$2_$3.root"
