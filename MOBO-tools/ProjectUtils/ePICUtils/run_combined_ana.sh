#!/usr/bin/env bash

if [ $# -ne 8 ]; then
    echo "Usage: $0 <npart> <job_id> <p> <eta_min> <eta_max> <radiator> <nsamples per bootstrap> <nbootstraps>"    
    exit 1
fi

NPART=$1
JOB_ID=$2
P=$3
ETA_MIN=$4
ETA_MAX=$5
RADIATOR=$6
NSAMPLES=$7
NBOOTSTRAPS=$8

: "${AIDE_HOME:?ERROR: AIDE_HOME not set}"
: "${SIM_OUTPUT:?ERROR: SIM_OUTPUT not set}"

ANA_SCRIPT="${AIDE_HOME}/ProjectUtils/ePICUtils/dRICHAna_bootstrap"

INFILE_PI="${SIM_OUTPUT}/recon_scan_${NPART}_${JOB_ID}_pi+_p_${P}_eta_${ETA_MIN}_${ETA_MAX}.root"
INFILE_K="${SIM_OUTPUT}/recon_scan_${NPART}_${JOB_ID}_kaon+_p_${P}_eta_${ETA_MIN}_${ETA_MAX}.root"
OUTFILE="${AIDE_HOME}/log/results/recon_scan_${NPART}_${JOB_ID}_p_${P}_eta_${ETA_MIN}_${ETA_MAX}.txt"

# make sure the output dir exists
mkdir -p "$(dirname "$OUTFILE")"

cat << EOF | $EIC_SHELL_HOME/eic-shell
"$ANA_SCRIPT" \
  "$INFILE_PI" \
  "$INFILE_K" \
  "$OUTFILE"  \
  "$RADIATOR" \
  "$NSAMPLES" \
  "$NBOOTSTRAPS"
EOF
