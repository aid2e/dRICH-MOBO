#!/bin/bash


if [ "$#" -lt 8 ] || [ "$#" -gt 9 ]; then
    echo "Usage: $0 <p> <etamin> <etamax> <n particles> <radiator> <job id> <particle> <input_name> [json_filename (optional)]"
    exit 1
fi

JSON_FNAME="blaa"
#if there are 9 arguments make 9th argument name of json file  JSON_FNAME
if [ "$#" -eq 9 ]; then
    JSON_FNAME="$9"
fi

current_dir=$(pwd)

mkdir -p $AIDE_WORKDIR/log/sim_files/
mkdir -p $AIDE_WORKDIR/log/results/

# produce pi+ and K+ sample
if [ -f "${current_dir}/eic-shell" ]; then
    eic_shell="${current_dir}/eic-shell"
else
    eic_shell="${EIC_SHELL_HOME}/eic-shell"
fi

cat << EOF | "${eic_shell}"
source $AIDE_HOME/load_epic.sh

echo "shell_wrapper_job dRICHAna_fake"

echo $AIDE_HOME/ProjectUtils/ePICUtils/dRICHAna_fake $8 recon_scan_$6_$7_p_$1_eta_$2_$3.txt $AIDE_WORKDIR/log/results/ $5 $JSON_FNAME
$AIDE_HOME/ProjectUtils/ePICUtils/dRICHAna_fake $8 recon_scan_$6_$7_p_$1_eta_$2_$3.txt $AIDE_WORKDIR/log/results/ $5 $JSON_FNAME

EOF

# python $EPIC_MOBO_UTILS/getNsigma.py recon_scan_$6_pi+_p_$1_eta_$2_$3.txt recon_scan_$6_kaon+_p_$1_eta_$2_$3.txt

# rm recon_scan_$6_kaon+_p_$1_eta_$2_$3.txt
# rm recon_scan_$6_pi+_p_$1_eta_$2_$3.txt
# rm $AIDE_WORKDIR/log/sim_files/recon_scan_$6_$7_p_$1_eta_$2_$3.root
# rm $AIDE_WORKDIR/log/sim_files/scan_$6_$7_p_$1_eta_$2_$3.root
# rm $AIDE_WORKDIR/log/sim_files/scan_$6_pi+_p_$1_eta_$2_$3.root

