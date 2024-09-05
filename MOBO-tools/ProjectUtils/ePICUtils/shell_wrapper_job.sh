#!/bin/bash


if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <p> <etamin> <etamax> <n particles> <radiator> <job id> <particle>"
    exit 1
fi


current_dir=$(pwd)

# produce pi+ and K+ sample
cat << EOF | $EIC_SHELL_HOME/eic-shell
source $AIDE_HOME/load_epic.sh
$EPIC_MOBO_UTILS/genMomentumScan_job.sh $1 $2 $3 $4 $7 $6
$EPIC_MOBO_UTILS/dRICHAna $AIDE_HOME/log/sim_files/recon_scan_$4_$6_$7_p_$1_eta_$2_$3.root recon_scan_$4_$6_$7_p_$1_eta_$2_$3.txt $AIDE_HOME/log/results/ $5
EOF

#python $EPIC_MOBO_UTILS/getNsigma.py recon_scan_$6_pi+_p_$1_eta_$2_$3.txt recon_scan_$6_kaon+_p_$1_eta_$2_$3.txt

#rm recon_scan_$6_kaon+_p_$1_eta_$2_$3.txt
#rm recon_scan_$6_pi+_p_$1_eta_$2_$3.txt
rm $AIDE_HOME/log/sim_files/recon_scan_$4_$6_$7_p_$1_eta_$2_$3.root
rm $AIDE_HOME/log/sim_files/scan_$4_$6_$7_p_$1_eta_$2_$3.root
#rm $AIDE_HOME/log/sim_files/scan_$6_pi+_p_$1_eta_$2_$3.root

