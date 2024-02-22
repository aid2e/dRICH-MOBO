#!/bin/bash


if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <p> <etamin> <etamax> <n particles> <radiator> <job id>"
    exit 1
fi


current_dir=$(pwd)

# produce pi+ and K+ sample
cat << EOF | $EIC_SHELL_HOME/eic-shell
cd $EPIC_HOME
source install/setup.sh
cd "$current_dir"
$EPIC_MOBO_UTILS/genMomentumScan_job.sh $1 $2 $3 $4 "kaon+" $6
$EPIC_MOBO_UTILS/dRICHAna recon_scan_$6_kaon+_p_$1_eta_$2_$3.root recon_scan_$6_kaon+_p_$1_eta_$2_$3.txt $AIDE_HOME/log/results/ $5
$EPIC_MOBO_UTILS/genMomentumScan_job.sh $1 $2 $3 $4 "pi+" $6
$EPIC_MOBO_UTILS/dRICHAna recon_scan_$6_pi+_p_$1_eta_$2_$3.root recon_scan_$6_pi+_p_$1_eta_$2_$3.txt $AIDE_HOME/log/results/  $5
EOF

#python $EPIC_MOBO_UTILS/getNsigma.py recon_scan_$6_pi+_p_$1_eta_$2_$3.txt recon_scan_$6_kaon+_p_$1_eta_$2_$3.txt

#rm recon_scan_$6_kaon+_p_$1_eta_$2_$3.txt
#rm recon_scan_$6_pi+_p_$1_eta_$2_$3.txt
rm recon_scan_$6_kaon+_p_$1_eta_$2_$3.root
rm recon_scan_$6_pi+_p_$1_eta_$2_$3.root
rm scan_$6_kaon+_p_$1_eta_$2_$3.root
rm scan_$6_pi+_p_$1_eta_$2_$3.root

