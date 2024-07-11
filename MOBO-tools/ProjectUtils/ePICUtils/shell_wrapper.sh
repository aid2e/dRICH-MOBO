#!/bin/bash


if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <p> <n particles>"
    exit 1
fi


current_dir=$(pwd)

# produce mu- and pi- sample
cat << EOF | $EIC_SHELL_HOME/eic-shell
cd $EPIC_HOME
source install/setup.sh
cd "$current_dir"
$EPIC_MOBO_UTILS/genMomentumScan.sh $1 $2 "mu-"
$EPIC_MOBO_UTILS/genMomentumScan.sh $1 $2 "pi-"
EOF

# python $EPIC_MOBO_UTILS/getNsigma.py recon_scan_pi+_p_$1_eta_$2_$3.txt recon_scan_kaon+_p_$1_eta_$2_$3.txt

rm scan_mu-_p_$1.root
rm scan_pi-_p_$1.root