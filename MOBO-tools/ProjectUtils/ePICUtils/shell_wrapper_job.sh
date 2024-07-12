#!/bin/bash


if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <p> <n particles> <job id>"
    exit 1
fi


current_dir=$(pwd)

# produce mu- and pi- sample
cat << EOF | $EIC_SHELL_HOME/eic-shell
source $AIDE_HOME/setup.sh
source $AIDE_HOME/load_epic.sh
$EPIC_MOBO_UTILS/genMomentumScan_job.sh $1 $2 "mu-" $3
$EPIC_MOBO_UTILS/genMomentumScan_job.sh $1 $2 "pi-" $3
EOF

python $EPIC_MOBO_UTILS/getObjectives.py $AIDE_HOME/log/sim_files/scan_$3_mu-_p_$1.edm4hep.root $AIDE_HOME/log/sim_files/scan_$3_pi-_p_$1.edm4hep.root $AIDE_HOME/log/results/objectives_$3_p_$1.txt $3

rm $AIDE_HOME/log/sim_files/scan_$3_mu-_p_$1.edm4hep.root
rm $AIDE_HOME/log/sim_files/scan_$3_pi-_p_$1.edm4hep.root