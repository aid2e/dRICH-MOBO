#!/bin/bash


if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <p> <n particles> <theta min> <theta max> <job id>"
    exit 1
fi


current_dir=$(pwd)

# produce mu- and pi- sample
cat << EOF | $EIC_SHELL_HOME/eic-shell
source $AIDE_HOME/setup.sh
source $AIDE_HOME/load_epic.sh
$EPIC_MOBO_UTILS/genMomentumScan_job.sh $1 $2 "mu-" $3 $4 $5
$EPIC_MOBO_UTILS/genMomentumScan_job.sh $1 $2 "pi-" $3 $4 $5
EOF