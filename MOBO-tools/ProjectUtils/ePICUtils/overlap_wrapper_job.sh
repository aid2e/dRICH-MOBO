#!/bin/bash


if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [job number]"
    exit 1
fi


current_dir=$(pwd)

cat << EOF | $EIC_SHELL_HOME/eic-shell
cd $EPIC_HOME
source install/setup.sh
cd $current_dir
$EPIC_MOBO_UTILS/runOverlapCheck_jobGeo.sh $1
EOF

line=$(cat $AIDE_HOME/log/overlaps/overlap_log_$1.txt | grep "Number of illegal overlaps/extrusions")

# empty: something wrong in overlap check, return -1
if [ -z "$line" ]; then
    echo -1
    exit 1
fi
words=($line)
# last word: number of overlaps
echo "${words[-1]}"


