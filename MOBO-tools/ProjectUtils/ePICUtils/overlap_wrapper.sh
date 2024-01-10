#!/bin/bash


if [ "$#" -ne 0 ]; then
    echo "Usage: $0 "
    exit 1
fi


current_dir=$(pwd)

cat << EOF | $EIC_SHELL_HOME/eic-shell
cd $EPIC_HOME
source install/setup.sh
cd $current_dir
$EPIC_MOBO_UTILS/runOverlapCheck.sh
EOF

line=$(cat overlap_log.txt | grep "Number of illegal overlaps/extrusions")

# empty: something wrong in overlap check, return -1
if [ -z "$line" ]; then
    echo -1
    exit 1
fi
words=($line)
# last word: number of overlaps
echo "${words[-1]}"


