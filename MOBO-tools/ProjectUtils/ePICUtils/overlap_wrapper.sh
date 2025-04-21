#!/bin/bash


if [ "$#" -ne 0 ]; then
    echo "Usage: $0 "
    exit 1
fi


current_dir=$(pwd)

if [ -f "${current_dir}/eic-shell" ]; then
    eic_shell="${current_dir}/eic-shell"
else
    eic_shell="${EIC_SHELL_HOME}/eic-shell"
fi

cat << EOF | "${eic_shell}"
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


