#!/bin/bash


if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [job number]"
    exit 1
fi


current_dir=$(pwd)

mkdir -p $AIDE_WORKDIR/log/overlaps/

if [ -f "${current_dir}/eic-shell" ]; then
    eic_shell="${current_dir}/eic-shell"
else
    eic_shell="${EIC_SHELL_HOME}/eic-shell"
fi

cat << EOF | "${eic_shell}"
source $AIDE_HOME/load_epic.sh
# $EPIC_MOBO_UTILS/runOverlapCheck_jobGeo_local.sh $1 >&2
$AIDE_HOME/ProjectUtils/ePICUtils/runOverlapCheck_jobGeo_local.sh $1 >&2
EOF

cat $AIDE_WORKDIR/log/overlaps/overlap_log_$1.txt >&2

line=$(cat $AIDE_WORKDIR/log/overlaps/overlap_log_$1.txt | grep "Number of illegal overlaps/extrusions")

# empty: something wrong in overlap check, return -1
if [ -z "$line" ]; then
    echo -1
    exit 1
fi
words=($line)
# last word: number of overlaps
echo "${words[-1]}"


