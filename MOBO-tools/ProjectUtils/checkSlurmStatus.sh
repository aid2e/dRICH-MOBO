#!/bin/bash

if [ "$#" != 1 ]; then
    echo "Usage: $0 [slurm job id]"
    exit 1
fi

read -r state exit_code <<< $(sacct -j $1 --format=State,ExitCode --noheader | awk '{print $1,$3}')

if [ "$state" == "COMPLETED" ] && [ "$exit_code" == "0:0" ]; then
    echo 1
elif [ "$state" == "PENDING" ] || [ "$state" == "RUNNING" ] || [ "$state" == "COMPLETING" ] || [ "$state" == "SUSPENDED" ]; then
    echo 0
elif [ "$state" == "FAILED" ]; then
    echo -1