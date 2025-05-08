#!/bin/bash

export SINGULARITY_OPTIONS='--bind /cvmfs:/cvmfs'

CurrentDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

EPIC_DIR=${CurrentDir}/epic-geom-drich-mobo/
EIC_SHELL_DIR=${CurrentDir}
EICRECON_DIR=${CurrentDir}/EICrecon-drich-mobo/

export EIC_SHELL_HOME=$EIC_SHELL_DIR
export EPIC_HOME=$EPIC_DIR

export DETECTOR_PATH=$EPIC_HOME
export DETECTOR_CONFIG=epic_craterlake

export EPIC_MOBO_UTILS=${CurrentDir}/ProjectUtils/ePICUtils/
# export AIDE_HOME=${CurrentDir}

export EIC_SOFTWARE=${CurrentDir}/eic-software/
#source load_epic.sh

export AIDE_WORKDIR=$(pwd)/work

export AIDE_HOME=$(pwd)
