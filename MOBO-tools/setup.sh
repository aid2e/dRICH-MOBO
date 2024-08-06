#!/bin/bash


EPIC_DIR=$(pwd)/epic-geom-drich-mobo/
EIC_SHELL_DIR=$(pwd)
EICRECON_DIR=$(pwd)/EICrecon-drich-mobo/

export EIC_SHELL_HOME=$EIC_SHELL_DIR
export EPIC_HOME=$EPIC_DIR

export DETECTOR_PATH=$EPIC_HOME
export DETECTOR_CONFIG=epic_craterlake

export EPIC_MOBO_UTILS=$(pwd)/ProjectUtils/ePICUtils/
export AIDE_HOME=$(pwd)

export EIC_SOFTWARE=$(pwd)/eic-software/
source load_epic.sh
