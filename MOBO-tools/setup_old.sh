#!/bin/bash


EPIC_DIR=$(pwd)/epic-geom-drich-mobo/
if [ ! -d "$EPIC_DIR" ]; then
    echo "local ePIC source code not found, downloading"
    git clone "https://github.com/cpecar/epic-geom-drich-mobo.git"
    EPIC_DIR=$(pwd)/epic-geom-drich-mobo/
else
    echo "ePIC geometry source code found"
fi

EIC_SHELL_DIR=$(pwd)
if [ ! -f "$EIC_SHELL_DIR/eic-shell" ]; then
    echo "eic-shell not found, downloading"
    curl --location https://get.epic-eic.org | bash
    EIC_SHELL_DIR=$(pwd)
else
    echo "eic-shell found"
fi

export EIC_SHELL_HOME=$EIC_SHELL_DIR
export EPIC_HOME=$EPIC_DIR

export DETECTOR_PATH=$EPIC_HOME
export DETECTOR_CONFIG=epic_craterlake

export EPIC_MOBO_UTILS=$(pwd)/ProjectUtils/ePICUtils/
export AIDE_HOME=$(pwd)
