#!/bin/bash


EPIC_DIR=$(pwd)/epic_klm/
if [ ! -d "$EPIC_DIR" ]; then
    echo "local ePIC source code not found, downloading"
    git clone -b one_sector https://github.com/simons27/epic_klm.git
    EPIC_DIR=$(pwd)/epic_klm/
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
export DETECTOR_CONFIG=epic_klmws_only

export EPIC_MOBO_UTILS=$(pwd)/ProjectUtils/ePICUtils/
export AIDE_HOME=$(pwd)
