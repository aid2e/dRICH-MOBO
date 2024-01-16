#!/bin/bash


EPIC_DIR=$(pwd)/epic/
if [ ! -d "$EPIC_DIR" ]; then
    echo "local ePIC source code not found, downloading"
    git clone "https://github.com/eic/epic.git"
    EPIC_DIR=$(pwd)/epic/
else
    echo "ePIC source code found"
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
export EPIC_MOBO_UTILS=$(pwd)/ProjectUtils/ePICUtils/
