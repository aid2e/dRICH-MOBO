#!/bin/bash

#source $EIC_SOFTWARE/bin/eicrecon-this.sh
source $EIC_SOFTWARE/bin/thisepic.sh epic_craterlake

mkdir -p ${AIDE_WORKDIR}/share/epic/
cp -rf $EIC_SOFTWARE/share/epic/* ${AIDE_WORKDIR}/share/epic/
export DETECTOR_PATH=${AIDE_WORKDIR}/share/epic
