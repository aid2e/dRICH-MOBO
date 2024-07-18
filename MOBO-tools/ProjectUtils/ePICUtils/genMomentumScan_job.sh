#!/bin/bash

if [ "$#" != 6 ]; then
    echo "Usage: $0 [p] [N particles] [particle type] [theta min] [theta max] [job id]"
    exit 1
fi

ddsim --compactFile ${DETECTOR_PATH}/${DETECTOR_CONFIG}_$6.xml --runType "batch" -G -N $2 --gun.particle $3 --gun.momentumMin "$1*GeV" --gun.momentumMax "$1*GeV" --gun.thetaMin "$4*deg" --gun.thetaMax "$5*deg" --gun.distribution "uniform" --physics.list "FTFP_BERT" --part.userParticleHandler="" --outputFile $AIDE_HOME/log/sim_files/scan_$6_$3_p_$1.edm4hep.root
echo "done with ddsim ($3 job #$6)"