#!/bin/bash

if [ "$#" != 6 ]; then
    echo "Usage: $0 [p] [min eta] [max eta] [N particles] [particle type] [job id]"
    exit 1
fi

npsim --compactFile ${DETECTOR_PATH}/${DETECTOR_CONFIG}_$6.xml -G -N $4 --gun.etaMax $3  --gun.etaMin $2 --gun.momentumMax "$1*GeV" --gun.momentumMin "$1*GeV" --gun.particle $5 --gun.distribution "uniform" -O scan_$6_$5_p_$1_eta_$2_$3.root
eicrecon scan_$6_$5_p_$1_eta_$2_$3.root -Ppodio:output_file=recon_scan_$6_$5_p_$1_eta_$2_$3.root -Pdd4hep:xml_files=${DETECTOR_PATH}/${DETECTOR_CONFIG}_$6.xml
