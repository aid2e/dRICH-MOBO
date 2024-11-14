#!/bin/bash

if [ "$#" != 6 ]; then
    echo "Usage: $0 [p] [min eta] [max eta] [N particles] [particle type] [job id]"
    exit 1
fi

# npsim --compactFile ${DETECTOR_PATH}/${DETECTOR_CONFIG}_$6.xml -G -N $4 --gun.etaMax $3  --gun.etaMin $2 --gun.phiMin "0" --gun.phiMax "6.2831853" --gun.momentumMax "$1*GeV" --gun.momentumMin "$1*GeV" --gun.particle $5 --gun.distribution "eta" -O $AIDE_HOME/log/sim_files/scan_$6_$5_p_$1_eta_$2_$3.root
# echo "done with npsim"
# eicrecon $AIDE_HOME/log/sim_files/scan_$6_$5_p_$1_eta_$2_$3.root -Ppodio:output_file=$AIDE_HOME/log/sim_files/recon_scan_$6_$5_p_$1_eta_$2_$3.root -Pdd4hep:xml_files=${DETECTOR_PATH}/${DETECTOR_CONFIG}_$6.xml -Ppodio:output_include_collections=DRICHHits,MCParticles,DRICHRawHits,DRICHRawHitsAssociations,DRICHAerogelTracks,DRICHGasTracks,DRICHAerogelIrtCherenkovParticleID,DRICHGasIrtCherenkovParticleID,DRICHMergedIrtCherenkovParticleID
# echo "done with eicrecon"

npsim --compactFile ${AIDE_WORKDIR}/share/epic/${DETECTOR_CONFIG}_$6.xml -G -N $4 --gun.etaMax $3  --gun.etaMin $2 --gun.phiMin "0" --gun.phiMax "6.2831853" --gun.momentumMax "$1*GeV" --gun.momentumMin "$1*GeV" --gun.particle $5 --gun.distribution "eta" -O ${AIDE_WORKDIR}/log/sim_files/scan_$6_$5_p_$1_eta_$2_$3.root
echo "done with npsim"
eicrecon ${AIDE_WORKDIR}/log/sim_files/scan_$6_$5_p_$1_eta_$2_$3.root -Ppodio:output_file=${AIDE_WORKDIR}/log/sim_files/recon_scan_$6_$5_p_$1_eta_$2_$3.root -Pdd4hep:xml_files=${AIDE_WORKDIR}/share/epic/${DETECTOR_CONFIG}_$6.xml -Ppodio:output_include_collections=DRICHHits,MCParticles,DRICHRawHits,DRICHRawHitsAssociations,DRICHAerogelTracks,DRICHGasTracks,DRICHAerogelIrtCherenkovParticleID,DRICHGasIrtCherenkovParticleID,DRICHMergedIrtCherenkovParticleID
echo "done with eicrecon"
