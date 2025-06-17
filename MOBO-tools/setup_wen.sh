## FROM WEN
source /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fyingtsai/eic_xl:24.11.1/opt/conda/setup_mamba.sh;
source /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fyingtsai/eic_xl:24.11.1/opt/conda/dRICH-MOBO//MOBO-tools/setup_new.sh;
command -v singularity &> /dev/null || export SINGULARITY=/cvmfs/oasis.opensciencegrid.org/mis/singularity/current/bin/singularity;
export AIDE_HOME=$(pwd)
export PWD_PATH=$(pwd)
export SINGULARITY_OPTIONS="--bind /cvmfs:/cvmfs,$(pwd):$(pwd)";
export SIF=/cvmfs/singularity.opensciencegrid.org/eicweb/eic_xl:24.11.1-stable 
export SINGULARITY_BINDPATH=/cvmfs,/afs
env