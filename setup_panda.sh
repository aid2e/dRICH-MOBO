#source  /gpfs02/eic/abashyal2/dRICH-MOBO3/pclient/etc/panda/panda_container.sh
#source /gpfs02/eic/abashyal2/dRICH-MOBO3/pclient/etc/panda/panda_setup.sh
source /gpfs/mnt/gpfs02/eic/abashyal2/miniconda3/envs/condaenv/etc/panda/panda_setup.sh 
export PANDA_URL_SSL=https://pandaserver01.sdcc.bnl.gov:25443/server/panda
export PANDA_URL=https://pandaserver01.sdcc.bnl.gov:25443/server/panda
export PANDACACHE_URL=https://pandaserver01.sdcc.bnl.gov:25443/server/panda
export PANDAMON_URL=https://pandamon01.sdcc.bnl.gov
export PANDA_AUTH=oidc
export PANDA_AUTH_VO=EIC
export PANDA_USE_NATIVE_HTTPLIB=1
export PANDA_BEHIND_REAL_LB=1

