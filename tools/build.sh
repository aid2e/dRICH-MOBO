# download eic-shell
curl --location https://get.epic-eic.org | bash

# get files
# git clone -b v1.15-drich-3mirror https://github.com/cpecar/EICrecon-drich-mobo.git
# git clone -b 24.07.0-drich-3mirror https://github.com/cpecar/epic-geom-drich-mobo.git
# git clone -b multi-mirror-irt https://github.com/eic/irt.git

git clone -b v1.19-drich-2mirror https://github.com/cpecar/EICrecon-drich-mobo.git
git clone -b 24.11.1-drich-2mirror https://github.com/cpecar/epic-geom-drich-mobo.git
git clone -b multi-mirror-irt https://github.com/eic/irt.git

source setup_new.sh

# run eic-shell to setup singularity
# bash eic-shell
# ./eic-shell -c jux_xl -v 24.08.1-stable

./eic-shell -c eic_xl -v 24.11.1-stable

# build
./build_epic.sh epic-geom-drich-mobo $EIC_SOFTWARE

./build_irt.sh irt $EIC_SOFTWARE

# source  eic-software/setup.sh
./build_eicrecon.sh EICrecon-drich-mobo $EIC_SOFTWARE

#
cd ProjectUtils/ePICUtils
make
