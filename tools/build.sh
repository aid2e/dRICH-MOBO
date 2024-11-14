# download eic-shell
curl --location https://get.epic-eic.org | bash

# get files
git clone -b v1.15-drich-3mirror https://github.com/cpecar/EICrecon-drich-mobo.git
git clone -b 24.07.0-drich-3mirror https://github.com/cpecar/epic-geom-drich-mobo.git
git clone -b multi-mirror-irt https://github.com/eic/irt.git

# run eic-shell to setup singularity
# bash eic-shell
./eic-shell -c jug_xl -v 24.08.1-stable

# build
bash build_eicrecon.sh EICrecon-drich-mobo/ $EIC_SOFTWARE/
bash build_epic.sh epic-geom-drich-mobo/ $EIC_SOFTWARE/
bash build_irt.sh irt/ $EIC_SOFTWARE/

#
cd ProjectUtils/ePICUtils
make
