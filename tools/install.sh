#!/bin/bash

source /afs/cern.ch/user/w/wguan/workdisk/conda/setup_mini.sh

# conda env create --prefix=/afs/cern.ch/user/w/wguan/workdisk/dRICH-MOBO/.conda -f tools/conda_requirements.yml

# conda activate /afs/cern.ch/user/w/wguan/workdisk/dRICH-MOBO/.conda

conda env create --prefix=/afs/cern.ch/user/w/wguan/workdisk/eic/conda  -f /afs/cern.ch/user/w/wguan/workdisk/eic/dRICH-MOBO/tools/conda_requirements.yml
conda activate /afs/cern.ch/user/w/wguan/workdisk/eic/conda

# required by AX
# conda install -y -c conda-forge postgresql

# pip install ax ax.metrics
# git clone https://github.com/axonchisel/ax_metrics.git
# fix the setup
# python setup.py install

# ax is not ax-platform
pip install ax-platform

pip install torch pandas numpy matplotlib wandb botorch

