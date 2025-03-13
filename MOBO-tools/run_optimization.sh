conda activate /hpc/group/vossenlab/rck32/eic/dRICH-MOBO/
source setup.sh
python wrapper_slurm_basic.py -c optimize.config -d parameters.config
conda deactivate
