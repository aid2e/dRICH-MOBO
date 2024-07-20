conda activate mobo
source setup.sh
python wrapper_slurm_basic.py -c optimize.config -d parameters.config
conda deactivate
