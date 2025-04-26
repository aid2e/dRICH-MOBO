#!/bin/bash  
#SBATCH --chdir=/hpc/group/vossenlab/rck32/eic/dRICH-MOBO/MOBO-tools/
#SBATCH --job-name=run_optimization_0_linear_ratioz
#SBATCH --output=/hpc/group/vossenlab/rck32/eic/dRICH-MOBO/MOBO-tools/run_opt_slurm/err_out/%j.out
#SBATCH --error=/hpc/group/vossenlab/rck32/eic/dRICH-MOBO/MOBO-tools/run_opt_slurm/err_out/%j.err
#SBATCH -p common
#SBATCH --account=vossenlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-user=rck32@duke.edu
echo began job
source ~/.mobo-bashrc
source run_optimization.sh