#!/bin/bash
#SBATCH --job-name=network_inference
#SBATCH --output=arr_%A-%a.out
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --array=2-19

module load python/3.7.3
module use $HOME/.modulefiles
module load miniconda3
cd $HOME/network-dynamics-analyzer/joint-tracking
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate idtxl
python full_networktest_parallel.py pre-border ${SLURM_ARRAY_TASK_ID}
