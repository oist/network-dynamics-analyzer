#!/bin/bash
#SBATCH -t 2-6
#SBATCH --mem=8G
#SBATCH -c 16

module load python/3.7.3
module use $HOME/.modulefiles
module load miniconda3
cd $HOME/network-dynamics-analyzer/joint-tracking
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate idtxl
python full_networktest.py
