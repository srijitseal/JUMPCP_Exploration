#!/bin/bash

#SBATCH --job-name=sim_analysis           # Name of the job
#SBATCH --partition=icelake              # Name of the partition you want to submit to
#SBATCH --account=BENDER-SL3-CPU         # Account/Project name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks per node
#SBATCH --cpus-per-task=10               # Number of CPUs per task
#SBATCH --time=11:59:00                  # Time limit hrs:min:sec (set the "wall time")
#SBATCH --output=sim_analysis_%j.out
#SBATCH --error=sim_analysis_%j.err

# Load necessary modules (if any, e.g., Python, GCC, etc.)
# module load python/3.8

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate my-rdkit-env

# Run your Python script
python 01_Plot_comparsions.py
