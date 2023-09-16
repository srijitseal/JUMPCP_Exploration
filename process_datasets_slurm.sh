#!/bin/bash

#SBATCH --job-name=process_datasets      # Name of the job
#SBATCH --partition=icelake               # Name of the partition you want to submit to
#SBATCH --account=BENDER-SL3-CPU         # Account/Project name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks per node
#SBATCH --cpus-per-task=76               # Number of CPUs per task
#SBATCH --time=11:59:00                  # Time limit hrs:min:sec (set the "wall time")
#SBATCH --output=process_datasets_%j.out
#SBATCH --error=process_datasets_%j.err

# Load necessary modules (if any, e.g., Python, GCC, etc.)
# module load python/3.8

# Activate the conda environment

source ~/miniconda3/etc/profile.d/conda.sh
conda activate my-rdkit-env

datasets=("clintox" "bace" "BBBP" "HIV" "muv" "sider" "tox21" "toxcast" "JUMPCP_compounds")

for dataset in "${datasets[@]}"; do
     
    echo ${dataset}
    
    if [[ -e "data/raw/${dataset}/${dataset}.csv" ]]; then
        file_format="csv"
    elif [[ -e "data/raw/${dataset}/${dataset}.csv.gz" ]]; then
        file_format="csv.gz"
    else
        echo "No dataset found for ${dataset} in expected formats."
        continue
    fi
    
    python 01_standardise_smiles/01_process_raw_datasets.py \
           --raw_path "data/raw/${dataset}/${dataset}.${file_format}" \
           --save_path "data/processed/${dataset}/${dataset}_processed.csv.gz" \
           --smiles_variable 'smiles'
done
