#!/bin/bash

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