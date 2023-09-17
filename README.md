# JUMPCP Chemical Space and Data Processing

Welcome to the repository dedicated to exploring the chemical space of the JUMPCP dataset and further data processing. Here, you'll find a collection of Jupyter Notebooks, scripts, raw and processed datasets essential for the detailed investigation of JUMPCP compounds.

## Repository Structure:

### Folder: 01_standardise_smiles
Contains the tools for standardizing the SMILES notation of compounds.

- `01_process_raw_datasets.py`: Python script to process raw datasets and smiles standardisation.
- `02_Explore_Datasets.ipynb`: Jupyter Notebook for dataset exploration.

### Folder: 02_Explore_JUMP_compounds
Houses the notebooks and data files for exploring the JUMPCP compounds.

- `01_DrugSpace_Exploration.ipynb`: Explore the drug space of the dataset.
- `02_Explore_Chemical_Space_PCA_TSNE.ipynb`: Investigate the chemical space via PCA and t-SNE methods.
- `03_Find_Overlaps_with_data_JUMPCP_and_save_datasets.ipynb`: Find overlaps in data and save datasets.

### Folder: 03_splitting_data
This section helps in data stratification.

- `01_data_splitting.py`: Main script for splitting data.

### Folder: scripts
Utility scripts for data processing.

- `stratified_split_helper.py`: Helps in stratifying the splits.
- `standardise_smiles.py`: Standardizes the SMILES notation.

### data
A comprehensive data directory with raw, processed, overlapping, and split datasets from various sources like JUMPCP, BBBP, sider, tox21, muv, HIV, and more. This directory also contains notebooks for specific datasets.

### main
Main executable for the project (Further details needed).

### README.md
(You're here!) Introduction and guide to the repository.

### LICENSE
The license details for the code and data in this repository.

### Shell Scripts
- `process_datasets.sh`
- `process_datasets_slurm.sh`

## Contributions:
Feel free to contribute to this repository. Raise issues or submit pull requests if you find something that can be improved or added. Remember, collaboration is the essence of open source!
