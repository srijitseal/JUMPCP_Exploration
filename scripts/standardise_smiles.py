#!/usr/bin/env python
# coding: utf-8

from rdkit.Chem import inchi
from rdkit import Chem
import os
os.environ["PYTHONWARNINGS"] = "ignore"  # Suppress warnings

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

from dimorphite_dl.dimorphite_dl import DimorphiteDL
from rdkit.Chem import AddHs
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd

def standardize_jumpcp(smiles):
    # follows the steps in
    # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    # as described **excellently** (by Greg) in
    # https://www.youtube.com/watch?v=eWTApNX8dJQ
    try: 
        mol = Chem.MolFromSmiles(smiles)
        #print(smiles)
        
        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol) 
        #print(Chem.MolToSmiles(clean_mol))
        
        # if many fragments, get the "parent" (the actual mol we are interested in) 
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # try to neutralize molecule
        uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        #print(uncharged_parent_clean_mol)
        
        # use pH 7.4 https://git.durrantlab.pitt.edu/jdurrant/dimorphite_dl/
        dimorphite = DimorphiteDL(min_ph=7.4, max_ph=7.4, pka_precision=0)
        protonated_smiles = dimorphite.protonate(Chem.MolToSmiles(uncharged_parent_clean_mol))

        #print("protonated_smiles")
        
        if len(protonated_smiles) > 0:
                protonated_smile = protonated_smiles[0]

        protonated_mol = Chem.MolFromSmiles(protonated_smile)
        #protonated_mol= AddHs(protonated_mol)
        #protonated_smile = Chem.MolToSmiles(protonated_mol)


        # attempt is made at reionization at this step
        # at 7.4 pH

        te = rdMolStandardize.TautomerEnumerator() # idem
        taut_uncharged_parent_clean_mol = te.Canonicalize(protonated_mol)
     
        return  Chem.MolToSmiles(taut_uncharged_parent_clean_mol)
    
    except: 
        
        return "Cannot_do"

def inchi_from_standardised_smile(value):

    try: return Chem.MolToInchi(Chem.MolFromSmiles(value))
    except: return "Cannot_do"

def process_data(data_path, smiles_variable):
    # Read the data
    if data_path.endswith('.gz'):
        data = pd.read_csv(data_path, compression='gzip')
    else:
        data = pd.read_csv(data_path)

    # Apply the standardize_oasis function
    data['Standardized_SMILES'] = data[smiles_variable].parallel_apply(standardize_jumpcp)
    # Convert standardized SMILES to InChI
    data['Standardized_InChI'] = data['Standardized_SMILES'].parallel_apply(inchi_from_standardised_smile)

    
    # Filter out SMILES strings that couldn't be standardized

    filtered_data = data[data['Standardized_SMILES'] != "Cannot_do"]
    # Filter out InChI strings that couldn't be standardized
    filtered_data = filtered_data[filtered_data['Standardized_InChI'] != "Cannot_do"].reset_index(drop=True)
    
    return filtered_data

def save_data(df, save_path):
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    df.to_csv(save_path, index=False)