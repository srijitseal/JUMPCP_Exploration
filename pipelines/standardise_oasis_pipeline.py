#!/usr/bin/env python
# coding: utf-8

# In[2]:


from rdkit.Chem import inchi
from rdkit import Chem
from pandarallel import pandarallel
pandarallel.initialize()
from dimorphite_dl.dimorphite_dl import DimorphiteDL
from rdkit.Chem import AddHs
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize_oasis(smiles):
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

