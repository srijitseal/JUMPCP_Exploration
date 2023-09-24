import os
import numpy as np
import pandas as pd
from tqdm import tqdm


from scipy.stats import randint, uniform
from pandarallel import pandarallel
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import ptitprince as pt
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, ttest_rel
from statannotations.Annotator import Annotator

import math
import sys
sys.path.append('/home/ss2686/JUMPCP')

import argparse
from scripts.evaluation_functions import evaluate_classifier, evaluate_regression, fold_error, optimize_threshold_j_statistic

# Initialize pandarallel for parallel processing
pandarallel.initialize()
pandarallel.initialize(progress_bar=False)

import gzip


data_path = '../data/processed_splits/'
# Define the path to your gzip-compressed image_features.csv.gz file
csv_file_path = '../data/JUMP_features/JUMP_features.csv.gz'


def create_molecule_dict(csv_file_path):
    molecule_dict = {}

    with gzip.open(csv_file_path, 'rt') as f:
        next(f)  # Skip the first line (header)
        for line in f:
            data = line.strip().split(',')
            smiles = data[0]
            features = np.array(data[1:299], dtype=float)
            molecule_dict[smiles] = features

    return molecule_dict

# Call create_molecule_dict once to create the dictionary
molecule_dict = create_molecule_dict(csv_file_path)

# Create a function to calculate Tanimoto similarities and means
def calculate_tanimoto_and_mean(row, combined_df, activity, knn, boolean_fingerprints):
    
    i = row.name
    if combined_df.iloc[i][activity] != 1:
        return None, None
    
    active_active_similarities = []
    active_inactive_similarities = []

    for j, index in enumerate(knn.kneighbors([boolean_fingerprints[i]])[1][0]):
        if i != index:
            similarity = 1 - knn.kneighbors([boolean_fingerprints[i]])[0][0][j]

            if combined_df.iloc[index][activity] == 1:
                active_active_similarities.append(similarity)
            elif combined_df.iloc[index][activity] == 0:
                active_inactive_similarities.append(similarity)

    if active_active_similarities:
        mean_active_active = np.median(sorted(active_active_similarities, reverse=True)[:5])
    else:
        mean_active_active = None

    if active_inactive_similarities:
        mean_active_inactive = np.median(sorted(active_inactive_similarities, reverse=True)[:5])
    else:
        mean_active_inactive = None

    return (1-mean_active_active), (1-mean_active_inactive)


from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance


def calculate_eucledian_and_mean(row, combined_df, activity, knn, descriptors):
    
    i = row.name
    if combined_df.iloc[i][activity] != 1:
        return None, None
    
    active_active_correlations = []
    active_inactive_correlations = []

    for j, index in enumerate(knn.kneighbors([descriptors[i]])[1][0]):
        if i != index:
            descriptor1 = descriptors[i]
            descriptor2 = descriptors[index]
            # Reshape data to fit the scaler's expected input
            x = [[i] for i in descriptor1]
            y = [[i] for i in descriptor2]
            # Normalize using Z-score normalization
            scaler = StandardScaler()
            x_normalized = scaler.fit_transform(x)
            y_normalized = scaler.transform(y)  # Use the same scaler to transform y
            # Flatten the data back to 1-dimensional arrays
            x_normalized = x_normalized.flatten()
            y_normalized = y_normalized.flatten()

            if combined_df.iloc[index][activity] == 1:
                # Compute Euclidean distance
                euclidean_dist = distance.euclidean(x_normalized, y_normalized)
                active_active_correlations.append(euclidean_dist)
                
            elif combined_df.iloc[index][activity] == 0:
                euclidean_dist = distance.euclidean(x_normalized, y_normalized)
                active_inactive_correlations.append(euclidean_dist)

    if active_active_correlations:
        mean_active_active = np.median(sorted(active_active_correlations, reverse=True)[:5])
    else:
        mean_active_active = None

    if active_inactive_correlations:
        mean_active_inactive = np.median(sorted(active_inactive_correlations, reverse=True)[:5])
    else:
        mean_active_inactive = None

    return (mean_active_active), (mean_active_inactive)


'''def calculate_pearson_and_mean(row, combined_df, activity, knn, descriptors):
    
    i = row.name
    if combined_df.iloc[i][activity] != 1:
        return None, None
    
    active_active_correlations = []
    active_inactive_correlations = []

    for j, index in enumerate(knn.kneighbors([descriptors[i]])[1][0]):
        if i != index:
            descriptor1 = descriptors[i]
            descriptor2 = descriptors[index]

            if combined_df.iloc[index][activity] == 1:
                correlation, _ = pearsonr(descriptor1, descriptor2)
                active_active_correlations.append(correlation)
            elif combined_df.iloc[index][activity] == 0:
                correlation, _ = pearsonr(descriptor1, descriptor2)
                active_inactive_correlations.append(correlation)

    if active_active_correlations:
        mean_active_active = np.median(sorted(active_active_correlations, reverse=True)[:])
    else:
        mean_active_active = None

    if active_inactive_correlations:
        mean_active_inactive = np.median(sorted(active_inactive_correlations, reverse=True)[:])
    else:
        mean_active_inactive = None

    return mean_active_active, mean_active_inactive
    
'''


def generate_cellpainting(smiles):
    return molecule_dict.get(smiles, np.zeros(298, dtype=float))

def generate_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    
    return np.array(fp)


results_fp = {}
results_cp = {}

results_significance = {}

# Initialize lists to store results
mean_tanimoto_active_active = []
mean_tanimoto_active_inactive = []

mean_eucledian_active_active= []
mean_eucledian_active_inactive= []

data = []

for dataset in os.listdir(data_path):   
    
    if dataset not in results_significance:
        results_significance[dataset] = {}
    
    
    if dataset != "PK_Lombardo":
        print(dataset)
        
        # Get all the file names for this dataset
        all_files = os.listdir(os.path.join(data_path, dataset))

        # Extract activity names by removing the _train.csv.gz or _test.csv.gz from file names
        activity_names = list(set([f.replace("_train.csv.gz", "").replace("_test.csv.gz", "") for f in all_files]))

        for activity in tqdm(activity_names, desc="Processing activities"):

            if activity not in results_significance[dataset]:
                results_significance[dataset][activity] = {}
            print(activity)

            train_path = os.path.join(data_path, dataset, f"{activity}_train.csv.gz")
            test_path = os.path.join(data_path, dataset, f"{activity}_test.csv.gz")

            train_df = pd.read_csv(train_path, compression='gzip')
            test_df = pd.read_csv(test_path, compression='gzip')

            # Combine train and test data
            combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
            
            print(len(combined_df))

            #STRUCTURAL
            
            # Generate Morgan fingerprints for the combined data
            fingerprints = combined_df['Standardized_SMILES'].parallel_apply(generate_fingerprints)
            fingerprints = np.array(fingerprints.to_list())

            threshold = 0.5  #Binarisation
            boolean_fingerprints = fingerprints > threshold
            #print("boolean_fingerprints complete")

            # Calculate Tanimoto similarity using Jaccard distance
            knn_fp = NearestNeighbors(n_neighbors=len(combined_df) - 1, metric='jaccard', n_jobs=1)  # Use Jaccard distance for Tanimoto similarity
            knn_fp.fit(boolean_fingerprints)
            #print("knn_fit complete")

            # Initialize lists to store mean similarities
            mean_tanimoto_active_active_activity = []
            mean_tanimoto_active_inactive_activity = []

            def apply_func_calculate_tanimoto_and_mean(row):
                return calculate_tanimoto_and_mean(row, combined_df, activity, knn_fp, boolean_fingerprints)

            # Apply the function to each row of combined_df in parallel
            results_fp = combined_df.parallel_apply(apply_func_calculate_tanimoto_and_mean, axis=1)

            # Separate the results into two lists
            mean_tanimoto_active_active = [result[0] for result in results_fp if result[0] is not None]
            mean_tanimoto_active_inactive = [result[1] for result in results_fp if result[1] is not None]

            # Raincloud plots
            #pal = "Set2"
            #sns.set(rc={'figure.figsize':(5,2), "figure.dpi":100}, font_scale=1)
            #sns.set_style("white")

            df_plot = pd.DataFrame({
                'Category': ['Active vs Active'] * len(mean_tanimoto_active_active) + ['Active vs Inactive'] * len(mean_tanimoto_active_inactive),
                'Mean Tanimoto Distance': mean_tanimoto_active_active + mean_tanimoto_active_inactive
            })

            
            '''pal = "colorblind"
            sns.set_style("white")

            ax=pt.half_violinplot(x = 'Mean Tanimoto Distance', y = 'Category', data = df_plot, palette = pal,
                 bw = .2, cut = 0.,scale = "area", width = .6, 
                 inner = None, orient = 'h')

            ax=sns.stripplot( x = 'Mean Tanimoto Distance', y = 'Category', data = df_plot, palette = pal,
                  edgecolor = "white",size = 3, jitter = 1, zorder = 0,
                  orient = 'h')

            ax=sns.boxplot( x = 'Mean Tanimoto Distance', y = 'Category', data = df_plot, color = "black",
                  width = .15, zorder = 10, showcaps = True,
                  boxprops = {'facecolor':'none', "zorder":10}, showfliers=True,
                  whiskerprops = {'linewidth':2, "zorder":10}, 
                  saturation = 1, orient = 'h')

            # Add significance annotations
            annotator = Annotator(ax, data=df_plot, y='Category', x='Mean Tanimoto Distance',
                                  pairs=[("Active vs Active", "Active vs Inactive")],
                                  order=['Active vs Active', 'Active vs Inactive'],
                                 orient='h')

            annotator.configure(test='t-test_ind', text_format='star', loc='outside')
            annotator.apply_and_annotate()
            '''

            # Extract data for both categories
            active_active_values = df_plot[df_plot['Category'] == 'Active vs Active']['Mean Tanimoto Distance']
            active_inactive_values = df_plot[df_plot['Category'] == 'Active vs Inactive']['Mean Tanimoto Distance']
            
            # Perform t-test
            t_stat, p_value = ttest_ind(active_active_values, active_inactive_values)
            # Print p-value
            results_significance[dataset][activity]['structural'] = {'t-statistic': t_stat, 'p-value': p_value}

            
            #Customize the plot
            #plt.xlabel("Mean Tanimoto Distance")
            #plt.ylabel("")
            #plt.xticks(rotation=0)
            #plt.show()
            
            
            #CELL PAINTING
            
            # Generate Cell Painting descriptors for the combined data
            cp_descriptors = combined_df['Standardized_SMILES'].parallel_apply(generate_cellpainting)
            cp_descriptors = np.array(cp_descriptors.to_list())
            
            # Initialize the K-nearest neighbors model for Eucledian distance
            knn_cp = NearestNeighbors(n_neighbors=len(combined_df) - 1, metric='correlation', n_jobs=1)  # Use Euclidean distance for correlations
            knn_cp.fit(cp_descriptors)  
            
            # Initialize lists to store mean correlations
            mean_eucledian_active_active_activity = []
            mean_eucledian_active_inactive_activity = []
            
            def apply_func_calculate_eucledian_and_mean(row):
                return calculate_eucledian_and_mean(row, combined_df, activity, knn_cp, cp_descriptors)

            # Apply the function to each row of combined_df in parallel
            results_cp = combined_df.parallel_apply(apply_func_calculate_eucledian_and_mean, axis=1)
            
            # Separate the results into two lists
            mean_eucledian_active_active = [result[0] for result in results_cp if result[0] is not None]
            mean_eucledian_active_inactive = [result[1] for result in results_cp if result[1] is not None]
            
            # Raincloud plots

            #print(activity)

            #pal = "Set2"
            #sns.set(rc={'figure.figsize':(5,2), "figure.dpi":100}, font_scale=1)
            #sns.set_style("white")

            df_plot = pd.DataFrame({
                'Category': ['Active vs Active'] * len(mean_eucledian_active_active) + ['Active vs Inactive'] * len(mean_eucledian_active_inactive),
                'Mean Eucledian Distance': mean_eucledian_active_active + mean_eucledian_active_inactive
            })

            '''
            pal = "colorblind"
            sns.set_style("white")

            ax=pt.half_violinplot( x = 'Mean Eucledian Distance', y = 'Category', data = df_plot, palette = pal,
                 bw = .2, cut = 0.,scale = "area", width = .6, 
                 inner = None, orient = 'h')

            ax=sns.stripplot( x = 'Mean Eucledian Distance', y = 'Category', data = df_plot, palette = pal,
                  edgecolor = "white",size = 3, jitter = 1, zorder = 0,
                  orient = 'h')

            ax=sns.boxplot( x = 'Mean Eucledian Distance', y = 'Category', data = df_plot, color = "black",
                  width = .15, zorder = 10, showcaps = True,
                  boxprops = {'facecolor':'none', "zorder":10}, showfliers=True,
                  whiskerprops = {'linewidth':2, "zorder":10}, 
                  saturation = 1, orient = 'h')

            # Add significance annotations
            annotator = Annotator(ax, data=df_plot, y='Category', x='Mean Eucledian Distance',
                                  pairs=[("Active vs Active", "Active vs Inactive")],
                                  order=['Active vs Active', 'Active vs Inactive'],
                                 orient='h')

            annotator.configure(test='t-test_ind', text_format='star', loc='outside')
            annotator.apply_and_annotate()
            
            '''
            
            # Extract data for both categories
            active_active_values = df_plot[df_plot['Category'] == 'Active vs Active']['Mean Eucledian Distance']
            active_inactive_values = df_plot[df_plot['Category'] == 'Active vs Inactive']['Mean Eucledian Distance']

            # Perform t-test
            t_stat, p_value = ttest_ind(active_active_values, active_inactive_values)
            # Print p-value
            results_significance[dataset][activity]['image'] = {'t-statistic': t_stat, 'p-value': p_value}


            # Customize the plot
            #plt.xlabel("Mean Eucledian Distance")
            #plt.ylabel("")
            #plt.xticks(rotation=0)
            #plt.show()

            # Create a list to hold the rows of the dataframe


# Iterate through the dictionary to extract the data
for task, activities in results_significance.items():
    for activity, features in activities.items():
        for featureset, values in features.items():
            row = {
                'dataset': dataset,
                'activity': activity,
                'featureset': featureset,
                't-statistic': values['t-statistic'],
                'p-value': values['p-value']
            }
            data.append(row)

# Convert the list of rows to a dataframe
df = pd.DataFrame(data)
df.to_csv("Plot_comparsions_similarityv2.csv", index=False)

