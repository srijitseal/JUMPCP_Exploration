import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, matthews_corrcoef, average_precision_score, confusion_matrix
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import roc_curve
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint, uniform
from pandarallel import pandarallel
from sklearn.model_selection import StratifiedKFold

import math
import sys
sys.path.append('/home/ss2686/JUMPCP')

import argparse
from scripts.evaluation_functions import evaluate_classifier, evaluate_regression, fold_error, optimize_threshold_j_statistic

# Initialize pandarallel for parallel processing
pandarallel.initialize()
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

# Assuming you call create_molecule_dict once to create the dictionary
molecule_dict = create_molecule_dict(csv_file_path)

def generate_cellpainting(smiles):
    return molecule_dict.get(smiles, np.zeros(298, dtype=float))

#Exammple usage:

#smiles_list = [
#    'CCc1nccn1-c1cccc(C2CCC[NH+]2C(=O)c2ccc(OCC[NH+](C)C)cc2)n1',
#    'O=C1NCCC[NH+]1Cc1ccc(Cl)cc1',
#    'O=C1NC(=O)c2cc(Nc3ccccc3)c(Nc3ccccc3)cc21',
#    'CCCn1nccc1S(=O)(=O)[NH+]1CC2CCC1C[NH2+]C2',
#    'CCNC(=O)CC1N=C(c2ccc(Cl)cc2)c2cc(OC)ccc2-n2c(C)nnc21'
#]

# Create a DataFrame with the SMILES
#smiles_df = pd.DataFrame({'SMILES': smiles_list})

#X_train = smiles_df['SMILES'].parallel_apply(generate_cellpainting)
#X_train = np.array(X_train.to_list())
#X_train

# Assuming image-based dataset is regression and others are classification
results = {}

for dataset in os.listdir(data_path):
    
    # Exclude hidden files or directories like .ipynb_checkpoints
    if dataset.startswith('.'):
        continue
    print(dataset)


    # Get all the file names for this dataset
    all_files = os.listdir(os.path.join(data_path, dataset))

    # Extract activity names by removing the _train.csv.gz or _test.csv.gz from file names
    activity_names = list(set([f.replace("_train.csv.gz", "").replace("_test.csv.gz", "") for f in all_files]))

    for activity in tqdm(activity_names, desc="Processing activities"):
        
        train_path = os.path.join(data_path, dataset, f"{activity}_train.csv.gz")
        test_path = os.path.join(data_path, dataset, f"{activity}_test.csv.gz")

        train_df = pd.read_csv(train_path, compression='gzip')#.sample(20)
        test_df = pd.read_csv(test_path, compression='gzip')#.sample(20)

        X_train = train_df['Standardized_SMILES'].parallel_apply(generate_cellpainting)
        X_train = np.array(X_train.to_list())
        
        X_test = test_df['Standardized_SMILES'].parallel_apply(generate_cellpainting)
        X_test = np.array(X_test.to_list())
        
        y_train = train_df[activity]
        y_test = test_df[activity]


        if dataset == "PK_Lombardo":
            # Regression
            regressor = RandomForestRegressor(n_jobs=-1)
            
            if activity != "fraction_unbound_in_plasma_fu":
                # Log-transform the target variable for non-"fraction_unbound_in_plasma_fu" activities
                y_train = np.log10(y_train)
                y_test = np.log10(y_test)

            print(X_train.shape)
            print(X_test.shape)
            print(len(y_train))
            print(len(y_test))
            
            # Regression
            # Define parameter search space
            param_dist_regression = {
                'max_depth': randint(10, 20),
                'max_features': randint(40, 50),
                'min_samples_leaf': randint(5, 15),
                'min_samples_split': randint(5, 15),
                'n_estimators': [200, 300, 400, 500, 600],
                'bootstrap': [True, False],
                'n_jobs': [40],
                'random_state': [42]
            }

            # Create a HalvingRandomSearchCV object
            regression_search = HalvingRandomSearchCV(
                regressor,
                param_distributions=param_dist_regression,
                factor=3,  
                cv=5,  # Number of cross-validation folds
                random_state=42,
                verbose=1,
                n_jobs=40,  # Number of parallel jobs
                scoring='neg_root_mean_squared_error'  # Scoring metric
            )

            regression_search.fit(X_train, y_train)
            best_model = regression_search.best_estimator_
            
            # Train the best model on the full training data
            best_model.fit(X_train, y_train)
            
            # Make predictions on training and test data
            predictions_train = best_model.predict(X_train)
            predictions_test = best_model.predict(X_test)

            print(len(predictions_train))
            print(len(predictions_test))
            
            
            # Calculate CV R-squared using the best model
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, n_jobs=20, scoring='r2')

            results[activity] = {
                'CV_R2_mean': np.mean(cv_scores),
                'CV_R2_std': np.std(cv_scores),
                **evaluate_regression(y_test, predictions_test, activity)
            }
            
            
        else:
            # Classification
            model = RandomForestClassifier(n_jobs=40)
            
            # Hyperparameter Optimization
            param_dist_classification = {'max_depth': randint(10, 20),
                          'max_features': randint(40, 50),
                          'min_samples_leaf': randint(5, 15),
                          'min_samples_split': randint(5, 15),
                          'n_estimators':[200, 300, 400, 500, 600],
                          'bootstrap': [True, False],
                          'oob_score': [False],
                          'random_state': [42],
                          'criterion': ['gini', 'entropy'],
                          'n_jobs': [40],
                          'class_weight' : [None, 'balanced']
                         }
            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   
            
            classification_search = HalvingRandomSearchCV(
                model,
                param_dist_classification,
                factor=3,
                cv=inner_cv,
                random_state=42,
                verbose=1,
                n_jobs=40)
            
            classification_search.fit(X_train, y_train)
            best_model = classification_search.best_estimator_
            
            # Random Over-sampling and Threshold Optimization
            sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
            
            pipeline = Pipeline(steps=[('sampler', sampler), ('model', best_model)])
            pipeline.fit(X_train, y_train)
            
            # Predict using threshold-optimized model
            predictions_train = pipeline.predict(X_train)
            probs_train = pipeline.predict_proba(X_train)[:, 1]
            probs_test = pipeline.predict_proba(X_test)[:, 1]
            
            # Use the optimize_threshold_j_statistic function to find the best threshold
            best_threshold = optimize_threshold_j_statistic(y_train, probs_train)
            #Apply the best threshold to get binary predictions on the test data
            predictions_test = (probs_test >= best_threshold).astype(int)
            
            # Calculate CV AUC using threshold-optimized model
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, n_jobs=-1, scoring='roc_auc')

            results[activity] = {
                'CV_AUC_mean': np.mean(cv_scores),
                'CV_AUC_std': np.std(cv_scores),
                **evaluate_classifier(y_test, predictions_test, probs_test)
            }
   
        
        #break

        #Save results at each step
        pd.DataFrame(results).T.to_csv('./cellpainting_model_resultsv4.csv')
            
        

# Save results
results_df = pd.DataFrame(results).T.reset_index(drop=False)
results_df = results_df.rename(columns={'index': 'endpoint'})
results_df.to_csv('./cellpainting_model_resultsv4.csv', index=False)

