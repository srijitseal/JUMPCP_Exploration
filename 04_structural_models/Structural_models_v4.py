#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
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
from sklearn.model_selection import StratifiedKFold
import math


import sys
sys.path.append('/home/ss2686/JUMPCP')
import argparse
from scripts.evaluation_functions import evaluate_classifier, evaluate_regression, fold_error, optimize_threshold_j_statistic


# Path where your data is stored
data_path = '../data/processed_splits/'

results = {}

def generate_fingerprints(smiles_list):
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.array(fps)


# Assuming PK dataset is regression and others are classification
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

        train_df = pd.read_csv(train_path, compression='gzip')
        test_df = pd.read_csv(test_path, compression='gzip')

        X_train = generate_fingerprints(train_df['Standardized_SMILES'])
        X_test = generate_fingerprints(test_df['Standardized_SMILES'])
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
                n_jobs=40,)
            
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
            
        
            
        # Save results at each step
        pd.DataFrame(results).T.to_csv('./structural_model_resultsv4.csv')
              

# Save results
results_df = pd.DataFrame(results).T.reset_index(drop=False)
results_df = results_df.rename(columns={'index': 'endpoint'})
results_df.to_csv('./structural_model_resultsv4.csv', index=False)





