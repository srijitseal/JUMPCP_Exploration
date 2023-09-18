#!/usr/bin/env python
# coding: utf-8


from sklearn.metrics import mean_squared_error, roc_auc_score, matthews_corrcoef, average_precision_score, confusion_matrix
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_curve
import math


def evaluate_classifier(true_labels, predictions, probs):
    auc = roc_auc_score(true_labels, probs)
    mcc = matthews_corrcoef(true_labels, predictions)
    avg_precision = average_precision_score(true_labels, probs)
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    spe = tn / (tn + fp)
    sen = tp / (tp + fn)
    ba = (spe + sen)/2
    return {'Held_out_TP': tp, 'Held_out_TN': tn,
            'Held_out_FP': fp, 'Held_out_FN': fn,
            'Held_out_BA': ba,
            'Held_out_AUC': auc, 'Held_out_MCC': mcc, 
            'Held_out_AUCPR': avg_precision, 'Held_out_Specificity': spe,
            'Held_out_Sensitivity': sen}

def fold_error(true_values, predictions, activity):
    if activity != "fraction_unbound_in_plasma_fu":    
        ratio = 10 **predictions / 10 **true_values
        
    if activity == "fraction_unbound_in_plasma_fu":    
        ratio = predictions / true_values
        
    adjusted_ratio = np.where(ratio < 1, 1/ratio, ratio)
    return adjusted_ratio

def evaluate_regression(true_values, predictions, activity):
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    r2 = np.corrcoef(true_values, predictions)[0, 1] ** 2
    fold_errors = fold_error(true_values, predictions, activity)
    # Calculate the percentage of data points within 2-fold error
    within_2_fold = np.sum((fold_errors <= 2.0)) / len(fold_errors) * 100
    # Calculate the percentage of data points within 3-fold error
    within_3_fold = np.sum((fold_errors <= 3.0)) / len(fold_errors) * 100
    # Calculate the percentage of data points within 5-fold error
    within_5_fold = np.sum((fold_errors <= 5.0)) / len(fold_errors) * 100


    median_fold_error = np.median(fold_error(true_values, predictions, activity))

    return {'Held_out_R2': r2, 'Held_out_RMSE': rmse,
            "Held_out_median_fold_error": median_fold_error,
           "Held_out_perc_2_fold": within_2_fold,
           "Held_out_perc_3_fold": within_3_fold,
            "Held_out_perc_5_fold": within_5_fold,}

def optimize_threshold_j_statistic(y_true, y_probs):
    # Example usage:
    # y_true is the true labels (binary)
    # y_probs is the predicted probabilities
    # best_threshold = optimize_threshold_j_statistic(y_true, y_probs)

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    # Calculate J statistic values
    j_statistic = tpr - fpr
    
    # Find the index of the threshold that maximizes J statistic
    best_threshold_idx = j_statistic.argmax()
    
    # Get the best threshold
    best_threshold = thresholds[best_threshold_idx]
    
    return best_threshold