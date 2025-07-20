'''
    anomaly_detection_backend.py
    - Desc:
    Performs a two-stage ensemble anomaly detection process.
    
    Stage 1: An unsupervised ensemble (Isolation Forest, One-Class SVM, LOF)
             identifies initial anomaly candidates.
    Stage 2: A supervised ensemble is trained on the results of Stage 1. The
             performance of the supervised models is used to calculate F1-score-based
             weights for the unsupervised models.
    
    A final weighted, normalized score is produced, and a threshold is applied
    to identify high-confidence anomalies.

    Args:
        features_df (pd.DataFrame): A DataFrame where the index is the datetime
                                    and columns are the features to be used.

    Returns:
        dict: A dictionary containing:
              - 'results': DataFrame with predictions and scores.
              - 'unsupervised_f1': Dict of F1 scores for unsupervised models.
              - 'supervised_f1': Dict of F1 scores for supervised models.
'''

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

@st.cache_data
def get_weighted_anomaly_scores(features_df):
        
    if features_df.empty:
        return {}

    features_df.dropna(inplace=True)
    if features_df.empty:
        st.warning("Feature set is empty after removing rows with missing values.")
        return {}

    X = features_df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- STAGE 1: Unsupervised Anomaly Detection ---
    unsupervised_models = {
        'IsolationForest': IsolationForest(contamination='auto', random_state=42),
        'OneClassSVM': OneClassSVM(nu=0.05, kernel='rbf', gamma='auto'),
        'LOF': LocalOutlierFactor(contamination='auto')
    }
    unsupervised_predictions = pd.DataFrame(index=features_df.index)
    for name, model in unsupervised_models.items():
        y_pred = model.fit_predict(X_scaled)
        unsupervised_predictions[name] = (y_pred == -1).astype(int)

    # --- STAGE 2: Supervised Learning for Weighting ---
    unsupervised_consensus = (unsupervised_predictions.sum(axis=1) >= 2).astype(int)
    
    if unsupervised_consensus.sum() == 0:
        results = unsupervised_predictions.copy()
        results['normalized_score'] = 0.0
        results['is_anomaly'] = False
        return {'results': results, 'unsupervised_f1': {}, 'supervised_f1': {}}

    supervised_models = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'SVM_Linear': SVC(kernel='linear', random_state=42),
        'SVM_RBF': SVC(kernel='rbf', random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier()
    }

    supervised_predictions = pd.DataFrame(index=features_df.index)
    supervised_f1_scores = {}
    for name, model in supervised_models.items():
        model.fit(X_scaled, unsupervised_consensus)
        y_pred_supervised = model.predict(X_scaled)
        supervised_predictions[name] = y_pred_supervised
        # Score supervised models against the initial unsupervised consensus
        score = f1_score(unsupervised_consensus, y_pred_supervised)
        supervised_f1_scores[name] = score
        
    # --- F1-Score Based Weighting ---
    supervised_consensus = (supervised_predictions.sum(axis=1) >= (len(supervised_models) / 2)).astype(int)
    unsupervised_f1_scores = {}
    for model_name in unsupervised_predictions.columns:
        score = f1_score(supervised_consensus, unsupervised_predictions[model_name])
        unsupervised_f1_scores[model_name] = score

    # --- Final Weighted Anomaly Score Calculation ---
    total_f1_score_sum = sum(unsupervised_f1_scores.values())
    if total_f1_score_sum > 0:
        weighted_scores = np.zeros(len(features_df))
        for model_name, f1_weight in unsupervised_f1_scores.items():
            weighted_scores += unsupervised_predictions[model_name] * f1_weight
        normalized_scores = weighted_scores / total_f1_score_sum
    else:
        normalized_scores = np.zeros(len(features_df))

    # --- Compile Final Results ---
    results = unsupervised_predictions.copy()
    results['supervised_consensus_label'] = supervised_consensus
    results['normalized_score'] = normalized_scores
    results['is_anomaly'] = results['normalized_score'] > 0.5
    
    # Return all results and scores in a dictionary
    return {
        'results': results,
        'unsupervised_f1': unsupervised_f1_scores,
        'supervised_f1': supervised_f1_scores
    }

