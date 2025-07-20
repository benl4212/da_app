'''
    tfresh_top_features.py
    - Desc: uses the extracted_features.pkl from tfresh.py to find the most important 
    features according to the unsupervised algorithms: IF, OCSVM, and LOF
    
    Random Forest Regressor is used as a proxy to use feature_importances_
    
    Outputs ranking of top features to tfresh_top_features.csv
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor


# 1. Load the extracted features
try:
    extracted_features = pd.read_pickle('extracted_features.pkl')
except FileNotFoundError:
    print("Error: 'extracted_features.pkl' not found.")
    exit()


# 2. Data cleaning
# 2 A. Replace all infinite values with NaN
extracted_features.replace([np.inf, -np.inf], np.nan, inplace=True)

# 2 B. Drop any column that contains at least one NaN
extracted_features.dropna(axis=1, inplace=True)

# 2 C. Remove constant columns by low variance threshold
variance = extracted_features.var()
extracted_features = extracted_features[variance[variance > 1e-8].index]


# 3 Preparing the features
# 3 A. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(extracted_features)
X_scaled_df = pd.DataFrame(X_scaled, columns=extracted_features.columns)

# 3 B. List to hold the results from each model
all_top_features = []

# 4. Initialize the model dictionary
models = {
    'IsolationForest': IsolationForest(contamination='auto', random_state=42),
    'OneClassSVM': OneClassSVM(nu=0.1, kernel="rbf", gamma='auto'),
    'LOF': LocalOutlierFactor(n_neighbors=20, novelty=True, contamination='auto')
}


# 5. Run the model dictionary
for model_name, model in models.items():
    print(f"Running {model_name}...")
    
    # Fit the anomaly detection model
    model.fit(X_scaled_df)
    
    # Get anomaly scores for each model (LOF is different)
    if model_name == 'LOF':
        # For LOF, higher score (less negative) is more anomalous    
        anomaly_scores = model.negative_outlier_factor_ * -1
    else:
        # For IF and OCSVM, higher score is more anomalous after inversion
        anomaly_scores = model.decision_function(X_scaled_df) * -1
    
    
    # 6. Use RFR as a proxy to find feature importances
    rf_proxy = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_proxy.fit(X_scaled_df, anomaly_scores)
    importances = rf_proxy.feature_importances_
    
    # 7. Sort a data frame with the features from RFR
    feature_importance_df = pd.DataFrame({
        'feature': extracted_features.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    # 8. Save the top 3 features for the current model
    top_3 = feature_importance_df.head(3).copy()
    top_3['model'] = model_name
    top_3['rank'] = [1, 2, 3]
    
    all_top_features.append(top_3)    
    
# 9. Combine feature ranking from each model
final_results = pd.concat(all_top_features)
final_results = final_results[['model', 'rank', 'feature', 'importance']]
final_results.to_csv('tfresh_features_model_ranking.csv', index=False)

print("\n-------------------------------------------------------------------\n")
print("\nFeature Ranking for: \n1 _ _ _ IF\n2 _ _ _ OCSVM\n3 _ _ _ LOF")
print("\nSaved _ _ _ 'tfresh_features_model_ranking.csv'")