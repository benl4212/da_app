'''
    tfresh_pca.py
    - Desc: finding the most anomalous generators from the tfresh extracted features (tfresh.py)
'''

from sklearn.processing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


# 1. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(extracted_features)

# 2. Apply PCA. n_components can be tuned.
# A common choice is to keep 95% of the variance.
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# 3. Reconstruct the data from the PCA representation
X_reconstructed = pca.inverse_transform(X_pca)

# 4. Calculate the reconstruction error for each generator
reconstruction_error = np.mean(np.square(X_scaled - X_reconstructed), axis=1)

# 5. Create a DataFrame with the anomaly scores
anomaly_scores = pd.DataFrame(
    {'reconstruction_error': reconstruction_error},
    index=extracted_features.index
)

# 6. Find the top 100 most anomalous generators
top_anomalies = anomaly_scores.sort_values(
    by='reconstruction_error',
    ascending=False
).head(100)

top_anomalies.to_csv('anomalous_gens.csv')

print("--- Top 100 Most Anomalous Generators ---")
print(top_anomalies)