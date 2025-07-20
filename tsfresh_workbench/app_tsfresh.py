'''
    tsfresh.py
    - Desc: extracting features from a csv to be used in conjuction
    with tfresh_pca.py
'''

# Started 7/7/2025
# grid_data.py for data analysis on csv file:
# "C:\Users\benl4\Documents\Spring_2025\Machine_Learning\Grid_Data\ACTIVSg_Time_Series\Time Series\ACTIVSg10k_load_time_series_MVAR.csv"
# WSL PATH:
# "/mnt/c/Users/benl4/Documents/Spring_2025/Machine_Learning/Grid_Data/ACTIVSg_Time_Series/Time_Series/ACTIVSg2000_renewable_time_series_MW.csv"
# RELATIVE PATH:
# Grid_Data\ACTIVSg_Time_Series\Time Series\ACTIVSg10k_load_time_series_MVAR.csv


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tsfresh import extract_features
import os

# --- Loading  ---
data_file = r"/mnt/c//Users/benl4/Documents/Spring_2025/Machine_Learning/Grid_Data/ACTIVSg_Time_Series/Time_Series/ACTIVISg2000_renewable_time_series_MW.csv"
df = pd.read_csv(data_file, header=1)

# --- Cleaning ---
df.columns = [str(col).strip().lower() for col in df.columns]

if 'date' in df.columns and 'time' in df.columns:
    df['temp_datetime'] = pd.to_datetime(
        df['date'] + ' ' + df['time'],
        errors='coerce',
        format='%m/%d/%Y %I:%M:%S %p'
    )

df.dropna(subset=['temp_datetime'], inplace=True)
df = df.set_index('temp_datetime')
df.index.name = 'datetime'

# Keep track of original time columns to exclude them from numeric conversion
original_time_cols = ['date', 'time', 'datetime']
df = df.drop(columns=original_time_cols, errors='ignore')

# --- FIX: Convert all data columns to a numeric type ---
# Loop through all columns in the DataFrame
for col in df.columns:
    # Use pd.to_numeric, turning anything that can't be converted into NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Now, fill any NaN values (including those created by the step above)
df.fillna(0, inplace=True)

# --- Reshape the DataFrame from wide to long format ---
df_long = df.stack().reset_index()
df_long.columns = ['datetime', 'id', 'value']

print(f"Shape of the DataFrame passed to tsfresh: {df_long.shape}")

# --- tsfresh call ---
if not df_long.empty:
    extracted_features = extract_features(
        df_long,
        column_id='id',
        column_sort='datetime',
        column_value='value'
    )
    print("Extracted Features Head:")
    print(extracted_features.head())
else:
    print("The DataFrame is still empty after processing. Please check the source CSV file for valid numeric data.")


# Save the features to multiple formats
# Saved column headers to a text file in ubuntu with: awk 'NR==1' data.csv > tfresh_features.txt
extracted_features.to_csv('extracted_features.csv')
extracted_features.to_pickle('extracted_features.pkl')



