'''
    "power_grid_utils.py"
    -backend for power grid data
    
	    POWER DATA SCHEMAS

    1.    ACTIVISg10k_renewable_time_series_MW
        | Date | Time | Num Renewable | Total solar Gen | Total wind Gen | Gen 10691 #1 Max MW - Gen 77289 #1 Max MW | 10760 - 80100

    2.    ACTIVISg2000_load_time_series_MVAR
        | Date | Time | Num Load | Total MW Load | Total Mvar Load | Bus 1001 #1 MVAR - Bus 8160 #1 MVAR

    3.    ACTIVISg2000_load_time_series_MW
        | Date | Time | Num Load | Total MW Load | Total Mvar Load | Bus 1001 #1 MW - Bus 8160 #1 MW

    4.    ACTIVISg2000_renewable_time_series_MW
        | Date | Time | Num Renewable | Total solar Gen | Total wind Gen | Gen 1011 #1 MW - Gen 5399 #1 MW

    5.    ACTIVSg10k_load_time_series_MVAR
        | Date | Time | Num Load | Total MW Load | Total Mvar Load | Bus 10001 #1 MVAR - Bus 80084 #1 MVAR

    6.    ACTIVSg10k_load_time_series_MW
        | Date | Time | Num Load | Total MW Load | Total Mvar Load | Bus 10001 #1 MW - Bus 80084 #1 MW
        

'''

import pandas as pd
import streamlit as st

@st.cache_data # Cache the function to avoid re-loading/processing on every rerun
def load_and_preprocess_power_grid_data(uploaded_file):
    """
    Loads and preprocesses time series data from an uploaded CSV file,
    dynamically handling different power grid data schemas with robust error handling.

    Args:
        uploaded_file: The file-like object from st.file_uploader().

    Returns:
        A tuple containing:
        - pd.DataFrame: The preprocessed data with a Datetime index.
        - list: A list of identified entity columns (e.g., 'Bus ...', 'Gen ...').
    """
    if uploaded_file is None:
        return pd.DataFrame(), []

    # --- Data Loading ---
    # The DtypeWarning suggests using low_memory=False to help pandas infer types
    # more efficiently on files with mixed types. This is a key performance fix.
    try:
        # Attempt to read with header=1 first.
        df = pd.read_csv(uploaded_file, header=1, low_memory=False)
    except Exception:
        # If that fails, rewind and try with a standard header.
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, header=0, low_memory=False)
        except Exception as e:
            st.error(f"Could not parse the CSV file. Please ensure it's a valid CSV. Error: {e}")
            return pd.DataFrame(), []

    # --- Column Name Normalization ---
    df.columns = [str(col).strip().lower() for col in df.columns]

    # --- Datetime Index Creation ---
    if 'datetime' in df.columns:
        df['temp_datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    elif 'date' in df.columns and 'time' in df.columns:
        try:
            # Be more explicit with the format to avoid the UserWarning and improve speed.
            df['temp_datetime'] = pd.to_datetime(
                df['date'] + ' ' + df['time'],
                format='%m/%d/%Y %I:%M:%S %p',
                errors='coerce'
            )
        except Exception:
            # If the primary format fails, fall back to letting pandas infer.
            df['temp_datetime'] = pd.to_datetime(
                df['date'] + ' ' + df['time'],
                errors='coerce'
            )
    else:
        st.error("Could not find a 'datetime' column, or a combination of 'date' and 'time' columns.")
        return pd.DataFrame(), []

    # Drop rows where datetime conversion failed and set the index.
    df.dropna(subset=['temp_datetime'], inplace=True)
    df = df.set_index('temp_datetime')
    df.index.name = 'datetime'
    df = df.drop(columns=['date', 'time', 'datetime'], errors='ignore')

    # --- Dynamic Column Identification (using normalized names) ---
    entity_cols = [
        col for col in df.columns
        if col.startswith('bus ') or col.startswith('gen ') or col.isdigit()
    ]
    entity_cols = sorted(list(set(entity_cols)))

    # --- Dynamic Numeric Conversion ---
    for col in df.columns:
        if col in entity_cols or 'load' in col or 'gen' in col:
             df[col] = pd.to_numeric(df[col], errors='coerce')

    return df, entity_cols
