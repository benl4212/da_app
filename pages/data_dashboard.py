'''
    data_dashboard.py
    - Desc: main data visualization page
'''

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from streamlit import session_state as ss

# Import the utility and new feature extraction functions
from power_grid_utils import load_and_preprocess_power_grid_data
from feature_extraction import extract_rolling_features

# --- Reusable Plotting Function ---
def plot_histogram(data, title, ax, color=None):
    """
    Generates a seaborn histogram on a given matplotlib axis.
    
    Args:
        data (pd.Series): The data to plot.
        title (str): The title for the plot.
        ax (matplotlib.axes.Axes): The axis object to draw the plot on.
        color (str, optional): The color for the plot.
    """
    sns.histplot(data=data, ax=ax, kde=True, color=color)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels for better fit
    
st.set_page_config(
    page_title="Power Grid Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Power Grid Data Analysis & Features")

# --- Check if data is available in session state ---
if "uploaded_file" not in ss or not ss.get("is_power_grid_data", False):
    st.warning("Please upload a power grid data CSV on the Home page first.")
    st.stop()

# --- Load and Preprocess Data ---
uploaded_file = ss["uploaded_file"]

try:
    df, entity_cols = load_and_preprocess_power_grid_data(uploaded_file)
    
    if df.empty:
        st.error("Data could not be loaded or is empty. Please check the file on the Home page.")
        st.stop()

    st.success("Data loaded and preprocessed!")
    st.dataframe(df.head())

    # --- Create a copy for modifications ---
    df_analysis = df.copy()

    # --- Feature Extraction ---
    st.subheader("Feature Engineering")
    window_size = st.slider("Select Rolling Window Size:", min_value=2, max_value=50, value=10)
    with st.spinner("Calculating rolling window features..."):
        df_features = extract_rolling_features(df, window_size)
    st.success("Feature extraction complete!")
    st.dataframe(df_features.head())

    # --- System-Wide Aggregated Analysis ---
    st.subheader("System-Wide Aggregated Analysis")
    
    total_cols = [col for col in df_analysis.columns if 'total' in col]
    
    if total_cols:
        df_analysis['total_of_totals'] = df_analysis[total_cols].sum(axis=1)
        total_cols.append('total_of_totals')
    elif entity_cols:
        st.info("No 'total' columns found. Creating a system-wide total from all individual entities.")
        df_analysis['system_wide_total'] = df_analysis[entity_cols].sum(axis=1)
        total_cols = ['system_wide_total']
    else:
        total_cols = []

    if total_cols:
        selected_total = st.selectbox(
            "Select an aggregated metric to analyze:",
            options=total_cols
        )

        if selected_total:
            hist_col1, hist_col2 = st.columns(2)
            variance_feature = f'{selected_total}_variance'
            sum_feature = f'{selected_total}_sum'

            if selected_total in ['total_of_totals', 'system_wide_total']:
                temp_features = extract_rolling_features(df_analysis[[selected_total]], window_size)
                df_features_combined = pd.concat([df_features, temp_features], axis=1)
            else:
                df_features_combined = df_features

            with hist_col1:
                if variance_feature in df_features_combined.columns:
                    fig_hist1, ax_hist1 = plt.subplots(figsize=(8, 4))
                    plot_histogram(df_features_combined[variance_feature], f'Distribution of {variance_feature}', ax_hist1)
                    fig_hist1.tight_layout()
                    st.pyplot(fig_hist1)
                    plt.close(fig_hist1)

            with hist_col2:
                if sum_feature in df_features_combined.columns:
                    fig_hist2, ax_hist2 = plt.subplots(figsize=(8, 4))
                    plot_histogram(df_features_combined[sum_feature], f'Distribution of {sum_feature}', ax_hist2, color='green')
                    fig_hist2.tight_layout()
                    st.pyplot(fig_hist2)
                    plt.close(fig_hist2)
    else:
        st.warning("No aggregated 'total' columns found or generated to display system-wide analysis.")

    # --- Individual Entity Analysis ---
    st.subheader("Individual Entity Analysis")
    
    if not entity_cols:
        st.warning("No individual entities (Bus/Gen) found to analyze.")
    else:
        max_index = len(entity_cols) - 1
        
        # Text input for selecting an entity by its index
        entity_index_str = st.text_input(
            f"Enter an entity index to analyze:",
            value="0"
        )
        st.caption(f"Valid range: 0 to {max_index}")

        try:
            entity_index = int(entity_index_str)
            
            # Validate the entered index
            if 0 <= entity_index <= max_index:
                selected_entity = entity_cols[entity_index]
                st.write(f"#### Now analyzing: `{selected_entity.upper()}`")

                # Create two rows of columns for the four plots
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)

                with col1:
                    st.write("**Original Data Trend**")
                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    ax1.plot(df_analysis.index, df_analysis[selected_entity], label=selected_entity, color='purple')
                    ax1.set_title(f'Original Data Over Time')
                    ax1.legend()
                    ax1.grid(True)
                    fig1.tight_layout()
                    st.pyplot(fig1)
                    plt.close(fig1)

                with col2:
                    st.write("**Rolling Standard Deviation (Volatility)**")
                    feature_col_name = f'{selected_entity}_std_dev'
                    if feature_col_name in df_features.columns:
                        fig2, ax2 = plt.subplots(figsize=(8, 4))
                        ax2.plot(df_features.index, df_features[feature_col_name], label=feature_col_name, color='crimson')
                        ax2.set_title(f'Volatility Over Time')
                        ax2.legend()
                        ax2.grid(True)
                        fig2.tight_layout()
                        st.pyplot(fig2)
                        plt.close(fig2)

                with col3:
                    st.write("**Distribution of Rolling Variance**")
                    variance_feature = f'{selected_entity}_variance'
                    if variance_feature in df_features.columns:
                        fig3, ax3 = plt.subplots(figsize=(8, 4))
                        plot_histogram(df_features[variance_feature], 'Distribution of Rolling Variance', ax3)
                        fig3.tight_layout()
                        st.pyplot(fig3)
                        plt.close(fig3)

                with col4:
                    st.write("**Distribution of Rolling Sum**")
                    sum_feature = f'{selected_entity}_sum'
                    if sum_feature in df_features.columns:
                        fig4, ax4 = plt.subplots(figsize=(8, 4))
                        plot_histogram(df_features[sum_feature], 'Distribution of Rolling Sum', ax4, color='green')
                        fig4.tight_layout()
                        st.pyplot(fig4)
                        plt.close(fig4)
            else:
                st.warning(f"Index out of range. Please enter a number between 0 and {max_index}.")

        except ValueError:
            st.warning("Invalid input. Please enter a valid integer.")

except Exception as e:
    st.error(f"An error occurred on the dashboard page: {e}")
    st.warning("Please verify the uploaded file's content and structure.")
