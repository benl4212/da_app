'''
ml_dashboard.py
- Desc: Anomaly detection and visualization page
'''
'''
ml_dashboard.py
- Desc: Anomaly detection and visualization page
'''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from streamlit import session_state as ss

# --- Import project-specific functions ---
from power_grid_utils import load_and_preprocess_power_grid_data
from feature_extraction import (
    extract_rolling_features, 
    extract_tsfresh_features,
    extract_rolling_fft,
    apply_custom_rolling_functions,
    calculate_lombscargle_power,
    calculate_cwt_energy,
    calculate_lyapunov,
    calculate_fractal_dimension,
    calculate_rqa_determinism,
    calculate_rqa_laminarity
)
from anomaly_detection_backend import get_weighted_anomaly_scores

# --- Page Configuration ---
st.set_page_config(
    page_title="ML Anomaly Detection",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Anomaly Detection")

# --- Utility function to clear results ---
def clear_anomaly_results():
    """Clears previous anomaly detection results from session state."""
    keys_to_delete = ["analysis_output", "model_input_features", "last_run_entity"]
    for key in keys_to_delete:
        if key in ss:
            del ss[key]

# --- Check if data is available in session state ---
if "uploaded_file" not in ss or not ss.get("is_power_grid_data", False):
    st.warning("Please upload a power grid data CSV on the Home page first.")
    st.stop()

# --- Load and Preprocess Data ---
if "master_df" not in ss or "entity_cols" not in ss:
    try:
        uploaded_file = ss["uploaded_file"]
        df, entity_cols = load_and_preprocess_power_grid_data(uploaded_file)
        if df.empty:
            st.error("Data could not be loaded or is empty.")
            st.stop()
        ss["master_df"] = df
        ss["entity_cols"] = entity_cols
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        st.stop()

df = ss["master_df"]
entity_cols = ss["entity_cols"]

# --- Sidebar for User Controls ---
with st.sidebar:
    st.header("⚙️ Analysis Controls")

    selected_entity = st.selectbox(
        "Select an Entity (Bus/Gen) to Analyze:",
        options=entity_cols,
        key="entity_selector",
        on_change=clear_anomaly_results
    )

    st.subheader("Feature Generation")
    
    with st.form("feature_generation_form"):
        st.write("Select feature sets to generate. This can be slow.")
        gen_basic = st.checkbox("Generate Basic Rolling Features", value=True)
        gen_fft = st.checkbox("Generate Rolling FFT Features")
        basic_window = st.slider("Basic Window Size", 2, 50, 10, key="basic_win")
        
        st.write("Select advanced features (optional):")
        gen_lombscargle = st.checkbox("Lomb-Scargle Power")
        gen_cwt = st.checkbox("CWT Energy")
        gen_lyapunov = st.checkbox("Lyapunov Exponent (Slow)")
        gen_fractal = st.checkbox("Fractal Dimension")
        gen_rqa_det = st.checkbox("RQA Determinism (Very Slow)")
        gen_rqa_lam = st.checkbox("RQA Laminarity (Very Slow)")
        advanced_window = st.slider("Advanced Window Size", 20, 100, 50, key="adv_win")
        
        submitted = st.form_submit_button("Generate Features")

    if submitted:
        clear_anomaly_results()
        with st.spinner("Generating features... Please wait."):
            tsfresh_features_df = extract_tsfresh_features(df)
            ss["tsfresh_features_df"] = tsfresh_features_df
            
            rolling_features_list = []
            if gen_basic:
                rolling_features_list.append(extract_rolling_features(df, basic_window))
            if gen_fft:
                rolling_features_list.append(extract_rolling_fft(df, basic_window))
            
            adv_funcs_to_run = {}
            if gen_lombscargle: adv_funcs_to_run['lombscargle_power'] = calculate_lombscargle_power
            if gen_cwt: adv_funcs_to_run['cwt_energy'] = calculate_cwt_energy
            if gen_lyapunov: adv_funcs_to_run['lyapunov_exp'] = calculate_lyapunov
            if gen_fractal: adv_funcs_to_run['fractal_dim'] = calculate_fractal_dimension
            if gen_rqa_det: adv_funcs_to_run['rqa_determinism'] = calculate_rqa_determinism
            if gen_rqa_lam: adv_funcs_to_run['rqa_laminarity'] = calculate_rqa_laminarity

            if adv_funcs_to_run:
                advanced_features = apply_custom_rolling_functions(
                    df, advanced_window, adv_funcs_to_run, selected_entity
                )
                rolling_features_list.append(advanced_features)
            
            if rolling_features_list:
                ss["rolling_features_df"] = pd.concat(rolling_features_list, axis=1).dropna(how='all')
            elif "rolling_features_df" in ss:
                del ss["rolling_features_df"]

    if "rolling_features_df" in ss and "tsfresh_features_df" in ss:
        st.subheader("Model Feature Selection")
        rolling_features_df = ss["rolling_features_df"]
        tsfresh_features_df = ss["tsfresh_features_df"]
        
        rolling_cols = rolling_features_df.columns.tolist()
        tsfresh_cols = tsfresh_features_df.columns.tolist()
        available_features = rolling_cols + tsfresh_cols
        
        default_rolling = [f for f in rolling_cols if selected_entity in f]
        default_selection = default_rolling + tsfresh_cols
        
        selected_features = st.multiselect(
            "Select features to use for anomaly detection:",
            options=available_features,
            default=default_selection
        )
        
        if st.button("🚀 Run Anomaly Detection"):
            if not selected_features:
                st.error("Please select at least one feature for the model.")
            else:
                with st.spinner("Running two-stage anomaly detection..."):
                    try:
                        selected_rolling = [f for f in selected_features if f in rolling_cols]
                        selected_tsfresh = [f for f in selected_features if f in tsfresh_cols]
                        
                        model_input_df = rolling_features_df[selected_rolling].copy()
                        
                        if selected_tsfresh and selected_entity in tsfresh_features_df.index:
                            tsfresh_values = tsfresh_features_df.loc[selected_entity, selected_tsfresh]
                            for feature_name, value in tsfresh_values.items():
                                model_input_df[feature_name] = value
                        
                        ss["analysis_output"] = get_weighted_anomaly_scores(model_input_df)
                        ss["model_input_features"] = model_input_df.columns.tolist()
                        ss["last_run_entity"] = selected_entity
                        st.success("Analysis complete!")
                    except Exception as e:
                        st.error(f"An error occurred during anomaly detection: {e}")

# --- Main Panel for Visualizations ---
if "analysis_output" not in ss:
    st.info("Configure and generate features in the sidebar, then click 'Run Anomaly Detection' to see results.")
else:
    analysis_output = ss["analysis_output"]
    
    if not analysis_output or 'results' not in analysis_output:
        st.warning("Analysis did not produce valid results. Please try again with different settings.")
    else:
        results_df = analysis_output['results']
        unsupervised_f1 = analysis_output['unsupervised_f1']
        supervised_f1 = analysis_output['supervised_f1']
        
        rolling_features_df = ss["rolling_features_df"]
        tsfresh_features_df = ss["tsfresh_features_df"]
        last_run_entity = ss["last_run_entity"]
        
        aligned_df = df.loc[results_df.index]
        
        st.header(f"Anomaly Analysis for: `{last_run_entity}`")
        st.dataframe(results_df.head())

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Unsupervised Model Weights (F1 Scores)")
            st.dataframe(pd.DataFrame.from_dict(unsupervised_f1, orient='index', columns=['F1 Score']))
        
        with col2:
            st.subheader("Supervised Model Performance (F1 Scores)")
            st.dataframe(pd.DataFrame.from_dict(supervised_f1, orient='index', columns=['F1 Score']))

        st.subheader("Time Series with Detected Anomalies")
        anomaly_points = results_df[results_df['is_anomaly'] == True]
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=aligned_df.index, y=aligned_df[last_run_entity], mode='lines', name='Original Data'))
        if not anomaly_points.empty:
            fig1.add_trace(go.Scatter(
                x=anomaly_points.index, y=aligned_df.loc[anomaly_points.index, last_run_entity],
                mode='markers', marker=dict(color='red', size=8, symbol='x'), name='Weighted Anomaly'
            ))
        fig1.update_layout(title=f"'{last_run_entity}' with Weighted Anomalies Overlay", xaxis_title="Datetime", yaxis_title="Value")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Feature & Anomaly Visualizations")
        model_features_used = ss.get("model_input_features", [])
        
        col_pca, col_heatmap = st.columns(2)
        
        with col_pca:
            if len(model_features_used) < 3:
                st.warning("Select at least 3 features for 3D PCA.")
            else:
                selected_rolling = [f for f in model_features_used if f in rolling_features_df.columns]
                selected_tsfresh = [f for f in model_features_used if f in tsfresh_features_df.columns]
                model_input_for_viz = rolling_features_df[selected_rolling].copy()
                if selected_tsfresh and last_run_entity in tsfresh_features_df.index:
                    tsfresh_values = tsfresh_features_df.loc[last_run_entity, selected_tsfresh]
                    for fname, val in tsfresh_values.items():
                        model_input_for_viz[fname] = val
                model_input_for_viz = model_input_for_viz.loc[results_df.index]
                
                pca = PCA(n_components=3)
                components = pca.fit_transform(model_input_for_viz)
                pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2', 'PC3'], index=results_df.index)
                pca_df['Anomaly'] = results_df['is_anomaly'].astype(str)
                pca_df['Score'] = results_df['normalized_score']
                fig2 = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Anomaly',
                                     color_discrete_map={'True': 'red', 'False': 'blue'},
                                     hover_data=['Score'], title="3D PCA of Features")
                fig2.update_traces(marker=dict(size=4))
                st.plotly_chart(fig2, use_container_width=True)

        with col_heatmap:
            st.write("**Time-Varying Feature Correlation**")
            tsfresh_feature_map = {
                'value__quantile__q_0.1': 'Quantile 0.1 (IF-1, OCSVM-1)',
                'value__quantile__q_0.2': 'Quantile 0.2 (IF-2, OCSVM-3)',
                'value__cwt_coefficients__coeff_11__w_5__widths_(2, 5, 10, 20)': 'CWT Coeff (IF-3)',
                'value__quantile__q_0.4': 'Quantile 0.4 (OCSVM-2, LOF-3)',
                'value__fft_coefficient__attr_"imag"__coeff_52': 'FFT Imag Coeff (LOF-1)',
                'value__change_quantiles__f_agg_"mean"__isabs_False__qh_0.2__ql_0.0': 'Change Quantile (LOF-2)'
            }

            # The heatmap ONLY shows rolling/advanced features, as static tsfresh
            # features will not have a meaningful correlation.
            selected_rolling = [f for f in model_features_used if f in rolling_features_df.columns]
            
            if not selected_rolling:
                st.info("No time-varying features were selected to display in heatmap.")
            else:
                heatmap_df = rolling_features_df[selected_rolling].copy()
                corr = heatmap_df.corr()
                fig3 = px.imshow(corr, text_auto=".2f", aspect="auto",
                                 title="Correlation of Time-Varying Features")
                st.plotly_chart(fig3, use_container_width=True)

            with st.expander("View TSFresh Feature Details"):
                st.write("This table shows the original `tsfresh` features that were pre-selected based on their importance scores from three different unsupervised models. Their impact is included in the model and the PCA plot.")
                csv_data = {
                    'Model': ['IsolationForest', 'IsolationForest', 'OneClassSVM', 'OneClassSVM', 'OneClassSVM', 'LOF', 'LOF', 'LOF'],
                    'Rank': [1, 2, 1, 2, 3, 1, 2, 3],
                    'Feature Name': [
                        'value__quantile__q_0.1',
                        'value__quantile__q_0.2',
                        'value__quantile__q_0.1',
                        'value__quantile__q_0.4',
                        'value__quantile__q_0.2',
                        'value__fft_coefficient__attr_"imag"__coeff_52',
                        'value__change_quantiles__f_agg_"mean"__isabs_False__qh_0.2__ql_0.0',
                        'value__quantile__q_0.4'
                    ],
                    'Alias': [
                        'Quantile 0.1 (IF-1, OCSVM-1)',
                        'Quantile 0.2 (IF-2, OCSVM-3)',
                        'Quantile 0.1 (IF-1, OCSVM-1)',
                        'Quantile 0.4 (OCSVM-2, LOF-3)',
                        'Quantile 0.2 (IF-2, OCSVM-3)',
                        'FFT Imag Coeff (LOF-1)',
                        'Change Quantile (LOF-2)',
                        'Quantile 0.4 (OCSVM-2, LOF-3)'
                    ],
                    'Importance': [0.096, 0.069, 0.106, 0.072, 0.062, 0.030, 0.029, 0.025]
                }
                ranking_df = pd.DataFrame(csv_data)
                st.dataframe(ranking_df)
