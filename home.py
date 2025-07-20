'''
    home.py
    - Desc: home page for DA app. Displays steps for users to input their data or use the example data file
    Points to the dashboard page where visualization and anomaly detection are.
'''
import streamlit as st
from streamlit import session_state as ss # Shorter alias for session_state
import os

st.set_page_config(
    page_title="App Home",
    page_icon="⚡",
    layout="centered"
)

st.title("Welcome to DA App ⚡\n(Data Analysis App)")
st.write("Upload your data or select an example file to begin your analysis.")

# --- Data Source Selection ---
st.subheader("1. Choose Your Data Source")

source_option = st.radio(
    "Select a data source:",
    ("Upload a file", "Use an example file"),
    key="data_source_selector",
    horizontal=True
)

# --- Logic to handle the selected data source ---
if source_option == "Upload a file":
    uploaded_file = st.file_uploader(
        "Choose a CSV or ZIP file",
        type=["csv", "zip"],
        help="Upload your power grid data. The app expects 'Date' and 'Time' columns."
    )
    if uploaded_file is not None:
        ss["uploaded_file"] = uploaded_file
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    else:
        # Clear the session state if the uploader is cleared
        if "uploaded_file" in ss and not isinstance(ss["uploaded_file"], str):
             del ss["uploaded_file"]

elif source_option == "Use an example file":
    example_dir = "example"
    try:
        # Find all .csv and .zip files in the example directory
        example_files = [f for f in os.listdir(example_dir) if f.endswith(('.csv', '.zip'))]
        
        if not example_files:
            st.warning(f"No example files (.csv, .zip) found in the '{example_dir}' directory.")
        else:
            selected_example = st.selectbox("Choose an example file:", example_files)
            if selected_example:
                # Store the full path to the selected file in session state.
                # Downstream functions can use this path directly.
                file_path = os.path.join(example_dir, selected_example)
                ss["uploaded_file"] = file_path
                st.success(f"Selected example file: '{selected_example}'")

    except FileNotFoundError:
        st.error(f"Error: The '{example_dir}' directory was not found in your project.")


# --- Data Type Selection ---
st.subheader("2. Confirm Data Type")

is_power_grid_data = st.checkbox(
    "This is Power Grid Data",
    value=True, # Default to True as it's the main app function
    help="Confirm that the selected data is power grid time series.",
    disabled=True # Disable as only this type is supported
)
ss["is_power_grid_data"] = is_power_grid_data


# --- Navigation Instructions ---
st.subheader("3. Start Analysis")

# Enable navigation instructions if a file is ready
if "uploaded_file" in ss and ss["uploaded_file"] is not None:
    st.info("Your data is ready. Use the menu on the left to navigate to a dashboard.")
else:
    st.info("Please select a data source to proceed.")

