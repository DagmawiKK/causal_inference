import streamlit as st
import requests
import pandas as pd
import os


st.set_page_config(page_title= "Causal Inference Analysis Tool", layout= "centered")
st.title("Causal Inference Analysis Tool")

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# File upload ui
st.header("Upload your CSV file")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

df = None
columns = []

if "df_data" not in st.session_state:
  st.session_state.df_data = None
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.session_state.df_data = df.to_dict(orient="records")
  st.session_state.columns = df.columns.tolist()
  st.success("File uploaded successfully")
  with st.expander("Click to show uploaded data", expanded=False):
    st.dataframe(df)

else:
  st.info("Awaiting CSV to upload")

# Tab choice ui
psm_tab, dml_tab = st.tabs(["PSM", "DML"])

with psm_tab:
  st.header("Propensity Score Matching (PSM)")
  if df is not None and len(columns) > 2:
    col1, col2 = st.columns(2)
    with col1:
      treatment_col = st.selectbox("Treatment column", columns)
    with col2:
      outcome_col = st.selectbox("Outcome column (Y)", [col for col in columns if col != treatment_col])
    
    default_confounders = [col for col in columns if col != treatment_col and col != outcome_col]

    confounders = st.multiselect("Covariates (X)", default_confounders)

    col3, col4 = st.columns(2)
    with col3:
      n_neighbors = st.number_input("n_neighbors", min_value = 1, value = 1, step = 1, key = "n_neighbors_new")
    with col4:
      random_state = st.number_input("random_state", value = 42, step = 1, key = "random_state_new")

    if st.button("Analyze Data"):
      if not treatment_col or not outcome_col or not confounders:
        st.error("Please input treatement, outcome and confounder columns")
      # else:
      #   try:
      #     request_file = uploaded_file.getvalue()



with dml_tab:
  st.header("Double Machine Learning (DML)")
  st.info("To be implemented")

