import streamlit as st
import requests
import pandas as pd
import os

st.set_page_config(page_title= "Causal Inference Analysis Tool", layout= "centered")
st.title("Causal Inference Analysis Tool")

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

st.header("Upload your CSV file")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

df = None
columns = []

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.success("File uploaded successfully")
  with st.expander("Click to show uploaded data", expanded=False):
    st.dataframe(df)
  columns = df.columns.tolist()

else:
  st.info("Awaiting CSV to upload")
  


