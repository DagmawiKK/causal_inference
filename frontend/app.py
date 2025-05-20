import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(page_title="Causal Inference Analysis Tool", layout="centered")
st.title("Causal Inference Analysis Tool")

FASTAPI_URL = "http://localhost:8000/api"

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
    if st.session_state.df_data is not None and len(st.session_state.columns) > 2:
        columns = st.session_state.columns

        col1, col2 = st.columns(2)
        with col1:
            treatment_col = st.selectbox("Treatment column", columns, key="treatment_col_psm")
        with col2:
            outcome_col = st.selectbox("Outcome column (Y)", [col for col in columns if col != treatment_col], key="outcome_col_psm")

        default_confounders = [col for col in columns if col != treatment_col and col != outcome_col]

        confounders = st.multiselect("Covariates (X)", default_confounders, key="confounders_psm")

        col3, col4 = st.columns(2)
        with col3:
            n_neighbors = st.number_input("n_neighbors", min_value=1, value=1, step=1, key="n_neighbors_psm")
        with col4:
            random_state = st.number_input("random_state", value=42, step=1, key="random_state_psm")

        st.markdown("### Advanced Options (optional)")
        scale_features = st.checkbox("Scale features for matching", value=True, key="scale_features_psm")
        show_prop_hist = st.checkbox("Show propensity score distribution plot", value=True, key="show_prop_hist_psm")
        show_matched_pair_hist = st.checkbox("Show outcome distribution of matched pairs", value=False, key="show_matched_pair_hist_psm")
        use_caliper = st.checkbox("Use caliper for matching", value=False, key="use_caliper_psm")
        caliper = st.number_input("Caliper (max distance allowed)", min_value=0.0, value=0.1, step=0.01, disabled=not use_caliper, key="caliper_psm")

        if st.button("Analyze Data", key="analyze_psm"):
            if not treatment_col or not outcome_col or not confounders:
                st.error("Please input treatement, outcome and confounder columns")
            else:
                payload = {
                    "data": st.session_state.df_data,
                    "treatment_col": treatment_col,
                    "outcome_col": outcome_col,
                    "confounders": confounders,
                    "n_neighbors": n_neighbors,
                    "random_state": random_state,
                    "scale_features": scale_features,
                    "use_caliper": use_caliper,
                    "caliper": caliper if use_caliper else 0.0,
                    "show_prop_hist": show_prop_hist,
                    "show_matched_pair_hist": show_matched_pair_hist
                }

                try:
                    with st.spinner("Loading..."):
                        response = requests.post(f"{FASTAPI_URL}/psm", json=payload, timeout=120)
                    results = response.json()
                    if results.get("message"):
                        if "Error" in results["message"] or "No " in results["message"]:
                            st.error(results["message"])
                        elif not "complete" in results["message"]:
                            st.info(results["message"])

                    if results.get("att") is not None:
                        st.write(f"**ATT (Matched units):** {results['att']:.4f}")
                    if results.get("ate_raw") is not None:
                        st.write(f"**ATE (Raw difference):** {results['ate_raw']:.4f}")
                    if results.get("num_matched_pairs") is not None:
                        st.write(f"**Number of matched units/pairs:** {results['num_matched_pairs']}")
                    
                    # Conclusions
                    col5, col6 = st.columns(2)
                    with col5:
                        if results['ate_raw'] > 0:
                            st.success(f"ATE conclusion: On Average, on the entire population, the treatment increases the outcome by {results['ate_raw']:.4f}")
                        elif results['ate_raw'] < 0:
                            st.error(f" ATE conclusion: On Average, on the entire population, the treatment decreases the outcome by {results['ate_raw']:.4f}")
                        else:
                            st.warning("ATE conclusion: On Average, on the entire population, the treatment has no effect on the outcome")
                    with col6:
                        if results['att'] > 0:
                            st.success(f"ATT conclusion: On Average, on the treated, the treatment increases the outcome by {results['att']:.4f}")
                        elif results['att'] < 0:
                            st.error(f"ATT conclusion: On Average, on the treated, the treatment decreases the outcome by {results['att']:.4f}")
                        else:
                            st.warning("ATT conclusion: On Average, on the treated, the treatment has no effect on the outcome")

                    # Display DataFrame with PSM info
                    if results.get("full_data_with_psm_info"):
                        with st.expander("Show full data with propensity scores and matching info"):
                            df_psm_info = pd.DataFrame(results["full_data_with_psm_info"])
                            st.dataframe(df_psm_info)

                    # Display matched pairs table
                    if results.get("matched_pairs_table"):
                        with st.expander("Show all matched pairs (treated to control original indices)"):
                            df_matched_pairs = pd.DataFrame(results["matched_pairs_table"])
                            st.dataframe(df_matched_pairs)

                    # Display plots
                    if results.get("propensity_score_plot_data") and show_prop_hist:
                        plot_data = results["propensity_score_plot_data"]
                        fig, ax = plt.subplots(figsize=(7, 4))
                        if plot_data.get("treated_values"):
                            ax.hist(plot_data["treated_values"], bins=20, alpha=0.6, label=plot_data["legend_labels"][0] if plot_data.get("legend_labels") else "Treated", color="tab:blue")
                        if plot_data.get("control_values"):
                            ax.hist(plot_data["control_values"], bins=20, alpha=0.6, label=plot_data["legend_labels"][1] if plot_data.get("legend_labels") else "Control", color="tab:orange")
                        ax.set_xlabel(plot_data.get("xlabel", "Propensity Score"))
                        ax.set_ylabel(plot_data.get("ylabel", "Count"))
                        ax.set_title(plot_data.get("title", "Propensity Score Distribution"))
                        ax.legend()
                        st.pyplot(fig)

                    if results.get("matched_outcome_plot_data") and show_matched_pair_hist:
                        plot_data = results["matched_outcome_plot_data"]
                        fig2, ax2 = plt.subplots(figsize=(7, 4))
                        if plot_data.get("treated_values"):
                            ax2.hist(plot_data["treated_values"], bins=10, alpha=0.6, label=plot_data["legend_labels"][0] if plot_data.get("legend_labels") else "Matched Treated", color="tab:purple")
                        if plot_data.get("control_values"):
                            ax2.hist(plot_data["control_values"], bins=10, alpha=0.6, label=plot_data["legend_labels"][1] if plot_data.get("legend_labels") else "Matched Control", color="tab:green")
                        ax2.set_xlabel(plot_data.get("xlabel", "Outcome"))
                        ax2.set_ylabel(plot_data.get("ylabel", "Count"))
                        ax2.set_title(plot_data.get("title", "Outcome Distribution of Matched Units"))
                        ax2.legend()
                        st.pyplot(fig2)
                except Exception as e:
                    st.error(f"An error occurred while communicating with the backend: {e}")
    else:
        st.info("Please upload your data")

with dml_tab:
    st.header("Double Machine Learning (DML)")
    if st.session_state.df_data is not None and len(st.session_state.columns) > 2:
        columns = st.session_state.columns

        col1, col2 = st.columns(2)
        with col1:
            treatment_col = st.selectbox("Treatment column", columns, key="treatment_col_dml")
        with col2:
            outcome_col = st.selectbox("Outcome column (Y)", [col for col in columns if col != treatment_col], key="outcome_col_dml")

        default_confounders = [col for col in columns if col != treatment_col and col != outcome_col]

        confounders = st.multiselect("Covariates (X)", default_confounders, key="confounders_dml")
        col3, col4 = st.columns(2)
        with col3:
            n_splits = st.number_input("n_splits", min_value=2, value=5, step=1, key="n_splits_dml")
        with col4:
            random_state = st.number_input("random_state", value=42, step=1, key="random_state_dml")

        scale_features = st.checkbox("Scale features for matching", value=True, key="scale_features_dml")
        view_outcome_plot = st.checkbox("Show outcome distribution plot", value=False, key="view_outcome_plot_dml")

        if st.button("Analyze Data", key="analyze_dml"):
            if not treatment_col or not outcome_col or not confounders:
                st.error("Please input treatement, outcome and confounder columns")
            else:
                payload = {
                    "data": st.session_state.df_data,
                    "treatment_col": treatment_col,
                    "outcome_col": outcome_col,
                    "confounders": confounders,
                    "n_splits": n_splits,
                    "random_state": random_state,
                    "scale_features": scale_features,
                    "view_outcome_plot": view_outcome_plot
                }

                try:
                    with st.spinner("Loading..."):
                        response = requests.post(f"{FASTAPI_URL}/dml", json=payload, timeout=120)
                    results = response.json()

                    if results.get("att") is not None:
                        st.write(f"**ATT (Matched units):** {results['att']:.4f}")
                    if results.get("ate") is not None:
                        st.write(f"**ATE (Raw difference):** {results['ate']:.4f}")
                    col5, col6 = st.columns(2)

                    # Conclusions
                    with col5:
                        if results['ate'] > 0:
                            st.success(f"ATE conclusion: On Average, on the entire population, the treatment increases the outcome by {results['ate']:.4f}")
                        elif results['ate'] < 0:
                            st.error(f" ATE conclusion: On Average, on the entire population, the treatment decreases the outcome by {results['ate']:.4f}")
                        else:
                            st.warning("ATE conclusion: On Average, on the entire population, the treatment has no effect on the outcome")
                    with col6:
                        if results['att'] > 0:
                            st.success(f"ATT conclusion: On Average, on the treated, the treatment increases the outcome by {results['att']:.4f}")
                        elif results['att'] < 0:
                            st.error(f"ATT conclusion: On Average, on the treated, the treatment decreases the outcome by {results['att']:.4f}")
                        else:
                            st.warning("ATT conclusion: On Average, on the treated, the treatment has no effect on the outcome")
                    
                    # Output plot
                    if results.get("outcome_plot"):
                        plot_data = results["outcome_plot"]
                        fig, ax = plt.subplots(figsize=(7, 4))
                        if plot_data.get("treated_values"):
                            ax.hist(plot_data["treated_values"], bins=20, alpha=0.6, label=plot_data["legend_labels"][0] if plot_data.get("legend_labels") else "Treated", color="tab:blue")
                        if plot_data.get("control_values"):
                            ax.hist(plot_data["control_values"], bins=20, alpha=0.6, label=plot_data["legend_labels"][1] if plot_data.get("legend_labels") else "Control", color="tab:orange")
                        ax.set_xlabel(plot_data.get("xlabel", "Outcome"))
                        ax.set_ylabel(plot_data.get("ylabel", "Count"))
                        ax.set_title(plot_data.get("title", "Outcome Distribution by Treatment Group"))
                        ax.legend()
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"An error occurred while communicating with the backend: {e}")
    else:
        st.info("Please upload your data")