import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from api.schemas import PSMResponse, PlotData, MatchedPairInfo

def psm(
    data: list,
    treatment_col: str,
    outcome_col: str,
    confounders: list,
    n_neighbors: int,
    random_state: int,
    scale_features: bool,
    use_caliper: bool,
    caliper: float | None,
    show_prop_hist: bool,
    show_matched_pair_hist: bool
) -> PSMResponse:
  try:
    df = pd.DataFrame(data)
    X = df[confounders]
    treatment = df[treatment_col]
    Y = df[outcome_col]

    X_enc = pd.get_dummies(X, drop_first=True)
    if scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_enc)
    else:
        X_scaled = X_enc.values

    #  Calculate Propensity
    lr = LogisticRegression(random_state=int(random_state), solver="lbfgs", max_iter=1000)
    lr.fit(X_scaled, treatment)
    prop_score = lr.predict_proba(X_scaled)[:, 1]

    df_psm = df.copy()
    df_psm["_propensity_score"] = prop_score

    treated_idx = treatment == 1
    control_idx = treatment == 0
    treated = df_psm[treated_idx].copy()
    control = df_psm[control_idx].copy()

    df_psm["_matched_id"] = np.nan
    df_psm["_match_distance"] = np.nan
    df_psm["_matched_group"] = np.nan

    # Match
    nbrs = NearestNeighbors(n_neighbors=int(n_neighbors), algorithm="ball_tree")
    nbrs.fit(control[["_propensity_score"]])

    # For matched control table
    matched_controls = []
    matched_treated = []
    matched_distances = []
    matched_pairs = []

    for treated_index, treated_row in treated.iterrows():
      distances, indices = nbrs.kneighbors( pd.DataFrame([[treated_row["_propensity_score"]]], columns=["_propensity_score"]) )
      matched_ids = []
      for d, idx in zip(distances[0], indices[0]):
        if use_caliper and d > caliper:
            continue
        control_idx_in_df = control.index[idx]
        matched_controls.append(control.loc[control_idx_in_df, outcome_col])
        matched_treated.append(treated_row[outcome_col])
        matched_distances.append(d)
        matched_ids.append(control_idx_in_df)
        # Mark both treated and control as matched
        df_psm.at[treated_index, "_matched_id"] = str(control_idx_in_df)
        df_psm.at[treated_index, "_match_distance"] = float(d)
        df_psm.at[treated_index, "_matched_group"] = "Treated"
        df_psm.at[control_idx_in_df, "_matched_id"] = str(treated_index)
        df_psm.at[control_idx_in_df, "_match_distance"] = float(d)
        df_psm.at[control_idx_in_df, "_matched_group"] = "Control"
        matched_pairs.append(MatchedPairInfo(
            treated_index=str(treated_index),
            control_index=str(control_idx_in_df),
            distance=float(d)
        ))

    matched_controls = np.array(matched_controls)
    matched_treated = np.array(matched_treated)

    att = (matched_treated - matched_controls).mean()
    num_matched = len(matched_controls)

    nbrs_atc = NearestNeighbors(n_neighbors=int(n_neighbors), algorithm="ball_tree")
    nbrs_atc.fit(treated[["_propensity_score"]])

    atc_effects = []
    for control_index, control_row in control.iterrows():
        distances, indices = nbrs_atc.kneighbors(pd.DataFrame([[control_row["_propensity_score"]]], columns=["_propensity_score"]))
        for d, idx in zip(distances[0], indices[0]):
            if use_caliper and d > caliper:
                continue
            treated_idx_in_df = treated.index[idx]
            matched_outcome = treated.loc[treated_idx_in_df, outcome_col]
            atc_effects.append(matched_outcome - control_row[outcome_col])

    atc = np.mean(atc_effects) if atc_effects else np.nan

    n_treated = len(treated)
    n_control = len(control)
    n_total = n_treated + n_control

    ate = (n_treated / n_total) * att + (n_control / n_total) * atc

    prop_hist_data = None
    matched_outcome_hist_data = None

    if show_prop_hist:
        prop_hist_data = PlotData(
            treated_values=df_psm.loc[treated_idx, "_propensity_score"].tolist(),
            control_values=df_psm.loc[control_idx, "_propensity_score"].tolist(),
            title="Propensity Score Distribution",
            xlabel="Propensity Score",
            ylabel="Count",
            legend_labels=["Treated", "Control"]
        )

    if show_matched_pair_hist:
        matched_outcome_hist_data = PlotData(
            treated_values=matched_treated.tolist(),
            control_values=matched_controls.tolist(),
            title="Outcome Distribution of Matched Units",
            xlabel="Outcome",
            ylabel="Count",
            legend_labels=["Matched Treated", "Matched Control"]
        )
  except Exception as e:
     print(f"error: {e}")
  return PSMResponse (
        att=att,
        ate_raw=ate,
        num_matched_pairs=num_matched,
        message="PSM analysis complete.",
        propensity_score_plot_data=prop_hist_data,
        matched_outcome_plot_data=matched_outcome_hist_data,
        full_data_with_psm_info=[{str(k): v for k, v in row.items()} for row in df_psm.to_dict(orient='records')],
        matched_pairs_table=matched_pairs
  )