import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from api.schemas import DMLResponse, PlotData
from sklearn.preprocessing import StandardScaler

def dml(
    data: list,
    treatment_col: str,
    outcome_col: str,
    confounders: list,
    n_splits: int = 2,
    random_state: int = 42,
    scale_features: bool = True
) -> DMLResponse:
    df = pd.DataFrame(data)
    X = pd.get_dummies(df[confounders], drop_first=True)
    if scale_features:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    treatment = df[treatment_col].values
    outcomes = df[outcome_col].values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    treatment_difference = np.zeros_like(treatment, dtype=float)
    outcome_difference = np.zeros_like(outcomes, dtype=float)

    for train_idx, test_idx in kf.split(X):
        # Model for treatment
        treatment_model = LogisticRegression(max_iter=1000)
        treatment_model.fit(X.iloc[train_idx], treatment[train_idx])
        treatment_model_prediction = treatment_model.predict_proba(X.iloc[test_idx])[:, 1]

        # Model for outcome
        outcome_model = RandomForestRegressor(random_state=random_state)
        outcome_model.fit(X.iloc[train_idx], outcomes[train_idx])
        outcome_model_prediction = outcome_model.predict(X.iloc[test_idx])

        # Residuals
        treatment_difference[test_idx] = treatment[test_idx] - treatment_model_prediction
        outcome_difference[test_idx] = outcomes[test_idx] - outcome_model_prediction

    final_model = LassoCV(cv=3, random_state=random_state)
    final_model.fit(treatment_difference.reshape(-1, 1), outcome_difference)
    ate = final_model.coef_[0]
    
    mask = (treatment == 1)
    att_final_model = LassoCV(cv=3, random_state=random_state)
    att_final_model.fit(treatment_difference[mask].reshape(-1, 1), outcome_difference[mask])
    att = att_final_model.coef_[0]

    outcome_plot = PlotData(
        treated_values=outcomes[treatment == 1].tolist(),
        control_values=outcomes[treatment == 0].tolist(),
        title="Outcome Distribution by Treatment Group",
        xlabel="Outcome",
        ylabel="Count",
        legend_labels=["Treated", "Control"]
    )

    return DMLResponse(
        att=att,
        ate=ate,
        message="DML analysis complete.",
        outcome_plot=outcome_plot,
    )