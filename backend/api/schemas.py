from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

class PSMRequest(BaseModel):
    data: List[Dict[str, Any]]
    treatment_col: str
    outcome_col: str
    confounders: List[str]
    n_neighbors: int = Field(default=1, ge=1)
    random_state: int = 42
    scale_features: bool = True
    use_caliper: bool = False
    caliper: Optional[float] = Field(default=0.1, ge=0.0)
    show_prop_hist: bool = True
    show_matched_pair_hist: bool = False

class PlotData(BaseModel):
    treated_values: Optional[List[float]] = None
    control_values: Optional[List[float]] = None
    title: str
    xlabel: str
    ylabel: str
    legend_labels: Optional[List[str]] = None

class MatchedPairInfo(BaseModel):
    treated_index: Union[int, str] 
    control_index: Union[int, str] 
    distance: float

class PSMResponse(BaseModel):
    att: Optional[float] = None
    ate_raw: Optional[float] = None 
    num_matched_pairs: Optional[int] = None
    message: Optional[str] = None
    propensity_score_plot_data: Optional[PlotData] = None
    matched_outcome_plot_data: Optional[PlotData] = None
    full_data_with_psm_info: Optional[List[Dict[str, Any]]] = None
    matched_pairs_table: Optional[List[MatchedPairInfo]] = None

class DMLRequest(BaseModel):
    data: List[Dict[str, Any]]
    treatment_col: str
    outcome_col: str
    confounders: List[str]
    n_splits: int = Field(default=5, ge=2)
    random_state: int = 42
    scale_features: bool = True
    show_outcome_hist: bool = True

class DMLResponse(BaseModel):
    ate: Optional[float] = None
    att: Optional[float] = None
    message: Optional[str] = None
    outcome_plot: Optional[PlotData] = None
    linear_regression_plot: Optional[dict] = None