from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # To allow requests from Streamlit
from .schemas import PSMRequest, PSMResponse
from ..main import app
from core.psm import psm
from core.dml import dml


@app.post("/psm", response_model = PSMResponse)
async def psm_endpoint(request: PSMRequest):
  result = psm(
    data = request.data
    treatment_col = request.treatment_col,
    outcome_col = request.outcome_col,
    counfounders = request.counfounders,
    n_neighbors = request.n_neighbors,
    random_state = request.random_state,
    scale_features = request.scale_features,
    use_caliper = request.use_caliper,
    caliper = request.caliper,
    show_prop_hist = request.show_prop_hist
    show_matched_pair_hist = request.show_matched_pair_hist
  )

  return result