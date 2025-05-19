from fastapi import FastAPI, HTTPException, APIRouter
from .schemas import PSMRequest, PSMResponse
from ..core.psm import psm
# from core.dml import dml
router = APIRouter()

@router.post("/psm", response_model = PSMResponse)
async def psm_endpoint(request: PSMRequest):
  result = psm(
      data=request.data,
      treatment_col=request.treatment_col,
      outcome_col=request.outcome_col,
      confounders=request.confounders, 
      n_neighbors=request.n_neighbors,
      random_state=request.random_state,
      scale_features=request.scale_features,
      use_caliper=request.use_caliper,
      caliper=request.caliper,
      show_prop_hist=request.show_prop_hist,
      show_matched_pair_hist=request.show_matched_pair_hist
  )

  return result

@router.post("/dml", response_model=DMLResponse)
async def dml_endpoint(request: DMLRequest):
    result = dml(
        data=request.data,
        treatment_col=request.treatment_col,
        outcome_col=request.outcome_col,
        confounders=request.confounders,
        n_splits=request.n_splits,
        random_state=request.random_state
    )

    return result