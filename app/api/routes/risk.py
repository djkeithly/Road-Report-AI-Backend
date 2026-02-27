"""Risk prediction endpoints."""

from fastapi import APIRouter, HTTPException

from app.schemas.risk import RiskRequest, RiskResponse
from app.services.risk import predict_risk

router = APIRouter(prefix="/risk", tags=["risk"])


@router.post("/predict", response_model=RiskResponse)
async def get_risk_score(request: RiskRequest) -> RiskResponse:
    """
    Get crash risk score for a location.

    Accepts coordinates (and optional context) and returns a 0–1 risk score.
    Non-blocking; delegates to dedicated service for inference.
    """
    try:
        return predict_risk(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
