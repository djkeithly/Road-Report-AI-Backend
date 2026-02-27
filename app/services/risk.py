"""Risk prediction service - placeholder for AI model integration."""

from app.schemas.risk import RiskRequest, RiskResponse


def predict_risk(request: RiskRequest) -> RiskResponse:
    """
    Predict crash risk for a location.

    Args:
        request: RiskRequest with latitude, longitude, and optional context.

    Returns:
        RiskResponse with crash likelihood score (0–1), coordinates, and message.

    TODO: Replace with actual PyTorch model inference.
    For now returns a placeholder score based on coordinates.
    """
    # Placeholder: simple hash-based "score" for development
    # In Phase 2–3 you'll wire this to your trained model
    placeholder_score = 0.5  # Replace with model.predict(...)

    return RiskResponse(
        risk_score=placeholder_score,
        latitude=request.latitude,
        longitude=request.longitude,
        message="Risk score generated (placeholder - connect AI model)",
    )
