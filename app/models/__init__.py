"""SQLAlchemy models for persistent backend entities."""

from app.models.prediction import PredictionRecord
from app.models.user_report import UserReportRecord

__all__ = ["PredictionRecord", "UserReportRecord"]
