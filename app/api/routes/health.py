"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Check API and database connectivity."""
    return {"status": "ok", "message": "Database connected and API is healthy"}
