"""User report submission endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.models.user_report import UserReportRecord
from app.schemas.report import UserReportCreateRequest, UserReportCreateResponse

router = APIRouter(prefix="/reports", tags=["reports"])

DbSessionDep = Annotated[AsyncSession, Depends(get_db)]


@router.post("", response_model=UserReportCreateResponse)
async def create_user_report(
    request: UserReportCreateRequest,
    db: DbSessionDep,
) -> UserReportCreateResponse:
    """Persist a user-submitted roadway issue report."""
    record = UserReportRecord(
        road_name=request.road_name.strip(),
        issue_type=request.issue_type,
        description=request.description.strip(),
        latitude=request.latitude,
        longitude=request.longitude,
    )
    db.add(record)
    await db.flush()
    return UserReportCreateResponse(id=record.id, createdAt=record.created_at)
