"""User report submission endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.models.user_report import UserReportRecord
from app.schemas.report import (
    UserReportCreateRequest,
    UserReportCreateResponse,
    UserReportResponse,
)

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


@router.get("", response_model=list[UserReportResponse])
async def list_user_reports(
    db: DbSessionDep,
    road_name: Annotated[str | None, Query(min_length=1, max_length=255)] = None,
    limit: Annotated[int, Query(ge=1, le=25)] = 8,
) -> list[UserReportResponse]:
    """Return recent roadway issue reports, optionally filtered by road name."""
    stmt = (
        select(UserReportRecord)
        .order_by(UserReportRecord.created_at.desc())
        .limit(limit)
    )

    if road_name:
        needle = f"%{road_name.strip()}%"
        stmt = (
            select(UserReportRecord)
            .where(UserReportRecord.road_name.ilike(needle))
            .order_by(UserReportRecord.created_at.desc())
            .limit(limit)
        )

    result = await db.execute(stmt)
    records = result.scalars().all()
    return [
        UserReportResponse(
            id=record.id,
            roadName=record.road_name,
            issueType=record.issue_type,
            description=record.description,
            latitude=record.latitude,
            longitude=record.longitude,
            createdAt=record.created_at,
        )
        for record in records
    ]
