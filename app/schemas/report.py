"""Schemas for user-submitted roadway reports."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ReportIssueType = Literal[
    "pothole",
    "flooding",
    "debris",
    "construction",
    "signal_outage",
    "other",
]


class UserReportCreateRequest(BaseModel):
    """Request body for creating a roadway issue report."""

    road_name: str = Field(min_length=1, max_length=255)
    issue_type: ReportIssueType
    description: str = Field(min_length=4, max_length=2000)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)


class UserReportCreateResponse(BaseModel):
    """Response payload after creating a roadway issue report."""

    id: int
    message: str = "Report saved"
    created_at: datetime = Field(alias="createdAt")
    model_config = ConfigDict(populate_by_name=True)


class UserReportResponse(BaseModel):
    """Stored roadway issue report returned to frontend consumers."""

    id: int
    road_name: str = Field(alias="roadName")
    issue_type: ReportIssueType = Field(alias="issueType")
    description: str
    latitude: float | None = None
    longitude: float | None = None
    created_at: datetime = Field(alias="createdAt")
    model_config = ConfigDict(populate_by_name=True)
