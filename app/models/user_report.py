"""User-submitted roadway report persistence model."""

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class UserReportRecord(Base):
    """Stores user-submitted roadway issue reports."""

    __tablename__ = "user_reports"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    road_name: Mapped[str] = mapped_column(String(255))
    issue_type: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(Text)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
    )
