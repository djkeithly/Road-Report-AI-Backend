"""Database configuration and async session management."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.config import get_settings

settings = get_settings()
engine = create_async_engine(
    settings.async_database_url,
    echo=settings.debug,
)
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async database session via FastAPI dependency injection."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
