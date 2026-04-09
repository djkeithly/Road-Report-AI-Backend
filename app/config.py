"""Application configuration from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings loaded from .env."""

    # Database (use DATABASE_URL in .env for PostgreSQL)
    # For async: postgresql+asyncpg:// or sqlite+aiosqlite://
    database_url: str = "sqlite:///./test.db"

    # API
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    # External APIs (add as needed)
    google_maps_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def async_database_url(self) -> str:
        """Return database URL with async driver for SQLAlchemy."""
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if self.database_url.startswith("postgresql+asyncpg://"):
            return self.database_url
        if self.database_url.startswith("sqlite"):
            return self.database_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
