"""Application configuration from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings loaded from .env."""

    # Database (use DATABASE_URL in .env for PostgreSQL)
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


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
