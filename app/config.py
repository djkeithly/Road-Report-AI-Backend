"""Application configuration from environment variables."""

from functools import lru_cache
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings loaded from .env."""

    # Database (use DATABASE_URL in .env for PostgreSQL)
    # For async: postgresql+asyncpg:// or sqlite+aiosqlite://
    database_url: str = "sqlite:///./test.db"

    # API
    api_v1_prefix: str = "/api/v1"
    debug: bool = False
    cors_origins_csv: str = "http://localhost:5173,http://localhost:3000"

    # External APIs (add as needed)
    google_maps_api_key: str | None = None
    weather_api_base_url: str = "https://api.weather.gov"
    weather_user_agent: str = "(roadreportai.local, dev@roadreportai.local)"
    weather_timeout_seconds: float = 6.0

    # ML scoring runtime configuration
    model_file_path: str = "ml/artifacts/latest-model.pt"
    model_version: str = "baseline-v0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
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

    @property
    def cors_origins(self) -> list[str]:
        """Return normalized CORS origins from comma-separated configuration."""
        origins = [value.strip().rstrip("/") for value in self.cors_origins_csv.split(",")]
        return [origin for origin in origins if origin]

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value: Any) -> Any:
        """Allow empty env values for DEBUG by falling back to default."""
        if value == "":
            return False
        return value

    @field_validator("database_url", mode="before")
    @classmethod
    def parse_database_url(cls, value: Any) -> Any:
        """Allow empty env values for DATABASE_URL by falling back to SQLite."""
        if value == "":
            return "sqlite:///./test.db"
        return value

    @field_validator("api_v1_prefix", mode="before")
    @classmethod
    def parse_api_v1_prefix(cls, value: Any) -> Any:
        """Allow empty env values for API_V1_PREFIX by falling back to default."""
        if value == "":
            return "/api/v1"
        return value

    @field_validator("cors_origins_csv", mode="before")
    @classmethod
    def parse_cors_origins_csv(cls, value: Any) -> Any:
        """Allow empty env values for CORS origins by falling back to localhost."""
        if value == "":
            return "http://localhost:5173,http://localhost:3000"
        return value

    @field_validator("weather_timeout_seconds", mode="before")
    @classmethod
    def parse_weather_timeout_seconds(cls, value: Any) -> Any:
        """Allow empty env values for weather timeout by falling back to default."""
        if value == "":
            return 6.0
        return value


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
