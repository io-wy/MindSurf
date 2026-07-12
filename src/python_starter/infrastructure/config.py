"""Application configuration loaded from environment variables and .env files.

Reference: src-go/internal/config/config.go
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with defaults, loaded from env vars and .env files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    env: str = Field(default="development", alias="ENV")
    debug: bool = Field(default=False, alias="DEBUG")
    app_name: str = Field(default="python-starter", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")

    # API Server
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Database
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="dev", alias="POSTGRES_USER")
    postgres_password: str = Field(default="dev", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="appdb", alias="POSTGRES_DB")
    postgres_url: str | None = Field(default=None, alias="POSTGRES_URL")

    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # Celery
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1", alias="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/2", alias="CELERY_RESULT_BACKEND"
    )

    # Experiment Tracking
    wandb_project: str = Field(default="python-starter", alias="WANDB_PROJECT")
    wandb_api_key: str | None = Field(default=None, alias="WANDB_API_KEY")
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="default", alias="MLFLOW_EXPERIMENT_NAME"
    )

    # Training
    cuda_visible_devices: str = Field(default="0", alias="CUDA_VISIBLE_DEVICES")
    default_device: str = Field(default="auto", alias="DEFAULT_DEVICE")

    # Security
    secret_key: str = Field(
        default="dev-secret-change-me-in-production-32ch", alias="SECRET_KEY"
    )

    @property
    def database_url(self) -> str:
        """Return the async PostgreSQL URL.

        If POSTGRES_URL is explicitly set, use it. Otherwise construct from parts.
        """
        if self.postgres_url:
            return self.postgres_url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.env.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance.

    LRU cache ensures the config is loaded only once per process,
    similar to viper singleton pattern in Go.
    """
    return Settings()
