"""FastAPI application factory with lifespan management.

Reference: src-go/cmd/server/main.go startup/shutdown logic.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from python_starter.api.routers import experiments, health, inference
from python_starter.infrastructure.config import Settings, get_settings
from python_starter.infrastructure.database import DatabaseManager
from python_starter.infrastructure.logging import configure_logging, get_logger
from python_starter.infrastructure.redis_client import RedisManager

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional Settings instance. If None, loads from environment.

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan: startup and shutdown hooks."""
        # Startup
        logger.info(
            "starting_server",
            app_name=settings.app_name,
            version=settings.app_version,
            env=settings.env,
        )

        # Connect to PostgreSQL (fail-open)
        db_manager = DatabaseManager(settings)
        db_ok = await db_manager.connect()
        if db_ok:
            await db_manager.create_tables()
        app.state.db_manager = db_manager

        # Connect to Redis (fail-open)
        redis_manager = RedisManager(settings)
        redis_ok = await redis_manager.connect()
        app.state.redis_manager = redis_manager

        if not db_ok or not redis_ok:
            logger.warning(
                "server_degraded",
                database_ok=db_ok,
                redis_ok=redis_ok,
            )

        yield

        # Shutdown
        logger.info("shutting_down_server")
        await db_manager.disconnect()
        await redis_manager.disconnect()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Python ML Starter API - Model inference and experiment management",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # Register routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(inference.router, prefix="/inference", tags=["inference"])
    app.include_router(experiments.router, prefix="/experiments", tags=["experiments"])

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error("unhandled_exception", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    return app


# Default app instance for uvicorn (python -m python_starter.api.main)
app = create_app()
