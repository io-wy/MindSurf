"""Health check endpoints.

Reference: src-go/internal/handler/health.go
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request, status

from python_starter.api.schemas.models import HealthResponse, HealthStatus
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Basic liveness probe. Always returns 200 if the process is running."""
    settings = request.app.state.settings if hasattr(request.app.state, "settings") else None
    version = getattr(settings, "app_version", "0.1.0") if settings else "0.1.0"

    return HealthResponse(
        status=HealthStatus.OK,
        version=version,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness_check(request: Request) -> HealthResponse:
    """Readiness probe - checks if dependencies are available."""
    db_manager = getattr(request.app.state, "db_manager", None)
    redis_manager = getattr(request.app.state, "redis_manager", None)

    db_ok = db_manager is not None and db_manager.engine is not None
    redis_ok = redis_manager is not None and redis_manager.client is not None

    services = {
        "database": db_ok,
        "redis": redis_ok,
    }

    all_ok = all(services.values())
    status = HealthStatus.OK if all_ok else HealthStatus.DEGRADED

    settings = getattr(request.app.state, "settings", None)
    version = getattr(settings, "app_version", "0.1.0") if settings else "0.1.0"

    return HealthResponse(
        status=status,
        version=version,
        timestamp=datetime.now(timezone.utc),
        services=services,
    )
