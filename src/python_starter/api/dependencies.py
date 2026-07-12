"""FastAPI dependency injection providers.

Reference: src-go/cmd/server/main.go dependency wiring pattern.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from python_starter.infrastructure.config import Settings, get_settings
from python_starter.infrastructure.database import DatabaseManager
from python_starter.infrastructure.logging import get_logger
from python_starter.infrastructure.redis_client import RedisManager

logger = get_logger(__name__)


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session from the app's DatabaseManager."""
    db_manager: DatabaseManager | None = getattr(request.app.state, "db_manager", None)
    if db_manager is None or db_manager.session_factory is None:
        raise RuntimeError("Database not available")

    async with db_manager.session() as session:
        yield session


async def get_redis(request: Request) -> RedisManager:
    """Provide the Redis manager from app state."""
    redis_manager: RedisManager | None = getattr(
        request.app.state, "redis_manager", None
    )
    if redis_manager is None:
        raise RuntimeError("Redis not available")
    return redis_manager


async def get_db_manager(request: Request) -> DatabaseManager:
    """Provide the DatabaseManager from app state."""
    db_manager: DatabaseManager | None = getattr(request.app.state, "db_manager", None)
    if db_manager is None:
        raise RuntimeError("Database manager not available")
    return db_manager


# Annotated dependency types for cleaner router signatures
DBSession = Annotated[AsyncSession, Depends(get_db)]
RedisDep = Annotated[RedisManager, Depends(get_redis)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
DBManagerDep = Annotated[DatabaseManager, Depends(get_db_manager)]
