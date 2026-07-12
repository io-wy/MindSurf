"""Async PostgreSQL database setup with SQLAlchemy 2.0.

Reference: src-go/pkg/database/postgres.go (fail-open pattern)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from python_starter.infrastructure.config import Settings
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)
Base = declarative_base()


class DatabaseManager:
    """Manages async PostgreSQL connection lifecycle."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker[AsyncSession] | None = None

    async def connect(self) -> bool:
        """Initialize database engine and session factory.

        Returns True on success, False on failure (fail-open).
        """
        try:
            self.engine = create_async_engine(
                self.settings.database_url,
                echo=self.settings.debug,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
            )
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
            logger.info("database_connected", url=self.settings.database_url)
            return True
        except Exception as e:
            logger.warning(
                "database_unavailable",
                error=str(e),
                url=self.settings.database_url,
            )
            return False

    async def disconnect(self) -> None:
        """Close database connections gracefully."""
        if self.engine:
            await self.engine.dispose()
            logger.info("database_disconnected")

    async def create_tables(self) -> None:
        """Create all tables (for development/testing only)."""
        if self.engine:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("database_tables_created")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide a transactional scope around a series of operations."""
        if self.session_factory is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        async with self.session_factory() as db:
            try:
                yield db
                await db.commit()
            except Exception:
                await db.rollback()
                raise
