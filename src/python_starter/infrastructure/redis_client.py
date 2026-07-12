"""Redis client setup with fail-open pattern.

Reference: src-go/pkg/database/redis.go
"""

from __future__ import annotations

import redis.asyncio as aioredis

from python_starter.infrastructure.config import Settings
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)


class RedisManager:
    """Manages Redis connection lifecycle."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: aioredis.Redis | None = None

    async def connect(self) -> bool:
        """Initialize Redis client.

        Returns True on success, False on failure (fail-open).
        """
        try:
            self.client = aioredis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
            await self.client.ping()
            logger.info("redis_connected", url=self.settings.redis_url)
            return True
        except Exception as e:
            logger.warning("redis_unavailable", error=str(e), url=self.settings.redis_url)
            return False

    async def disconnect(self) -> None:
        """Close Redis connection gracefully."""
        if self.client:
            await self.client.close()
            logger.info("redis_disconnected")

    def get_client(self) -> aioredis.Redis | None:
        """Return the Redis client instance, or None if unavailable."""
        return self.client
