"""Infrastructure lifecycle and tokenizer wrapper tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from python_starter.api.dependencies import (
    get_db,
    get_db_manager,
    get_redis,
)
from python_starter.core.tokenizer import decode_tokens, encode_text, load_tokenizer
from python_starter.infrastructure.config import Settings
from python_starter.infrastructure.database import DatabaseManager
from python_starter.infrastructure.redis_client import RedisManager


class _AsyncContext:
    def __init__(self, value: Any) -> None:
        self.value = value

    async def __aenter__(self) -> Any:
        return self.value

    async def __aexit__(self, *args: Any) -> None:
        return None


class _Connection:
    def __init__(self) -> None:
        self.called = False

    async def run_sync(self, function: Any) -> None:
        self.called = True


class _Engine:
    def __init__(self) -> None:
        self.connection = _Connection()
        self.disposed = False

    def begin(self) -> _AsyncContext:
        return _AsyncContext(self.connection)

    async def dispose(self) -> None:
        self.disposed = True


class _Session:
    def __init__(self, fail_commit: bool = False) -> None:
        self.fail_commit = fail_commit
        self.committed = False
        self.rolled_back = False

    async def commit(self) -> None:
        if self.fail_commit:
            raise RuntimeError("commit failed")
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


class _Tokenizer:
    pad_token = None
    pad_token_id = None
    eos_token = "<eos>"
    eos_token_id = 1

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return [1, 2]

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        return "decoded"


def test_tokenizer_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _Tokenizer()
    monkeypatch.setattr(
        "python_starter.core.tokenizer.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    loaded = load_tokenizer("local")
    assert loaded.pad_token == "<eos>"
    assert loaded.pad_token_id == 1
    assert encode_text(loaded, "text", max_length=2) == [1, 2]
    assert decode_tokens(loaded, [1, 2]) == "decoded"


@pytest.mark.asyncio
async def test_database_manager_success_failure_and_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import python_starter.infrastructure.database as database

    settings = Settings(postgres_url="postgresql+asyncpg://test")
    engine = _Engine()
    session = _Session()
    monkeypatch.setattr(database, "create_async_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(
        database,
        "async_sessionmaker",
        lambda *_args, **_kwargs: lambda: _AsyncContext(session),
    )
    manager = DatabaseManager(settings)
    assert await manager.connect() is True
    await manager.create_tables()
    assert engine.connection.called is True
    async with manager.session() as value:
        assert value is cast(Any, session)
    assert session.committed is True
    await manager.disconnect()
    assert engine.disposed is True

    monkeypatch.setattr(
        database,
        "create_async_engine",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    assert await DatabaseManager(settings).connect() is False

    disconnected = DatabaseManager(settings)
    with pytest.raises(RuntimeError, match="not connected"):
        async with disconnected.session():
            pass


@pytest.mark.asyncio
async def test_redis_manager_and_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Redis:
        def __init__(self) -> None:
            self.closed = False

        async def ping(self) -> bool:
            return True

        async def close(self) -> None:
            self.closed = True

    redis = _Redis()
    monkeypatch.setattr(
        "python_starter.infrastructure.redis_client.aioredis.from_url",
        lambda *_args, **_kwargs: redis,
    )
    manager = RedisManager(Settings(redis_url="redis://test"))
    assert await manager.connect() is True
    assert manager.get_client() is cast(Any, redis)
    await manager.disconnect()
    assert redis.closed is True

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                redis_manager=manager,
                db_manager=SimpleNamespace(session_factory=lambda: None),
            )
        )
    )
    assert await get_redis(request) is manager  # type: ignore[arg-type]
    assert await get_db_manager(request) is request.app.state.db_manager  # type: ignore[arg-type]

    missing = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="Redis"):
        await get_redis(missing)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="Database manager"):
        await get_db_manager(missing)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="Database not available"):
        async for _ in get_db(missing):  # type: ignore[arg-type]
            pass

    monkeypatch.setattr(
        "python_starter.infrastructure.redis_client.aioredis.from_url",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    assert await RedisManager(Settings(redis_url="redis://offline")).connect() is False
