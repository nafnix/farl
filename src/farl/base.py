from typing import Literal, cast, overload

import limits
import limits.aio
from pydantic import RedisDsn

from farl.types import AsyncFarlProtocol, FarlProtocol


try:
    from redis import ConnectionPool as RedisConnectionPool
    from redis.asyncio import ConnectionPool as AsyncRedisConnectionPool
except ImportError:
    RedisConnectionPool = None
    AsyncRedisConnectionPool = None


class Farl(FarlProtocol):
    @overload
    def __init__(
        self,
        *,
        redis_uri: str | RedisDsn | None,
        redis_connection_pool: "RedisConnectionPool | None" = None,  # pyright: ignore [reportInvalidTypeForm]
        redis_key_prefix: str = "farl",
        redis_wrap_exceptions: bool = False,
        redis_options: dict | None = None,
        strategy: Literal[
            "fixed-window",
            "moving-window",
            "sliding-window-counter",
        ] = "fixed-window",
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        storage_uri: str | RedisDsn,
        storage_options: dict | None = None,
        strategy: Literal[
            "fixed-window",
            "moving-window",
            "sliding-window-counter",
        ] = "fixed-window",
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        strategy: Literal[
            "fixed-window",
            "moving-window",
            "sliding-window-counter",
        ] = "fixed-window",
    ) -> None: ...

    def __init__(
        self,
        *,
        redis_uri: str | RedisDsn | None = None,
        redis_connection_pool: "RedisConnectionPool | None" = None,  # pyright: ignore [reportInvalidTypeForm]
        redis_key_prefix: str = "farl",
        redis_wrap_exceptions: bool = False,
        redis_options: dict | None = None,
        storage_uri: str | RedisDsn | None = None,
        storage_options: dict | None = None,
        strategy: Literal[
            "fixed-window",
            "moving-window",
            "sliding-window-counter",
        ] = "fixed-window",
    ) -> None:
        storage_options = {**(storage_options or {})}
        if redis_uri:
            storage_options.update(
                {
                    "connection_pool": redis_connection_pool,
                    "key_prefix": redis_key_prefix,
                    "wrap_exceptions": redis_wrap_exceptions,
                }
            )
            if redis_options:
                storage_options.update(redis_options)

            if not isinstance(redis_uri, str):
                redis_uri = redis_uri.encoded_string()

            storage_uri = redis_uri

        if storage_uri:
            if not isinstance(storage_uri, str):
                storage_uri = storage_uri.encoded_string()

            storage = limits.storage.storage_from_string(storage_uri, **storage_options)
        else:
            storage = limits.storage.MemoryStorage()

        self.limiter: limits.strategies.RateLimiter
        if strategy == "fixed-window":
            self.limiter = limits.strategies.FixedWindowRateLimiter(storage)
        elif strategy == "moving-window":
            self.limiter = limits.strategies.MovingWindowRateLimiter(storage)
        elif strategy == "sliding-window-counter":
            self.limiter = limits.strategies.SlidingWindowCounterRateLimiter(storage)
        else:
            raise ValueError(
                f"Unsupported strategy: {strategy}. "
                "Available options are: 'fixed-window', 'moving-window', "
                "'sliding-window-counter'."
            )


class AsyncFarl(AsyncFarlProtocol):
    @overload
    def __init__(
        self,
        *,
        redis_uri: str | RedisDsn | None,
        redis_connection_pool: "AsyncRedisConnectionPool | None" = None,  # pyright: ignore [reportInvalidTypeForm]
        redis_key_prefix: str = limits.aio.storage.RedisStorage.PREFIX,
        redis_wrap_exceptions: bool = False,
        redis_options: dict | None = None,
        strategy: Literal[
            "fixed-window",
            "moving-window",
            "sliding-window-counter",
        ] = "fixed-window",
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        storage_uri: str | RedisDsn,
        storage_options: dict | None = None,
        strategy: Literal[
            "fixed-window",
            "moving-window",
            "sliding-window-counter",
        ] = "fixed-window",
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        strategy: Literal[
            "fixed-window",
            "moving-window",
            "sliding-window-counter",
        ] = "fixed-window",
    ) -> None: ...

    def __init__(
        self,
        *,
        redis_uri: str | RedisDsn | None = None,
        redis_connection_pool: "AsyncRedisConnectionPool | None" = None,  # pyright: ignore [reportInvalidTypeForm]
        redis_key_prefix: str = limits.aio.storage.RedisStorage.PREFIX,
        redis_wrap_exceptions: bool = False,
        redis_options: dict | None = None,
        storage_uri: str | RedisDsn | None = None,
        storage_options: dict | None = None,
        strategy: Literal[
            "fixed-window",
            "moving-window",
            "sliding-window-counter",
        ] = "fixed-window",
    ) -> None:
        storage_options = {**(storage_options or {})}
        if redis_uri:
            storage_options.update(
                {
                    "connection_pool": redis_connection_pool,
                    "key_prefix": redis_key_prefix,
                    "wrap_exceptions": redis_wrap_exceptions,
                }
            )
            if redis_options:
                storage_options.update(redis_options)

            if not isinstance(redis_uri, str):
                storage_uri = redis_uri.encoded_string()
            else:
                storage_uri = redis_uri

        if storage_uri:
            if not isinstance(storage_uri, str):
                storage_uri = storage_uri.encoded_string()
            async_storage_uri = self._to_async_storage_uri(storage_uri)

            storage = limits.storage.storage_from_string(
                async_storage_uri,
                **storage_options,
            )
        else:
            storage = limits.aio.storage.MemoryStorage()

        limiter_class: type[limits.aio.strategies.RateLimiter]
        if strategy == "fixed-window":
            limiter_class = limits.aio.strategies.FixedWindowRateLimiter
        elif strategy == "moving-window":
            limiter_class = limits.aio.strategies.MovingWindowRateLimiter
        elif strategy == "sliding-window-counter":
            limiter_class = limits.aio.strategies.SlidingWindowCounterRateLimiter

        else:
            raise ValueError(
                f"Unsupported strategy: {strategy}. "
                "Available options are: 'fixed-window', 'moving-window', "
                "'sliding-window-counter'."
            )

        self.limiter = limiter_class(cast(limits.aio.storage.Storage, storage))

    @staticmethod
    def _to_async_storage_uri(uri: str):
        if uri.startswith("async+"):
            return uri
        return f"async+{uri}"
