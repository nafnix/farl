# from typing import overload
# from pydantic import RedisDsn
# import limits


# class Navio:
#     @overload
#     def __init__(self) -> None: ...
#     @overload
#     def __init__(self, *, redis: RedisDsn | str) -> None: ...

#     def __init__(self, **kwds) -> None:
#         redis: RedisDsn | str | None = kwds.get("redis")
#         if redis is not None:
#             storage = limits.storage.RedisStorage(str(redis))
#         else:
#             storage = limits.storage.MemoryStorage()

#         self.limiter = limits.strategies.FixedWindowRateLimiter
