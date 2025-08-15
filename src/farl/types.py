from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TypeVar

import limits
import limits.aio
from fastapi import Request
from limits import RateLimitItem
from pydantic import networks

from farl.utils import RateLimitDictValue, RateLimitTimeArg


Key = str
KeyResult = Key | Sequence[Key]
GetKeyDependency = Callable[..., KeyResult | Awaitable[KeyResult]]
GetRequestKey = Callable[[Request], KeyResult | Awaitable[KeyResult]]


Cost = int
CostResult = Cost | Sequence[Cost]
GetCostDependency = Callable[..., CostResult | Awaitable[CostResult]]
GetRequestCost = Callable[[Request], CostResult | Awaitable[CostResult]]


RateLimitPolicy = (
    str
    | RateLimitItem
    | Sequence[RateLimitItem]
    | RateLimitTimeArg
    | Sequence[RateLimitTimeArg]
    | RateLimitDictValue
    | Sequence[RateLimitDictValue]
)
GetRateLimitPolicyDependency = Callable[
    ...,
    RateLimitPolicy | Awaitable[RateLimitPolicy],
]
GetRequestRateLimitPolicy = Callable[
    [Request],
    RateLimitPolicy | Awaitable[RateLimitPolicy],
]


class RedisDsn(networks.RedisDsn):
    _constraints = networks.UrlConstraints(
        allowed_schemes=[
            "redis",
            "rediss",
            "redis+sentinel",
            "redis+cluster",
        ],
        default_host="localhost",
        default_port=6379,
        default_path="/0",
        host_required=True,
    )


_T = TypeVar("_T")


class _FarlProtocol(Protocol[_T]):
    limiter: _T
    namespace: str | None
    key: KeyResult | GetRequestKey | None
    cost: CostResult | GetRequestCost | None
    policy: RateLimitPolicy | GetRequestRateLimitPolicy | None


FarlProtocol = _FarlProtocol[limits.strategies.RateLimiter]
AsyncFarlProtocol = _FarlProtocol[limits.aio.strategies.RateLimiter]
AnyFarlProtocol = FarlProtocol | AsyncFarlProtocol
