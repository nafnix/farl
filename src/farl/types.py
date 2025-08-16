import base64
from collections.abc import Awaitable, Callable
from datetime import datetime
from math import ceil
from typing import Literal, NamedTuple, NotRequired, Protocol, TypeVar, TypedDict

import limits
import limits.aio
from pydantic import networks


class HeaderRateLimit(NamedTuple):
    policy: str
    remaining: int
    reset_timestamp: float | None = None
    partition_key: str | bytes | None = None

    @property
    def quota_reset_seconds(self) -> int | None:
        if self.reset_timestamp is not None:
            return ceil(self.reset_timestamp - datetime.now().timestamp())
        return None

    def __str__(self) -> str:
        values = [f'"{self.policy}"', f"r={self.remaining}"]

        if (t := self.quota_reset_seconds) is not None:
            values.append(f"t={t}")

        if self.partition_key is not None:
            if isinstance(self.partition_key, bytes):
                pk = f":{base64.b64encode(self.partition_key).decode()}:"
            else:
                pk = f'"{self.partition_key}"'

            values.append(f"pk={pk}")

        return ";".join(values)


class HeaderRateLimitPolicy(NamedTuple):
    policy: str

    quota: int
    quota_unit: str | None = None
    window: int | None = None
    partition_key: str | bytes | None = None

    def __str__(self) -> str:
        values = [f'"{self.policy}"', f"q={self.quota}"]
        if self.quota_unit is not None:
            values.append(f'qu="{self.quota_unit}"')

        if self.window is not None:
            values.append(f"w={self.window}")

        if self.partition_key is not None:
            if isinstance(self.partition_key, bytes):
                pk = f":{base64.b64encode(self.partition_key).decode()}:"
            else:
                pk = f'"{self.partition_key}"'

            values.append(f"pk={pk}")

        return ";".join(values)


_T = TypeVar("_T")

_ResultT = _T | Awaitable[_T]

Quota = int
GetQuotaDependency = Callable[..., _ResultT[Quota]]

TimeType = Literal[
    "Y",
    "year",
    "M",
    "month",
    "D",
    "day",
    "h",
    "hour",
    "m",
    "minute",
    "s",
    "second",
]
GetTimeTypeDependency = Callable[..., _ResultT[TimeType]]


Period = int
GetPeriodDependency = Callable[..., _ResultT[Period]]


Key = str
GetKeyDependency = Callable[..., _ResultT[Key]]


Cost = int
GetCostDependency = Callable[..., _ResultT[Cost]]

PartitionCostMapping = dict[Key, Cost]
GetPartitionCostMappingDependency = Callable[..., _ResultT[PartitionCostMapping]]


PolicyName = str
GetPolicyNameDependency = Callable[..., _ResultT[PolicyName | None]]

QuotaUnit = str
GetQuotaUnitDependency = Callable[..., _ResultT[QuotaUnit | None]]


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


class _FarlProtocol(Protocol[_T]):
    limiter: _T


FarlProtocol = _FarlProtocol[limits.strategies.RateLimiter]
AsyncFarlProtocol = _FarlProtocol[limits.aio.strategies.RateLimiter]
AnyFarlProtocol = FarlProtocol | AsyncFarlProtocol


class FarlState(TypedDict):
    farl: NotRequired[AnyFarlProtocol]
    policy: list[HeaderRateLimitPolicy]
    state: list[HeaderRateLimit]
    violated: list[HeaderRateLimit]
