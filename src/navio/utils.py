import base64
from collections.abc import Sequence
from datetime import datetime
from math import ceil
from typing import TYPE_CHECKING, Literal, NamedTuple, NotRequired, TypedDict

from limits.limits import (
    RateLimitItem,
    RateLimitItemPerDay,
    RateLimitItemPerHour,
    RateLimitItemPerMinute,
    RateLimitItemPerMonth,
    RateLimitItemPerSecond,
    RateLimitItemPerYear,
)
from limits.util import parse_many

from navio.exceptions import NavioError


if TYPE_CHECKING:
    from navio.types import AnyNavioProtocol, RateLimitArgument


_TimeUnit = Literal[
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


class RateLimitDictValue(TypedDict):
    time: NotRequired[_TimeUnit]
    amount: int
    multiples: NotRequired[int]


class RateLimitTimeArg:
    __slots__ = ("time", "val")

    def __init__(
        self,
        *,
        time: _TimeUnit = "m",
        amount: int,
        multiples: int = 1,
    ) -> None:
        if time == "Y":
            time = "year"
        elif time == "M":
            time = "month"
        elif time == "D":
            time = "day"
        elif time == "h":
            time = "hour"
        elif time == "m":
            time = "minute"
        elif time == "s":
            time = "second"

        if time == "year":
            self.val = RateLimitItemPerYear(amount, multiples)
        elif time == "month":
            self.val = RateLimitItemPerMonth(amount, multiples)
        elif time == "day":
            self.val = RateLimitItemPerDay(amount, multiples)
        elif time == "hour":
            self.val = RateLimitItemPerHour(amount, multiples)
        elif time == "minute":
            self.val = RateLimitItemPerMinute(amount, multiples)
        elif time == "second":
            self.val = RateLimitItemPerSecond(amount, multiples)

        self.time = time


def parse_rate_limit_value(
    *args: "RateLimitArgument",
):
    result: list[RateLimitItem] = []
    for i in args:
        if isinstance(i, str):
            result.extend(parse_many(i))
        elif isinstance(i, RateLimitItem):
            result.append(i)
        elif isinstance(i, RateLimitTimeArg):
            result.append(i.val)
        elif isinstance(i, dict):
            result.append(
                RateLimitTimeArg(
                    time=i.get("time", "minute"),
                    amount=i["amount"],
                    multiples=i.get("multiples", 1),
                ).val
            )
        elif isinstance(i, Sequence):
            result.extend(parse_rate_limit_value(*i))
        else:
            raise TypeError(
                f"Unsupported rate limit argument type: {type(i).__name__}."
            )

    return result


class HeaderRateLimit(NamedTuple):
    policy: str
    remaining: int
    reset_timestamp: float | None = None
    partition_key: str | bytes | None = None
    error_class: type[NavioError] | None = None

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


class NavioState(TypedDict):
    navio: NotRequired["AnyNavioProtocol"]
    policy: list[HeaderRateLimitPolicy]
    state: list[HeaderRateLimit]
    violated: list[HeaderRateLimit]
