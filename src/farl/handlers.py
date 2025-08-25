import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

import anyio
from fastapi import Request
from limits.limits import (
    RateLimitItem,
    RateLimitItemPerDay,
    RateLimitItemPerHour,
    RateLimitItemPerMinute,
    RateLimitItemPerMonth,
    RateLimitItemPerSecond,
    RateLimitItemPerYear,
)

from farl.constants import DEFAULT_COST, DEFAULT_NAMESPACE, DEFAULT_PERIOD
from farl.exceptions import FarlError
from farl.types import (
    AnyFarlProtocol,
    Cost,
    FarlState,
    HeaderRateLimit,
    HeaderRateLimitPolicy,
    Key,
    PolicySettings,
    Quota,
    TimeType,
)


_T = TypeVar("_T")


logger = logging.getLogger("farl")


class AbstractPolicyHandler(ABC, Generic[_T]):
    def __init__(
        self,
        request: Request,
        name: str | None,
        quota: Quota,
        quota_unit: str | None,
        time: TimeType,
        period: int,
        cost: Cost,
        partition: Key | None,
        namespace: Key,
        state: FarlState,
        farl: AnyFarlProtocol,
        error_class: type[FarlError] | None,
    ):
        self.request = request
        self.quota = quota
        self.time: TimeType = time
        self.period = period
        self.name = name
        self.quota_unit = quota_unit
        self.namespace = namespace
        self.partition = partition
        self.cost = cost
        self.state = state
        self.farl = farl
        self.error_class = error_class

    def get_limit_value(self) -> RateLimitItem:
        time = self.time
        amount = self.quota
        multiples = self.period
        namespace = self.namespace

        if time in {"year", "Y"}:
            result = RateLimitItemPerYear(amount, multiples, namespace=namespace)
        elif time in {"month", "M"}:
            result = RateLimitItemPerMonth(amount, multiples, namespace=namespace)
        elif time in {"day", "D"}:
            result = RateLimitItemPerDay(amount, multiples, namespace=namespace)
        elif time in {"hour", "h"}:
            result = RateLimitItemPerHour(amount, multiples, namespace=namespace)
        elif time in {"minute", "m"}:
            result = RateLimitItemPerMinute(amount, multiples, namespace=namespace)
        elif time in {"second", "s"}:
            result = RateLimitItemPerSecond(amount, multiples, namespace=namespace)
        else:
            raise ValueError(f"Unsupported time type: {time}")

        return result

    @abstractmethod
    def __call__(self) -> _T: ...


class PolicyHandler(AbstractPolicyHandler[bool]):
    @staticmethod
    def _get_policy_name(value: RateLimitItem):
        items = ["pre", value.GRANULARITY.name]
        if value.multiples != 1:
            items.insert(1, str(value.multiples))

        return "".join(items)

    async def __call__(self):
        value = self.get_limit_value()
        name = self.name if self.name is not None else self._get_policy_name(value)
        limiter = self.farl.limiter
        self.state["policy"].append(
            HeaderRateLimitPolicy(
                name,
                self.quota,
                self.quota_unit,
                self.period * value.GRANULARITY.seconds,
                value.namespace,
            )
        )
        keys = []
        if self.partition is not None:
            keys.append(self.partition)

        hit_result = limiter.hit(value, *keys, cost=self.cost)
        if inspect.isawaitable(hit_result):
            hit_result = await hit_result

        stats_result = limiter.get_window_stats(value, *keys)
        if inspect.isawaitable(stats_result):
            stats_result = await stats_result

        ratelimit = HeaderRateLimit(
            name,
            stats_result.remaining,
            stats_result.reset_time,
            self.partition,
        )
        self.state["state"].append(ratelimit)

        if hit_result is False:
            self.state["violated"].append(ratelimit)

            if self.error_class is not None:
                logger.warning(
                    ("Rate limit exceeded - policy: %s"),
                    name,
                    extra={"violated": self.state["violated"]},
                )
                raise self.error_class(
                    violated_policies=[i.policy for i in self.state["violated"]]
                )
        return hit_result


class AbstractPolicySettingsHandler(ABC, Generic[_T]):
    def __init__(
        self,
        request: Request,
        settings: PolicySettings | Sequence[PolicySettings],
        state: FarlState,
        farl: AnyFarlProtocol,
        error_class: type[FarlError] | None,
    ):
        self.request = request
        self.settings = settings
        self.state = state
        self.farl = farl
        self.error_class = error_class

    @abstractmethod
    def __call__(self) -> _T: ...


class PolicySettingsHandler(AbstractPolicySettingsHandler[bool]):
    async def __call__(self) -> bool:
        items = (
            self.settings
            if isinstance(
                self.settings,
                Sequence,
            )
            else [self.settings]
        )

        handles = [
            PolicyHandler(
                self.request,
                name=settings.get("name"),
                quota=settings.get("quota"),
                quota_unit=settings.get("quota_unit"),
                time=settings.get("time", "minute"),
                period=settings.get("period", DEFAULT_PERIOD),
                cost=settings.get("cost", DEFAULT_COST),
                partition=settings.get("partition"),
                state=self.state,
                namespace=settings.get("namespace", DEFAULT_NAMESPACE),
                farl=self.farl,
                error_class=None,
            )
            for settings in items
        ]

        async with anyio.create_task_group() as tg:
            for h in handles:
                tg.start_soon(h)

        if self.state["violated"] and self.error_class is not None:
            logger.warning(
                "Rate limit exceeded - policies: %s",
                ", ".join(i.policy for i in self.state["violated"]),
                extra={"violated": self.state["violated"]},
            )
            raise self.error_class(
                violated_policies=[i.policy for i in self.state["violated"]]
            )
        return not self.state["violated"]
