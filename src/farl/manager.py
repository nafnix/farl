import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Generic, TypeVar, overload

from fastapi import Depends, Request
from limits.limits import (
    RateLimitItem,
    RateLimitItemPerDay,
    RateLimitItemPerHour,
    RateLimitItemPerMinute,
    RateLimitItemPerMonth,
    RateLimitItemPerSecond,
    RateLimitItemPerYear,
)

from farl.constants import STATE_KEY
from farl.decorators import rate_limit
from farl.exceptions import FarlError, QuotaExceeded
from farl.types import (
    AnyFarlProtocol,
    Cost,
    FarlState,
    GetCostDependency,
    GetKeyDependency,
    GetPartitionCostMappingDependency,
    GetPeriodDependency,
    GetPolicyNameDependency,
    GetPolicySettingsDependency,
    GetQuotaDependency,
    GetQuotaUnitDependency,
    GetTimeTypeDependency,
    HeaderRateLimit,
    HeaderRateLimitPolicy,
    Key,
    PartitionCostMapping,
    Period,
    PolicyName,
    PolicySettings,
    Quota,
    QuotaUnit,
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


class AbstractMultiPartitionPolicyHandler(ABC, Generic[_T]):
    def __init__(
        self,
        request: Request,
        name: str | None,
        quota: Quota,
        quota_unit: str | None,
        time: TimeType,
        period: int,
        partition_costs: PartitionCostMapping,
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
        self.partition_costs = partition_costs
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


class MultiPartitionPolicyHandler(AbstractMultiPartitionPolicyHandler[bool]):
    @staticmethod
    def _get_policy_name(value: RateLimitItem):
        items = ["multi", value.GRANULARITY.name]
        if value.multiples != 1:
            items.insert(1, str(value.multiples))

        return "".join(items)

    async def __call__(self):
        value = self.get_limit_value()
        name = self.name if self.name is not None else self._get_policy_name(value)
        limiter = self.farl.limiter

        for partition_key in self.partition_costs:
            self.state["policy"].append(
                HeaderRateLimitPolicy(
                    name,
                    self.quota,
                    self.quota_unit,
                    self.period * value.GRANULARITY.seconds,
                    partition_key,
                )
            )

        violated_partitions = []
        for partition_key, cost in self.partition_costs.items():
            cost = DEFAULT_COST if cost is None else cost
            hit_result = limiter.hit(value, partition_key, cost=cost)
            if inspect.isawaitable(hit_result):
                hit_result = await hit_result

            stats_result = limiter.get_window_stats(value, partition_key)
            if inspect.isawaitable(stats_result):
                stats_result = await stats_result

            ratelimit = HeaderRateLimit(
                name,
                stats_result.remaining,
                stats_result.reset_time,
                partition_key,
            )
            self.state["state"].append(ratelimit)

            if hit_result is False:
                violated_partitions.append(partition_key)
                self.state["violated"].append(ratelimit)

        # 如果有任何分区违反限制，记录并抛出异常
        if violated_partitions and self.error_class is not None:
            logger.warning(
                "Rate limit exceeded - policy: %s",
                name,
                extra={"violated": self.state["violated"]},
            )
            raise self.error_class(
                violated_policies=[i.policy for i in self.state["violated"]]
            )

        return not violated_partitions


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

        result = all(await asyncio.gather(*[h() for h in handles]))
        if self.state["violated"] and self.error_class is not None:
            logger.warning(
                "Rate limit exceeded - policies: %s",
                ", ".join(i.policy for i in self.state["violated"]),
                extra={"violated": self.state["violated"]},
            )
            raise self.error_class(
                violated_policies=[i.policy for i in self.state["violated"]]
            )
        return result


_Function = TypeVar("_Function", bound=Callable)

DEFAULT_NAMESPACE = "default"
DEFAULT_TIME = "minute"
DEFAULT_PERIOD = 1
DEFAULT_COST = 1


class RateLimitPolicyManager:
    def __init__(
        self,
        error_class: type[FarlError] | None = QuotaExceeded,
        farl: AnyFarlProtocol | None = None,
    ) -> None:
        self.error_class = error_class
        self.farl = farl
        self._dependencies: list[Callable[[bool], Callable]] = []

    @staticmethod
    def _get_value(value):
        def _():
            return value

        return _

    def _get_farl_state(self, request: Request):
        state = request.scope.setdefault("state", {})
        farl_state: FarlState = state.setdefault(
            STATE_KEY,
            FarlState(policy=[], state=[], violated=[]),
        )
        farl: AnyFarlProtocol | None = self.farl or farl_state.get("farl")
        if farl is None:
            raise ValueError("farl instance is required")
        return farl, farl_state

    def create(
        self,
        quota: Quota | GetQuotaDependency,
        *,
        time: TimeType | GetTimeTypeDependency = DEFAULT_TIME,
        period: Period | GetPeriodDependency = DEFAULT_PERIOD,
        quota_unit: QuotaUnit | GetQuotaUnitDependency | None = None,
        cost: Cost | GetCostDependency = DEFAULT_COST,
        name: PolicyName | GetPolicyNameDependency | None = None,
        namespace: Key | GetKeyDependency = DEFAULT_NAMESPACE,
        partition: Key | GetKeyDependency | None = None,
        handler: type[AbstractPolicyHandler] = PolicyHandler,
    ):
        name = name if callable(name) else self._get_value(name)
        quota = quota if callable(quota) else self._get_value(quota)
        quota_unit = quota_unit if callable(quota_unit) else self._get_value(quota_unit)
        time = time if callable(time) else self._get_value(time)
        period = period if callable(period) else self._get_value(period)
        cost = cost if callable(cost) else self._get_value(cost)
        namespace = namespace if callable(namespace) else self._get_value(namespace)
        partition = partition if callable(partition) else self._get_value(partition)

        def wrapper(auto_error: bool):  # noqa: FBT001
            async def dependency(
                request: Request,
                name_=Depends(name),
                quota_=Depends(quota),
                quota_unit_=Depends(quota_unit),
                time_=Depends(time),
                period_=Depends(period),
                cost_=Depends(cost),
                namespace_=Depends(namespace),
                partition_=Depends(partition),
            ):
                farl, state = self._get_farl_state(request)
                call = handler(
                    request,
                    name=name_,
                    quota=quota_,
                    quota_unit=quota_unit_,
                    time=time_,
                    period=period_,
                    cost=cost_,
                    namespace=namespace_,
                    partition=partition_,
                    state=state,
                    farl=farl,
                    error_class=self.error_class if auto_error else None,
                )

                result = call()
                if inspect.isawaitable(result):
                    result = await result

                return result

            return dependency

        self._dependencies.append(wrapper)
        return wrapper(auto_error=True)

    def create_multi_partition(
        self,
        quota: Quota | GetQuotaDependency,
        *,
        time: TimeType | GetTimeTypeDependency = DEFAULT_TIME,
        period: Period | GetPeriodDependency = DEFAULT_PERIOD,
        quota_unit: QuotaUnit | GetQuotaUnitDependency | None = None,
        partition_costs: PartitionCostMapping | GetPartitionCostMappingDependency,
        name: PolicyName | GetPolicyNameDependency | None = None,
        namespace: Key | GetKeyDependency = DEFAULT_NAMESPACE,
        handler: type[
            AbstractMultiPartitionPolicyHandler
        ] = MultiPartitionPolicyHandler,
    ):
        name = name if callable(name) else self._get_value(name)
        quota = quota if callable(quota) else self._get_value(quota)
        quota_unit = quota_unit if callable(quota_unit) else self._get_value(quota_unit)
        time = time if callable(time) else self._get_value(time)
        period = period if callable(period) else self._get_value(period)
        partition_costs = (
            partition_costs
            if callable(partition_costs)
            else self._get_value(partition_costs)
        )
        namespace = namespace if callable(namespace) else self._get_value(namespace)

        def wrapper(auto_error: bool):  # noqa: FBT001
            async def dependency(
                request: Request,
                name_=Depends(name),
                quota_=Depends(quota),
                quota_unit_=Depends(quota_unit),
                time_=Depends(time),
                period_=Depends(period),
                partition_costs_=Depends(partition_costs),
                namespace_=Depends(namespace),
            ):
                farl, state = self._get_farl_state(request)

                call = handler(
                    request,
                    name=name_,
                    quota=quota_,
                    quota_unit=quota_unit_,
                    time=time_,
                    period=period_,
                    namespace=namespace_,
                    state=state,
                    farl=farl,
                    error_class=self.error_class if auto_error else None,
                    partition_costs=partition_costs_,
                )
                result = call()
                if inspect.isawaitable(result):
                    result = await result

                return result

            return dependency

        self._dependencies.append(wrapper)
        return wrapper(auto_error=True)

    def create_settings(
        self,
        settings: (
            PolicySettings | Sequence[PolicySettings] | GetPolicySettingsDependency
        ),
        handler: type[AbstractPolicySettingsHandler] = PolicySettingsHandler,
    ):
        settings = settings if callable(settings) else self._get_value(settings)

        def wrapper(auto_error: bool):  # noqa: FBT001
            async def dependency(request: Request, settings=Depends(settings)):
                farl, state = self._get_farl_state(request)
                call = handler(
                    request=request,
                    settings=settings,
                    state=state,
                    farl=farl,
                    error_class=self.error_class if auto_error else None,
                )
                result = call()
                if inspect.isawaitable(result):
                    result = await result

                return result

            return dependency

        self._dependencies.append(wrapper)
        return wrapper(auto_error=True)

    @overload
    def __call__(self) -> Callable: ...
    @overload
    def __call__(self, fn: _Function) -> _Function: ...

    def __call__(self, fn: _Function | None = None) -> _Function | Callable:
        if fn is not None:
            return rate_limit(self)(fn)

        deps = [ii(False) for ii in self._dependencies]  # noqa: FBT003

        def dependency(request: Request, **_kwds):
            state = request.scope.setdefault("state", {})
            farl_state: FarlState = state.setdefault(
                STATE_KEY,
                FarlState(policy=[], state=[], violated=[]),
            )
            if self.error_class is not None and farl_state["violated"]:
                violated_policies = [v.policy for v in farl_state["violated"]]
                logger.warning(
                    "Multiple rate limits violated - policies: %s",
                    ", ".join(violated_policies),
                    extra={"violated": farl_state["violated"]},
                )
                raise self.error_class(violated_policies=violated_policies)

        sign = inspect.signature(dependency)
        param_mapping = sign.parameters.copy()
        param_mapping.pop("_kwds")
        params = list(param_mapping.values())
        params.extend(
            inspect.Parameter(
                name=f"_ratelimit_{index}",
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=Depends(i),
            )
            for index, i in enumerate(deps)
        )
        new_sign = sign.replace(parameters=params)
        setattr(dependency, "__signature__", new_sign)  # noqa: B010
        return dependency
