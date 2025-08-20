import inspect
import logging
from collections.abc import Callable, Sequence
from typing import TypeVar, overload

from fastapi import Depends, Request

from farl.constants import (
    DEFAULT_COST,
    DEFAULT_NAMESPACE,
    DEFAULT_PERIOD,
    DEFAULT_TIME,
    STATE_KEY,
)
from farl.decorators import rate_limit
from farl.exceptions import FarlError, QuotaExceeded
from farl.handlers import (
    AbstractMultiPartitionPolicyHandler,
    AbstractPolicyHandler,
    AbstractPolicySettingsHandler,
    MultiPartitionPolicyHandler,
    PolicyHandler,
    PolicySettingsHandler,
)
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
    Key,
    PartitionCostMapping,
    Period,
    PolicyName,
    PolicySettings,
    Quota,
    QuotaUnit,
    TimeType,
)


logger = logging.getLogger("farl")

_Function = TypeVar("_Function", bound=Callable)


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
        if callable(fn):
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
