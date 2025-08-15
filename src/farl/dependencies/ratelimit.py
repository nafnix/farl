import asyncio
import inspect
import logging
from collections.abc import Callable
from copy import copy

from fastapi import Depends, Request
from limits import RateLimitItem

from farl.constants import STATE_KEY
from farl.exceptions import FarlError, QuotaExceeded
from farl.types import (
    AnyFarlProtocol,
    Cost,
    CostResult,
    GetCostDependency,
    GetKeyDependency,
    GetRateLimitPolicyDependency,
    KeyResult,
    RateLimitPolicy,
)
from farl.utils import (
    FarlState,
    HeaderRateLimit,
    HeaderRateLimitPolicy,
    parse_rate_limit_value,
)

from .utils import request_endpoint


logger = logging.getLogger("farl")


def _get_policy_name(value: RateLimitItem):
    if value.multiples != 1:
        return f"pre{value.multiples}{value.GRANULARITY.name}"
    return f"pre{value.GRANULARITY.name}"


def _update_namespace(
    namespace: str | None,
    value: RateLimitItem,
):
    if namespace:
        result = copy(value)
        result.namespace = namespace
    else:
        result = value

    return value


async def _handle(
    *,
    farl: AnyFarlProtocol,
    farl_state: FarlState,
    policy_name: str,
    value: RateLimitItem,
    quota_unit: str | None,
    partition_key: str,
    keys: list,
    cost: int,
):
    limiter = farl.limiter
    farl_state["policy"].append(
        HeaderRateLimitPolicy(
            policy_name,
            value.amount,
            quota_unit,
            value.multiples * value.GRANULARITY.seconds,
            partition_key,
        )
    )

    hit_result = limiter.hit(value, *keys, cost=cost)
    if inspect.isawaitable(hit_result):
        hit_result = await hit_result

    stats_result = limiter.get_window_stats(value, *keys)
    if inspect.isawaitable(stats_result):
        stats_result = await stats_result

    ratelimit = HeaderRateLimit(
        policy_name,
        stats_result.remaining,
        stats_result.reset_time,
        partition_key,
    )

    farl_state["state"].append(ratelimit)

    if hit_result is False:
        logger.warning(
            "Rate limit exceeded for partition_key: %s, keys: %s, cost: %s",
            partition_key,
            keys,
            cost,
        )
        farl_state["violated"].append(ratelimit)


def _value(v):
    def _():
        return v

    return _


async def _get_keys(
    request: Request,
    dep_key: KeyResult | None,
    farl: AnyFarlProtocol,
    partition_key: str,
):
    keys: list[str] = []
    if farl.key is not None:
        k = farl.key
        if callable(k):
            k = k(request)
            if inspect.isawaitable(k):
                k = await k

        keys.extend([k] if isinstance(k, str) else k)

    keys.extend(
        ([dep_key] if isinstance(dep_key, str) else list(dep_key)) if dep_key else []
    )
    keys.append(partition_key)
    return keys


async def _get_cost(
    request: Request,
    dep_cost: CostResult | None,
    farl: AnyFarlProtocol,
):
    result = dep_cost
    if result is None:
        if farl.cost:
            cost_result = farl.cost
            if callable(cost_result):
                cost_result = cost_result(request)
                if inspect.isawaitable(cost_result):
                    cost_result = await cost_result
            result = cost_result
        else:
            result = 1

    return result


async def _check_partition(
    *,
    request: Request,
    key: KeyResult | None,
    farl: AnyFarlProtocol,
    partition_key: str,
    cost: Cost,
    value: RateLimitPolicy,
    farl_state: FarlState,
    default_policy_name: str | None,
    quota_unit: str | None,
):
    keys = await _get_keys(request, key, farl, partition_key)

    limit_values = parse_rate_limit_value(value)

    if len(limit_values) == 1:
        value = _update_namespace(farl.namespace, limit_values[0])

        policy_name = default_policy_name or _get_policy_name(value)

        await _handle(
            farl=farl,
            farl_state=farl_state,
            policy_name=policy_name,
            value=value,
            quota_unit=quota_unit,
            partition_key=partition_key,
            keys=keys,
            cost=cost,
        )
    else:
        for i in limit_values:
            value = _update_namespace(farl.namespace, i)

            policy_name = _get_policy_name(value)
            if default_policy_name is not None:
                policy_name = f"{default_policy_name}-{policy_name}"

            await _handle(
                farl=farl,
                farl_state=farl_state,
                policy_name=policy_name,
                value=value,
                quota_unit=quota_unit,
                partition_key=partition_key,
                keys=keys,
                cost=cost,
            )


def rate_limit(
    policy: RateLimitPolicy | GetRateLimitPolicyDependency,
    *,
    policy_name: str | None = None,
    quota_unit: str | None = None,
    get_key: KeyResult | GetKeyDependency | None = None,
    get_partition_key: KeyResult | GetKeyDependency = request_endpoint,
    get_cost: CostResult | GetCostDependency | None = None,
    error_class: type[FarlError] | None = QuotaExceeded,
    farl: AnyFarlProtocol | None = None,
):
    default_policy_name = policy_name

    value_dep = policy if callable(policy) else _value(policy)
    key_dep = get_key if callable(get_key) else _value(get_key)
    partition_key_dep = (
        get_partition_key if callable(get_partition_key) else _value(get_partition_key)
    )
    cost_dep = get_cost if callable(get_cost) else _value(get_cost)

    async def dependency(
        request: Request,
        value: RateLimitPolicy = Depends(value_dep),
        partition_key: KeyResult = Depends(partition_key_dep),
        key: KeyResult | None = Depends(key_dep),
        cost: CostResult | None = Depends(cost_dep),
    ):
        state = request.scope.setdefault("state", {})
        farl_state: FarlState = state.setdefault(
            STATE_KEY,
            FarlState(policy=[], state=[], violated=[]),
        )
        _farl: AnyFarlProtocol | None = farl or farl_state.get("farl")
        if _farl is None:
            raise ValueError("farl instance is required")

        if isinstance(partition_key, str):
            partition_key = [partition_key]

        costs = await _get_cost(request, cost, _farl)

        if isinstance(costs, int):
            costs = [costs for _ in partition_key]
        elif cost is None:
            costs = [1 for _ in partition_key]

        partition_keys = partition_key

        # Validate that costs and partition_keys have the same length
        if len(costs) != len(partition_keys):
            raise ValueError(
                f"Number of costs ({len(costs)}) must match number "
                f"of partition keys ({len(partition_keys)})"
            )

        await asyncio.gather(
            *[
                _check_partition(
                    request=request,
                    key=key,
                    farl=_farl,
                    partition_key=partition_key,
                    cost=cost,
                    value=value,
                    farl_state=farl_state,
                    default_policy_name=default_policy_name,
                    quota_unit=quota_unit,
                )
                for partition_key, cost in zip(partition_keys, costs, strict=True)
            ]
        )

        if farl_state["violated"] and error_class is not None:
            raise error_class(
                violated_policies=[i.policy for i in farl_state["violated"]]
            )

    return dependency


def rate_limits(*args: Callable, error_class: type[FarlError] = QuotaExceeded):
    def dependency(request: Request, **_ratelimit_deps):
        state: dict = request.scope.setdefault("state", {})
        farl_state: FarlState = state.setdefault(
            STATE_KEY,
            FarlState(policy=[], state=[], violated=[]),
        )
        if farl_state["violated"]:
            raise error_class(
                violated_policies=[i.policy for i in farl_state["violated"]]
            )

    sign = inspect.signature(dependency)
    param_mapping = sign.parameters.copy()
    param_mapping.pop("_ratelimit_deps")
    params = list(param_mapping.values())
    params.extend(
        inspect.Parameter(
            name=f"_ratelimit_dep_{index}",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=Depends(i),
        )
        for index, i in enumerate(args)
    )

    new_sign = sign.replace(parameters=params)
    setattr(dependency, "__signature__", new_sign)  # noqa: B010
    return dependency
