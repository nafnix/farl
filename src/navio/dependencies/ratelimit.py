import inspect
import logging
from collections.abc import Callable
from copy import copy

from fastapi import Depends, Request
from limits import RateLimitItem

from navio.constants import STATE_KEY
from navio.exceptions import NavioError, QuotaExceeded
from navio.types import (
    AnyNavioProtocol,
    CostResult,
    GetCostDependency,
    GetKeyDependency,
    GetRateLimitArgumentDependency,
    KeyResult,
    RateLimitArgument,
)
from navio.utils import (
    HeaderRateLimit,
    HeaderRateLimitPolicy,
    NavioState,
    parse_rate_limit_value,
)

from .utils import request_endpoint


logger = logging.getLogger("navio")


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
    navio: AnyNavioProtocol,
    navio_state: NavioState,
    policy_name: str,
    value: RateLimitItem,
    quota_unit: str | None,
    partition_key: str,
    keys: list,
    cost: int,
):
    limiter = navio.limiter
    navio_state["policy"].append(
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

    navio_state["state"].append(ratelimit)

    if hit_result is False:
        logger.warning(
            "Rate limit exceeded for partition_key: %s, keys: %s, cost: %s",
            partition_key,
            keys,
            cost,
        )
        navio_state["violated"].append(ratelimit)


def _value(v):
    def _():
        return v

    return _


def rate_limit(
    argument: RateLimitArgument | GetRateLimitArgumentDependency,
    *,
    policy_name: str | None = None,
    quota_unit: str | None = None,
    get_key: KeyResult | GetKeyDependency | None = None,
    get_partition_key: KeyResult | GetKeyDependency = request_endpoint,
    get_cost: CostResult | GetCostDependency = 1,
    error_class: type[NavioError] | None = QuotaExceeded,
    navio: AnyNavioProtocol | None = None,
):
    value = argument if callable(argument) else _value(argument)

    default_policy_name = policy_name

    key_dep = get_key if callable(get_key) else _value(get_key)

    partition_key_dep = (
        get_partition_key if callable(get_partition_key) else _value(get_partition_key)
    )

    cost_dep = get_cost if callable(get_cost) else _value(get_cost)

    async def dependency(
        request: Request,
        value: RateLimitArgument = Depends(value),
        partition_key: KeyResult = Depends(partition_key_dep),
        key: KeyResult | None = Depends(key_dep),
        cost: CostResult = Depends(cost_dep),
    ):
        state = request.scope.setdefault("state", {})
        navio_state: NavioState = state.setdefault(
            STATE_KEY,
            NavioState(policy=[], state=[], violated=[]),
        )
        _navio: AnyNavioProtocol | None = navio or navio_state.get("navio")
        if _navio is None:
            raise ValueError("navio instance is required")

        pk_ = partition_key
        pk = pk_ if isinstance(pk_, str) else ":".join(pk_)

        keys = ([key] if isinstance(key, str) else list(key)) if key else []
        keys.append(pk)

        limit_values = parse_rate_limit_value(value)

        if len(limit_values) == 1:
            value = _update_namespace(_navio.namespace, limit_values[0])

            policy_name = default_policy_name or _get_policy_name(value)

            await _handle(
                navio=_navio,
                navio_state=navio_state,
                policy_name=policy_name,
                value=value,
                quota_unit=quota_unit,
                partition_key=pk,
                keys=keys,
                cost=cost,
            )
        else:
            for i in limit_values:
                value = _update_namespace(_navio.namespace, i)

                policy_name = _get_policy_name(value)
                if default_policy_name is not None:
                    policy_name = f"{default_policy_name}-{policy_name}"

                await _handle(
                    navio=_navio,
                    navio_state=navio_state,
                    policy_name=policy_name,
                    value=value,
                    quota_unit=quota_unit,
                    partition_key=pk,
                    keys=keys,
                    cost=cost,
                )

        if navio_state["violated"] and error_class is not None:
            raise error_class(
                violated_policies=[i.policy for i in navio_state["violated"]]
            )

    return dependency


def rate_limits(*args: Callable, error_class: type[NavioError] = QuotaExceeded):
    def dependency(request: Request, **_ratelimit_deps):
        state: dict = request.scope.setdefault("state", {})
        navio_state: NavioState = state.setdefault(
            STATE_KEY,
            NavioState(policy=[], state=[], violated=[]),
        )
        if navio_state["violated"]:
            raise error_class(
                violated_policies=[i.policy for i in navio_state["violated"]]
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
