# import inspect
# from collections.abc import Callable
# from functools import wraps
# from typing import Any, TypeVar

# from fastapi import Depends

# from farl.exceptions import FarlError, QuotaExceeded
# from farl.types import (
#     AnyFarlProtocol,
#     CostResult,
#     GetCostDependency,
#     GetKeyDependency,
#     GetRateLimitPolicyDependency,
#     KeyResult,
#     RateLimitPolicy,
# )

# from .dependencies import rate_limit as rate_limit_dep
# from .dependencies.utils import request_endpoint


# Fn = TypeVar("Fn", bound=Callable)

# INJECT_DEP_PREFIX = "__farl_ratelimit"


# def _exclude_kwds(kwds: dict[str, Any]):
#     exclude = [i for i in kwds if i.startswith(INJECT_DEP_PREFIX)]
#     for i in exclude:
#         kwds.pop(i)


# def _create_call(fn: Fn) -> Fn:
#     if inspect.iscoroutinefunction(fn):

#         @wraps(fn)
#         async def async_hdlr(*args, **kwds):
#             _exclude_kwds(kwds)
#             return await fn(*args, **kwds)

#         return async_hdlr  # pyright: ignore[reportReturnType]

#     @wraps(fn)
#     def hdlr(*args, **kwds):
#         _exclude_kwds(kwds)
#         return fn(*args, **kwds)

#     return hdlr  # pyright: ignore[reportReturnType]


# def rate_limit(
#     argument: RateLimitPolicy | GetRateLimitPolicyDependency,
#     *,
#     policy_name: str | None = None,
#     quota_unit: str | None = None,
#     get_key: KeyResult | GetKeyDependency | None = None,
#     get_partition_key: KeyResult | GetKeyDependency = request_endpoint,
#     get_cost: CostResult | GetCostDependency = 1,
#     error_class: type[FarlError] | None = QuotaExceeded,
#     farl: AnyFarlProtocol | None = None,
# ):
#     dep = rate_limit_dep(
#         argument,
#         policy_name=policy_name,
#         quota_unit=quota_unit,
#         get_key=get_key,
#         get_partition_key=get_partition_key,
#         get_cost=get_cost,
#         error_class=error_class,
#         farl=farl,
#     )

#     def decorate(fn: Fn) -> Fn:
#         new_fn = _create_call(fn)

#         sign = inspect.signature(new_fn)
#         param_mappings = sign.parameters.copy()

#         dep_no = len([i for i in param_mappings if i.startswith(INJECT_DEP_PREFIX)]) + 1
#         dep_param = inspect.Parameter(
#             f"{INJECT_DEP_PREFIX}_{dep_no}",
#             kind=inspect.Parameter.KEYWORD_ONLY,
#             default=Depends(dep),
#         )
#         params = list(param_mappings.values())
#         if params and params[-1].kind == inspect.Parameter.VAR_KEYWORD:
#             params.insert(-1, dep_param)
#         else:
#             params.append(dep_param)
#         new_fn.__signature__ = sign.replace(parameters=params)
#         return new_fn

#     return decorate
