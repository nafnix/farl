from typing import Literal

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import Field, create_model


class FarlError(Exception):
    status: int
    type: str
    title: str
    media_type: str = "application/problem+json"

    @classmethod
    def _build_schema(cls):
        cls.model = create_model(
            cls.__name__,
            type=(Literal[cls.type], ...),
            title=(Literal[cls.title], ...),
            violated_policies=(list[str], Field(..., alias="violated-policies")),
            __config__={"serialize_by_alias": True, "validate_by_alias": False},
        )

    def __init_subclass__(cls) -> None:
        cls._build_schema()

    def __init__(
        self,
        *,
        violated_policies: list[str] = [],  # noqa: B006
        headers: dict | None = None,
    ) -> None:
        self.data = self.model(
            type=self.type,
            title=self.title,
            violated_policies=violated_policies,
        )
        self.headers = headers or {}


# 超出配额
# 如果服务器希望通知客户端其发送的请求超出了一个或多个配额策略，则可以使用此问题类型。
class QuotaExceeded(FarlError):
    status = 429

    type = "https://iana.org/assignments/http-problem-types#quota-exceeded"
    title = "Request cannot be satisifed as assigned quota has been exceeded"


# 暂时减少容量
# 如果服务器想要告知客户端，由于服务限制导致容量暂时减少，
# 客户端发送的请求超出当前无法满足的范围，则可以使用此问题类型。
# 服务器可以选择包含一个 RateLimit-Policy 字段，指示新的暂时较低的配额。
# 此问题类型将扩展成员“violated-policies”定义为一个字符串数组，其值为超出配额的策略名称
class TemporaryReducedCapacity(FarlError):
    status = 503

    type = "https://iana.org/assignments/http-problem-types#temporary-reduced-capacity"
    title = "Request cannot be satisifed due to temporary server capacity constraints"


class AbnormalUsageDetected(FarlError):
    status = 429

    type = "https://iana.org/assignments/http-problem-types#abnormal-usage-detected"
    title = "Request not satisifed due to detection of abnormal request pattern"


def farl_exceptions_handler(_request: Request, exc):
    return JSONResponse(
        exc.data.model_dump(exclude_unset=True),
        status_code=exc.status,
        media_type=exc.media_type,
        headers=exc.headers,
    )
