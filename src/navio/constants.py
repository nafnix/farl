HEADER_RATELIMIT_POLICY = "RateLimit-Policy"


class RateLimitPolicyField:
    q = "q"
    """必需的“q”参数表示以配额单位衡量的此策略分配的配额。"""

    qu = "qu"
    """可选的“qu”参数值表示与“q”参数关联的配额单位。默认配额单位为“requests”。"""

    w = "w"
    """可选的“w”参数值传达一个时间窗口。"""

    pk = "pk"
    """可选的“pk”参数值传达与相应请求相关的分区键。"""
