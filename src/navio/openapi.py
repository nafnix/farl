from navio.constants import HEADER_RATELIMIT, HEADER_RATELIMIT_POLICY
from navio.exceptions import NavioError


def openapi(model: type[NavioError]):
    return {
        "model": model,
        "headers": {
            HEADER_RATELIMIT_POLICY: {
                "schema": {
                    "type": "string",
                    "title": "RateLimit-Policy",
                }
            },
            HEADER_RATELIMIT: {
                "schema": {
                    "type": "string",
                    "title": "RateLimit",
                }
            },
        },
    }
