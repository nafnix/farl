from farl.constants import HEADER_RATELIMIT, HEADER_RATELIMIT_POLICY
from farl.exceptions import FarlError


def openapi(model: type[FarlError]):
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
