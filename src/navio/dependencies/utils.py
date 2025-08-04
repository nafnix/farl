from fastapi import Request


def request_endpoint(request: Request):
    request_path = request.scope.get("path") or ""
    request_method = request.method
    return f"{request_path}:{request_method}"
