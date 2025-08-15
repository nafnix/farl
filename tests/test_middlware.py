from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from farl.constants import HEADER_RATELIMIT, HEADER_RATELIMIT_POLICY, STATE_KEY
from farl.middleware import FarlMiddleware
from farl.utils import FarlState, HeaderRateLimit, HeaderRateLimitPolicy


app = FastAPI()
app.add_middleware(FarlMiddleware)


a_policy = HeaderRateLimitPolicy("testvalue", 10, None, 60, "/a:GET")
a_r = HeaderRateLimit(a_policy.policy, 1, 23, a_policy.partition_key)


@app.get("/a")
def a(request: Request):
    state: dict = request.scope["state"]
    farl_state: FarlState = state[STATE_KEY]

    farl_state["policy"].append(a_policy)
    farl_state["state"].append(a_r)


b_policy = [
    HeaderRateLimitPolicy("testvalue-preMinute", 10, None, 60, "/b:GET"),
    HeaderRateLimitPolicy("testvalue-preHour", 10, None, 60 * 60, "/b:GET"),
]
b_r = [
    HeaderRateLimit(b_policy[0].policy, 6, 5, b_policy[0].partition_key),
    HeaderRateLimit(b_policy[1].policy, 3, 5, b_policy[1].partition_key),
]


@app.get("/b")
async def b(request: Request):
    state: dict = request.scope["state"]
    farl_state: FarlState = state[STATE_KEY]
    farl_state["policy"].extend(b_policy)
    farl_state["state"].extend(b_r)


c_policy = [
    HeaderRateLimitPolicy("testvalue-preMinute", 10, None, 60, "/c:GET"),
    HeaderRateLimitPolicy("testvalue-preHour", 10, None, 60 * 60, "/c:GET"),
    HeaderRateLimitPolicy("testvalue-preDay", 10, None, 60 * 60 * 24, "/c:GET"),
]
c_r = [
    HeaderRateLimit(c_policy[0].policy, 6, 2, c_policy[0].partition_key),
    HeaderRateLimit(c_policy[1].policy, 3, 10, c_policy[1].partition_key),
    HeaderRateLimit(c_policy[2].policy, 1, None, c_policy[2].partition_key),
]


@app.get("/c")
async def c(request: Request):
    state: dict = request.scope["state"]
    farl_state: FarlState = state[STATE_KEY]
    farl_state["policy"].extend(c_policy)
    farl_state["state"].extend(c_r)


d_policy = [
    HeaderRateLimitPolicy("testvalue-preMinute", 10, None, 60, "/d:GET"),
    HeaderRateLimitPolicy("testvalue-preHour", 10, None, 60 * 60, "/d:GET"),
    HeaderRateLimitPolicy("testvalue-preDay", 10, None, 60 * 60 * 24, "/d:GET"),
]
d_r = [
    HeaderRateLimit(d_policy[0].policy, 6, None, d_policy[0].partition_key),
    HeaderRateLimit(d_policy[1].policy, 3, None, d_policy[1].partition_key),
    HeaderRateLimit(d_policy[2].policy, 1, None, d_policy[2].partition_key),
]


@app.get("/d")
async def d(request: Request):
    state: dict = request.scope["state"]
    farl_state: FarlState = state[STATE_KEY]
    farl_state["policy"].extend(d_policy)
    farl_state["state"].extend(d_r)


def test_middleware():
    with TestClient(app) as client:
        a_res = client.get("/a")
        assert a_res.is_success
        assert a_res.headers[HEADER_RATELIMIT_POLICY] == str(a_policy)
        assert a_res.headers[HEADER_RATELIMIT] == str(a_r)

        b_res = client.get("/b")
        assert b_res.is_success
        b_value = ",".join(str(i) for i in b_policy)
        assert b_res.headers[HEADER_RATELIMIT_POLICY] == b_value
        assert b_res.headers[HEADER_RATELIMIT] == str(b_r[1])

        c_res = client.get("/c")
        assert c_res.is_success
        c_value = ",".join(str(i) for i in c_policy)
        assert c_res.headers[HEADER_RATELIMIT_POLICY] == c_value
        assert c_res.headers[HEADER_RATELIMIT] == str(c_r[1])

        d_res = client.get("/d")
        assert d_res.is_success
        d_value = ",".join(str(i) for i in d_policy)
        assert d_res.headers[HEADER_RATELIMIT_POLICY] == d_value
        assert d_res.headers[HEADER_RATELIMIT] == str(d_r[2])
