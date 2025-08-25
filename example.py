from fastapi import Depends, FastAPI

from farl import (
    Farl,
    FarlError,
    FarlMiddleware,
    RateLimitPolicyManager,
    farl_exceptions_handler,
)
from farl.types import PolicySettings


farl = Farl()
mng = RateLimitPolicyManager(farl=farl)


app = FastAPI()
app.add_exception_handler(FarlError, farl_exceptions_handler)
app.add_middleware(FarlMiddleware)


def fake_dep():
    return object()


def dynamic_policy(
    value=Depends(fake_dep, use_cache=False),
    value2=Depends(fake_dep, use_cache=False),
):
    return [
        PolicySettings(
            name=str(value),
            quota=10,
            cost=5,
            namespace="default-ns",
            partition=f"user:{value}",
        ),
        PolicySettings(name=str(value2), quota=10, namespace="ns2"),
    ]


dependency = mng.create({"quota": 1, "time": "day"})


@app.get("/exam1", dependencies=[Depends(dependency)])
def pre_minute_10_quota(): ...


@app.get("/exam2")
@mng
def managed_endpoint(): ...


@app.get("/exam3", dependencies=[Depends(mng())])
def managed_endpoint2(): ...


mng.create(dynamic_policy)


@app.get("/exam4")
@mng
def managed_endpoint3(): ...
