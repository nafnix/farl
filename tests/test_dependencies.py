import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from farl import AsyncFarl, Farl
from farl.dependencies import rate_limit, rate_limits
from farl.exceptions import QuotaExceeded


app = FastAPI()


@pytest.fixture
def api():
    with TestClient(app) as client:
        yield client


@app.get(
    "/unset-farl",
    dependencies=[Depends(rate_limit({"amount": 1}))],
)
def unset_farl(): ...


def test_unset_farl(api: TestClient):
    with pytest.raises(ValueError, match="farl instance is required") as excinfo:
        api.get("/unset-farl")
    assert excinfo.type is ValueError


@app.get(
    "/raise-exc",
    dependencies=[
        Depends(
            rate_limit(
                {"amount": 1},
                farl=Farl(),
            )
        )
    ],
)
def raise_exc(): ...


def test_raise_exc(api: TestClient):
    res = api.get("/raise-exc")
    assert res.is_success

    with pytest.raises(QuotaExceeded) as excinfo:
        api.get("/raise-exc")

    assert excinfo.type is QuotaExceeded
    assert excinfo.value.data.model_dump()["violated-policies"] == ["preminute"]


@app.get(
    "/multiple-ratelimit",
    dependencies=[
        Depends(
            rate_limit(
                [
                    {"amount": 1},
                    {"amount": 2, "multiples": 3, "time": "second"},
                ],
                farl=Farl(),
                policy_name="multiple",
            )
        )
    ],
)
def multiple_ratelimit(): ...


def test_multiple_ratelimit(api: TestClient):
    api.get("/multiple-ratelimit")
    with pytest.raises(QuotaExceeded) as excinfo:
        api.get("/multiple-ratelimit")

    assert excinfo.type is QuotaExceeded
    assert excinfo.value.data.model_dump()["violated-policies"] == [
        "multiple-preminute"
    ]


@app.get(
    "/rate-limits",
    dependencies=[
        Depends(
            rate_limits(
                rate_limit(
                    {"amount": 1},
                    farl=Farl(namespace="test"),
                    policy_name="a",
                    error_class=None,
                ),
                rate_limit(
                    [
                        {"amount": 1, "time": "month"},
                        {"amount": 2, "multiples": 3, "time": "second"},
                    ],
                    farl=AsyncFarl(),
                    policy_name="b",
                    error_class=None,
                ),
            )
        )
    ],
)
def _rate_limits(): ...


def test_rate_limits(api: TestClient):
    api.get("/rate-limits")
    with pytest.raises(QuotaExceeded) as excinfo:
        api.get("/rate-limits")

    assert excinfo.type is QuotaExceeded
    assert excinfo.value.data.model_dump()["violated-policies"] == ["a", "b-premonth"]
