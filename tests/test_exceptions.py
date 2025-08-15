from fastapi import FastAPI
from fastapi.testclient import TestClient

from farl.exceptions import (
    AbnormalUsageDetected,
    FarlError,
    QuotaExceeded,
    TemporaryReducedCapacity,
    farl_exceptions_handler,
)


app = FastAPI()


quota_exceeded_exc = QuotaExceeded(violated_policies=["preMinute"])


@app.get("/quota-exceeded")
def quota_exceeded():
    raise quota_exceeded_exc


abnormal_usage_detected_exc = AbnormalUsageDetected(
    violated_policies=["preMinute", "preHour"]
)


@app.get("/abnormal-usage-detected")
def abnormal_usage_detected():
    raise abnormal_usage_detected_exc


temporary_reduced_capacity_exc = TemporaryReducedCapacity(
    violated_policies=["preMinute"]
)


@app.get("/temporary-reduced-capacity")
def temporary_reduced_capacity():
    raise temporary_reduced_capacity_exc


app.add_exception_handler(FarlError, farl_exceptions_handler)


def test_exceptions():
    with TestClient(app) as client:
        res = client.get("/quota-exceeded")
        assert res.is_error
        assert res.headers["content-type"] == quota_exceeded_exc.media_type
        res_data: dict = res.json()
        assert res_data["type"] == quota_exceeded_exc.type
        assert res_data["title"] == quota_exceeded_exc.title
        assert res.status_code == quota_exceeded_exc.status
        assert (
            res_data["violated-policies"] == quota_exceeded_exc.data.violated_policies  # pyright: ignore [reportAttributeAccessIssue]
        )

        res = client.get("/abnormal-usage-detected")
        assert res.is_error
        assert res.headers["content-type"] == abnormal_usage_detected_exc.media_type
        res_data: dict = res.json()
        assert res_data["type"] == abnormal_usage_detected_exc.type
        assert res_data["title"] == abnormal_usage_detected_exc.title
        assert res.status_code == abnormal_usage_detected_exc.status
        assert (
            res_data["violated-policies"]
            == abnormal_usage_detected_exc.data.violated_policies  # pyright: ignore [reportAttributeAccessIssue]
        )

        res = client.get("/temporary-reduced-capacity")
        assert res.is_error
        assert res.headers["content-type"] == temporary_reduced_capacity_exc.media_type
        res_data: dict = res.json()
        assert res_data["type"] == temporary_reduced_capacity_exc.type
        assert res_data["title"] == temporary_reduced_capacity_exc.title
        assert res.status_code == temporary_reduced_capacity_exc.status
        assert (
            res_data["violated-policies"]
            == temporary_reduced_capacity_exc.data.violated_policies  # pyright: ignore [reportAttributeAccessIssue]
        )
