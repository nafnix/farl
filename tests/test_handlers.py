from unittest.mock import MagicMock

import pytest
from fastapi import Request

from farl.handlers import PolicyHandler, PolicySettingsHandler
from farl.types import FarlState, PolicySettings


class TestPolicyHandler:
    def setup_method(self):
        self.mock_request = MagicMock(spec=Request)
        self.mock_state: FarlState = {"policy": [], "state": [], "violated": []}
        self.mock_farl = MagicMock()
        self.mock_farl.limiter = MagicMock()

    def test_get_limit_value_year(self):
        handler = PolicyHandler(
            request=self.mock_request,
            name=None,
            quota=10,
            quota_unit=None,
            time="year",
            period=1,
            cost=1,
            partition=None,
            namespace="test_ns",
            state=self.mock_state,
            farl=self.mock_farl,
            error_class=None,
        )
        limit = handler.get_limit_value()
        assert limit.amount == 10
        assert limit.multiples == 1
        assert limit.namespace == "test_ns"
        assert hasattr(limit, "GRANULARITY")

    def test_get_limit_value_month(self):
        handler = PolicyHandler(
            request=self.mock_request,
            name=None,
            quota=5,
            quota_unit=None,
            time="month",
            period=2,
            cost=1,
            partition=None,
            namespace="test_ns",
            state=self.mock_state,
            farl=self.mock_farl,
            error_class=None,
        )
        limit = handler.get_limit_value()
        assert limit.amount == 5
        assert limit.multiples == 2
        assert limit.namespace == "test_ns"

    def test_get_limit_value_day(self):
        handler = PolicyHandler(
            request=self.mock_request,
            name=None,
            quota=20,
            quota_unit=None,
            time="day",
            period=1,
            cost=1,
            partition=None,
            namespace="test_ns",
            state=self.mock_state,
            farl=self.mock_farl,
            error_class=None,
        )
        limit = handler.get_limit_value()
        assert limit.amount == 20
        assert limit.multiples == 1
        assert limit.namespace == "test_ns"

    def test_get_limit_value_hour(self):
        handler = PolicyHandler(
            request=self.mock_request,
            name=None,
            quota=100,
            quota_unit=None,
            time="hour",
            period=1,
            cost=1,
            partition=None,
            namespace="test_ns",
            state=self.mock_state,
            farl=self.mock_farl,
            error_class=None,
        )
        limit = handler.get_limit_value()
        assert limit.amount == 100
        assert limit.multiples == 1
        assert limit.namespace == "test_ns"

    def test_get_limit_value_minute(self):
        handler = PolicyHandler(
            request=self.mock_request,
            name=None,
            quota=60,
            quota_unit=None,
            time="minute",
            period=1,
            cost=1,
            partition=None,
            namespace="test_ns",
            state=self.mock_state,
            farl=self.mock_farl,
            error_class=None,
        )
        limit = handler.get_limit_value()
        assert limit.amount == 60
        assert limit.multiples == 1
        assert limit.namespace == "test_ns"

    def test_get_limit_value_second(self):
        handler = PolicyHandler(
            request=self.mock_request,
            name=None,
            quota=30,
            quota_unit=None,
            time="second",
            period=1,
            cost=1,
            partition=None,
            namespace="test_ns",
            state=self.mock_state,
            farl=self.mock_farl,
            error_class=None,
        )
        limit = handler.get_limit_value()
        assert limit.amount == 30
        assert limit.multiples == 1
        assert limit.namespace == "test_ns"

    def test_get_limit_value_unsupported_time(self):
        handler = PolicyHandler(
            request=self.mock_request,
            name=None,
            quota=10,
            quota_unit=None,
            time="unsupported",  # type: ignore[arg-type]
            period=1,
            cost=1,
            partition=None,
            namespace="test_ns",
            state=self.mock_state,
            farl=self.mock_farl,
            error_class=None,
        )
        with pytest.raises(ValueError, match="Unsupported time type"):
            handler.get_limit_value()

    @pytest.mark.anyio
    async def test_call_success(self):
        # Mock limiter behavior
        mock_hit_result = True
        mock_stats = MagicMock()
        mock_stats.remaining = 5
        mock_stats.reset_time = 1234567890

        self.mock_farl.limiter.hit.return_value = mock_hit_result
        self.mock_farl.limiter.get_window_stats.return_value = mock_stats

        handler = PolicyHandler(
            request=self.mock_request,
            name="test_policy",
            quota=10,
            quota_unit="requests",
            time="minute",
            period=1,
            cost=1,
            partition="user:123",
            namespace="test_ns",
            state=self.mock_state,
            farl=self.mock_farl,
            error_class=None,
        )

        result = await handler()

        assert result is True
        assert len(self.mock_state["policy"]) == 1
        assert len(self.mock_state["state"]) == 1
        assert len(self.mock_state["violated"]) == 0

        policy = self.mock_state["policy"][0]
        assert policy.policy == "test_policy"
        assert policy.quota == 10

        state = self.mock_state["state"][0]
        assert state.policy == "test_policy"
        assert state.remaining == 5

    @pytest.mark.anyio
    async def test_call_violation_with_error_class(self):
        from farl.exceptions import QuotaExceeded

        mock_hit_result = False
        mock_stats = MagicMock()
        mock_stats.remaining = 0
        mock_stats.reset_time = 1234567890

        self.mock_farl.limiter.hit.return_value = mock_hit_result
        self.mock_farl.limiter.get_window_stats.return_value = mock_stats

        handler = PolicyHandler(
            request=self.mock_request,
            name="test_policy",
            quota=10,
            quota_unit="requests",
            time="minute",
            period=1,
            cost=1,
            partition="user:123",
            namespace="test_ns",
            state=self.mock_state,
            farl=self.mock_farl,
            error_class=QuotaExceeded,
        )

        with pytest.raises(QuotaExceeded) as exc_info:
            await handler()

        assert exc_info.value.data.model_dump()["violated-policies"] == ["test_policy"]
        assert len(self.mock_state["violated"]) == 1


class TestPolicySettingsHandler:
    def setup_method(self):
        self.mock_request = MagicMock(spec=Request)
        self.mock_state: FarlState = {"policy": [], "state": [], "violated": []}
        self.mock_farl = MagicMock()
        self.mock_farl.limiter = MagicMock()

    @pytest.mark.anyio
    async def test_call_single_policy_success(self):
        mock_hit_result = True
        mock_stats = MagicMock()
        mock_stats.remaining = 5
        mock_stats.reset_time = 1234567890

        self.mock_farl.limiter.hit.return_value = mock_hit_result
        self.mock_farl.limiter.get_window_stats.return_value = mock_stats

        settings: PolicySettings = {
            "quota": 10,
            "name": "test_policy",
            "quota_unit": "requests",
            "time": "minute",
            "period": 1,
            "cost": 1,
            "partition": "user:123",
            "namespace": "test_ns",
        }

        handler = PolicySettingsHandler(
            request=self.mock_request,
            settings=settings,
            state=self.mock_state,
            farl=self.mock_farl,
            error_class=None,
        )

        result = await handler()

        assert result is True
        assert len(self.mock_state["policy"]) == 1
        assert len(self.mock_state["state"]) == 1
        assert len(self.mock_state["violated"]) == 0

    @pytest.mark.anyio
    async def test_call_multiple_policies_success(self):
        mock_hit_result = True
        mock_stats = MagicMock()
        mock_stats.remaining = 5
        mock_stats.reset_time = 1234567890

        self.mock_farl.limiter.hit.return_value = mock_hit_result
        self.mock_farl.limiter.get_window_stats.return_value = mock_stats

        handler = PolicySettingsHandler(
            request=self.mock_request,
            settings=[
                {
                    "quota": 10,
                    "name": "policy1",
                    "time": "minute",
                    "period": 1,
                    "cost": 1,
                    "namespace": "test_ns",
                },
                {
                    "quota": 20,
                    "name": "policy2",
                    "time": "hour",
                    "period": 1,
                    "cost": 1,
                    "namespace": "test_ns",
                },
            ],
            state=self.mock_state,
            farl=self.mock_farl,
            error_class=None,
        )

        result = await handler()

        assert result is True
        assert len(self.mock_state["policy"]) == 2
        assert len(self.mock_state["state"]) == 2
        assert len(self.mock_state["violated"]) == 0

    @pytest.mark.anyio
    async def test_call_violation_with_error_class(self):
        from farl.exceptions import QuotaExceeded

        mock_hit_result = False
        mock_stats = MagicMock()
        mock_stats.remaining = 0
        mock_stats.reset_time = 1234567890

        self.mock_farl.limiter.hit.return_value = mock_hit_result
        self.mock_farl.limiter.get_window_stats.return_value = mock_stats

        settings: PolicySettings = {
            "quota": 10,
            "name": "test_policy",
            "time": "minute",
            "period": 1,
            "cost": 1,
            "namespace": "test_ns",
        }

        handler = PolicySettingsHandler(
            request=self.mock_request,
            settings=settings,
            state=self.mock_state,
            farl=self.mock_farl,
            error_class=QuotaExceeded,
        )

        with pytest.raises(QuotaExceeded) as exc_info:
            await handler()

        assert exc_info.value.data.model_dump()["violated-policies"] == ["test_policy"]
        assert len(self.mock_state["violated"]) == 1

    @pytest.mark.anyio
    async def test_call_partial_violation(self):
        # First policy succeeds, second fails
        mock_hit_results = [True, False, False]
        mock_stats = MagicMock()
        mock_stats.remaining = 5
        mock_stats.reset_time = 1234567890

        self.mock_farl.limiter.hit.side_effect = mock_hit_results
        self.mock_farl.limiter.get_window_stats.return_value = mock_stats

        handler = PolicySettingsHandler(
            request=self.mock_request,
            settings=[
                {
                    "name": "policy1",
                    "quota": 100,
                    "time": "minute",
                    "period": 1,
                    "cost": 1,
                    "namespace": "test_ns",
                },
                {
                    "name": "policy2",
                    "quota": 20,
                    "time": "hour",
                    "period": 1,
                    "cost": 1,
                    "namespace": "test_ns",
                },
                {
                    "name": "policy3",
                    "quota": 20,
                    "time": "Y",
                    "period": 1,
                    "cost": 1,
                    "namespace": "test_ns",
                },
            ],
            state=self.mock_state,
            farl=self.mock_farl,
            error_class=None,
        )

        result = await handler()

        assert result is False  # all() returns False if any is False
        assert len(self.mock_state["policy"]) == 3
        assert len(self.mock_state["state"]) == 3
        assert len(self.mock_state["violated"]) == 2
