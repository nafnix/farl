from unittest.mock import Mock

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient
from limits.limits import RateLimitItemPerMinute

from farl.base import Farl
from farl.constants import STATE_KEY
from farl.dependencies import RateLimitPolicyManager
from farl.dependencies.limits import MultiPartitionPolicyHandler, PolicyHandler
from farl.dependencies.utils import request_endpoint
from farl.exceptions import AbnormalUsageDetected, QuotaExceeded
from farl.types import FarlState


class TestRequestEndpoint:
    def test_request_endpoint_basic(self):
        mock_request = Mock(spec=Request)
        mock_request.scope = {"path": "/api/test"}
        mock_request.method = "GET"

        result = request_endpoint(mock_request)
        assert result == "/api/test:GET"

    def test_request_endpoint_no_path(self):
        mock_request = Mock(spec=Request)
        mock_request.scope = {}
        mock_request.method = "POST"

        result = request_endpoint(mock_request)
        assert result == ":POST"

    def test_request_endpoint_different_methods(self):
        mock_request = Mock(spec=Request)
        mock_request.scope = {"path": "/users"}

        for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
            mock_request.method = method
            result = request_endpoint(mock_request)
            assert result == f"/users:{method}"


class TestRateLimitPolicyManager:
    @pytest.fixture
    def mock_farl(self):
        farl = Mock(spec=Farl)
        farl.limiter = Mock()
        return farl

    @pytest.fixture
    def mock_request(self):
        request = Mock(spec=Request)
        request.scope = {"state": {}}
        return request

    @pytest.fixture
    def policy_manager(self, mock_farl):
        return RateLimitPolicyManager(farl=mock_farl)

    def test_init_default_error_class(self):
        manager = RateLimitPolicyManager()
        assert manager.error_class == QuotaExceeded

    def test_init_custom_error_class(self):
        manager = RateLimitPolicyManager(error_class=AbnormalUsageDetected)
        assert manager.error_class == AbnormalUsageDetected

    def test_init_no_error_class(self):
        manager = RateLimitPolicyManager(error_class=None)
        assert manager.error_class is None

    def test_get_value_static_method(self):
        manager = RateLimitPolicyManager()
        value_func = manager._get_value("test_value")
        assert value_func() == "test_value"

    def test_get_farl_state_new_state(self, policy_manager, mock_request):
        farl, state = policy_manager._get_farl_state(mock_request)

        assert farl == policy_manager.farl
        assert isinstance(state, dict)
        assert "policy" in state
        assert "state" in state
        assert "violated" in state

    def test_get_farl_state_existing_state(self, policy_manager, mock_request):
        existing_state = FarlState(policy=[], state=[], violated=[])
        mock_request.scope["state"][STATE_KEY] = existing_state

        farl, state = policy_manager._get_farl_state(mock_request)

        assert farl == policy_manager.farl
        assert state == existing_state

    def test_get_farl_state_no_farl_raises_error(self, mock_request):
        manager = RateLimitPolicyManager(farl=None)

        with pytest.raises(ValueError, match="farl instance is required"):
            manager._get_farl_state(mock_request)

    @pytest.mark.anyio
    async def test_create_basic_dependency(self, policy_manager):
        dependency = policy_manager.create(quota=10)

        assert callable(dependency)
        assert len(policy_manager._dependencies) == 1

    @pytest.mark.anyio
    async def test_create_with_callable_parameters(self, policy_manager):
        def get_quota():
            return 20

        def get_namespace():
            return "test_namespace"

        dependency = policy_manager.create(quota=get_quota, namespace=get_namespace)

        assert callable(dependency)

    @pytest.mark.anyio
    async def test_create_multi_partition_basic(self, policy_manager):
        partition_costs = {"user1": 1, "user2": 2}

        dependency = policy_manager.create_multi_partition(
            quota=10, partition_costs=partition_costs
        )

        assert callable(dependency)
        assert len(policy_manager._dependencies) == 1


class TestPolicyHandler:
    @pytest.fixture
    def mock_farl(self):
        farl = Mock(spec=Farl)
        farl.limiter = Mock()
        return farl

    @pytest.fixture
    def mock_request(self):
        return Mock(spec=Request)

    @pytest.fixture
    def farl_state(self):
        return FarlState(policy=[], state=[], violated=[])

    def test_get_limit_value_minute(self, mock_request, farl_state, mock_farl):
        handler = PolicyHandler(
            request=mock_request,
            name="test",
            quota=10,
            quota_unit=None,
            time="minute",
            period=1,
            cost=1,
            partition=None,
            namespace="default",
            state=farl_state,
            farl=mock_farl,
            error_class=None,
        )

        limit = handler.get_limit_value()
        assert isinstance(limit, RateLimitItemPerMinute)
        assert limit.amount == 10
        assert limit.multiples == 1

    def test_get_limit_value_unsupported_time(
        self, mock_request, farl_state, mock_farl
    ):
        handler = PolicyHandler(
            request=mock_request,
            name="test",
            quota=10,
            quota_unit=None,
            time="invalid",
            period=1,
            cost=1,
            partition=None,
            namespace="default",
            state=farl_state,
            farl=mock_farl,
            error_class=None,
        )

        with pytest.raises(ValueError, match="Unsupported time type: invalid"):
            handler.get_limit_value()

    def test_get_policy_name_basic(self):
        limit = RateLimitItemPerMinute(10, 1)
        name = PolicyHandler._get_policy_name(limit)
        assert name == "preminute"

    def test_get_policy_name_with_multiples(self):
        limit = RateLimitItemPerMinute(10, 5)
        name = PolicyHandler._get_policy_name(limit)
        assert name == "pre5minute"

    @pytest.mark.anyio
    async def test_call_success(self, mock_request, farl_state, mock_farl):
        mock_farl.limiter.hit.return_value = True

        mock_stats = Mock()
        mock_stats.remaining = 5
        mock_stats.reset_time = 60
        mock_farl.limiter.get_window_stats.return_value = mock_stats

        handler = PolicyHandler(
            request=mock_request,
            name="test",
            quota=10,
            quota_unit=None,
            time="minute",
            period=1,
            cost=1,
            partition="user123",
            namespace="default",
            state=farl_state,
            farl=mock_farl,
            error_class=None,
        )

        result = await handler()

        assert result is True
        assert len(farl_state["policy"]) == 1
        assert len(farl_state["state"]) == 1
        assert len(farl_state["violated"]) == 0

    @pytest.mark.anyio
    async def test_call_quota_exceeded_no_error(
        self, mock_request, farl_state, mock_farl
    ):
        mock_farl.limiter.hit.return_value = False

        mock_stats = Mock()
        mock_stats.remaining = 0
        mock_stats.reset_time = 60
        mock_farl.limiter.get_window_stats.return_value = mock_stats

        handler = PolicyHandler(
            request=mock_request,
            name="test",
            quota=10,
            quota_unit=None,
            time="minute",
            period=1,
            cost=1,
            partition="user123",
            namespace="default",
            state=farl_state,
            farl=mock_farl,
            error_class=None,
        )

        result = await handler()

        assert result is False
        assert len(farl_state["violated"]) == 1

    @pytest.mark.anyio
    async def test_call_quota_exceeded_with_error(
        self, mock_request, farl_state, mock_farl
    ):
        mock_farl.limiter.hit.return_value = False

        mock_stats = Mock()
        mock_stats.remaining = 0
        mock_stats.reset_time = 60
        mock_farl.limiter.get_window_stats.return_value = mock_stats

        handler = PolicyHandler(
            request=mock_request,
            name="test",
            quota=10,
            quota_unit=None,
            time="minute",
            period=1,
            cost=1,
            partition="user123",
            namespace="default",
            state=farl_state,
            farl=mock_farl,
            error_class=QuotaExceeded,
        )

        with pytest.raises(QuotaExceeded):
            await handler()

    @pytest.mark.anyio
    async def test_call_with_awaitable_results(
        self, mock_request, farl_state, mock_farl
    ):
        # Test async limiter methods
        async def mock_hit(*args, **kwargs):
            return True

        async def mock_stats(*args, **kwargs):
            stats = Mock()
            stats.remaining = 5
            stats.reset_time = 60
            return stats

        mock_farl.limiter.hit = mock_hit
        mock_farl.limiter.get_window_stats = mock_stats

        handler = PolicyHandler(
            request=mock_request,
            name="test",
            quota=10,
            quota_unit=None,
            time="minute",
            period=1,
            cost=1,
            partition="user123",
            namespace="default",
            state=farl_state,
            farl=mock_farl,
            error_class=None,
        )

        result = await handler()
        assert result is True


class TestMultiPartitionPolicyHandler:
    @pytest.fixture
    def mock_farl(self):
        farl = Mock(spec=Farl)
        farl.limiter = Mock()
        return farl

    @pytest.fixture
    def mock_request(self):
        return Mock(spec=Request)

    @pytest.fixture
    def farl_state(self):
        return FarlState(policy=[], state=[], violated=[])

    def test_get_policy_name_basic(self):
        limit = RateLimitItemPerMinute(10, 1)
        name = MultiPartitionPolicyHandler._get_policy_name(limit)
        assert name == "multiminute"

    def test_get_policy_name_with_multiples(self):
        limit = RateLimitItemPerMinute(10, 3)
        name = MultiPartitionPolicyHandler._get_policy_name(limit)
        assert name == "multi3minute"

    @pytest.mark.anyio
    async def test_call_all_partitions_success(
        self, mock_request, farl_state, mock_farl
    ):
        partition_costs = {"user1": 1, "user2": 2}

        mock_farl.limiter.hit.return_value = True

        mock_stats = Mock()
        mock_stats.remaining = 5
        mock_stats.reset_time = 60
        mock_farl.limiter.get_window_stats.return_value = mock_stats

        handler = MultiPartitionPolicyHandler(
            request=mock_request,
            name="multi_test",
            quota=10,
            quota_unit=None,
            time="minute",
            period=1,
            partition_costs=partition_costs,
            namespace="default",
            state=farl_state,
            farl=mock_farl,
            error_class=None,
        )

        result = await handler()

        assert result is True
        assert len(farl_state["policy"]) == 2  # One for each partition
        assert len(farl_state["state"]) == 2
        assert len(farl_state["violated"]) == 0

        # Verify hit was called for each partition with correct cost
        assert mock_farl.limiter.hit.call_count == 2

    @pytest.mark.anyio
    async def test_call_some_partitions_violated(
        self, mock_request, farl_state, mock_farl
    ):
        partition_costs = {"user1": 1, "user2": 2}

        # user1 succeeds, user2 fails
        hit_results = [True, False]
        mock_farl.limiter.hit.side_effect = hit_results

        mock_stats = Mock()
        mock_stats.remaining = 0
        mock_stats.reset_time = 60
        mock_farl.limiter.get_window_stats.return_value = mock_stats

        handler = MultiPartitionPolicyHandler(
            request=mock_request,
            name="multi_test",
            quota=10,
            quota_unit=None,
            time="minute",
            period=1,
            partition_costs=partition_costs,
            namespace="default",
            state=farl_state,
            farl=mock_farl,
            error_class=QuotaExceeded,
        )

        with pytest.raises(QuotaExceeded):
            await handler()

        assert len(farl_state["violated"]) == 1

    @pytest.mark.anyio
    async def test_call_with_awaitable_results(
        self, mock_request, farl_state, mock_farl
    ):
        partition_costs = {"user1": 1}

        async def mock_hit(*args, **kwargs):
            return True

        async def mock_stats(*args, **kwargs):
            stats = Mock()
            stats.remaining = 5
            stats.reset_time = 60
            return stats

        mock_farl.limiter.hit = mock_hit
        mock_farl.limiter.get_window_stats = mock_stats

        handler = MultiPartitionPolicyHandler(
            request=mock_request,
            name="multi_test",
            quota=10,
            quota_unit=None,
            time="minute",
            period=1,
            partition_costs=partition_costs,
            namespace="default",
            state=farl_state,
            farl=mock_farl,
            error_class=None,
        )

        result = await handler()
        assert result is True


class TestAdvancedPolicyScenarios:
    @pytest.fixture
    def mock_farl(self):
        farl = Mock(spec=Farl)
        farl.limiter = Mock()
        return farl

    @pytest.mark.anyio
    async def test_policy_handler_with_different_time_units(self, mock_farl):
        """Test PolicyHandler with various time units"""
        mock_request = Mock(spec=Request)
        farl_state = FarlState(policy=[], state=[], violated=[])

        time_units = ["second", "minute", "hour", "day"]

        for time_unit in time_units:
            handler = PolicyHandler(
                request=mock_request,
                name=f"test_{time_unit}",
                quota=10,
                quota_unit=None,
                time=time_unit,
                period=1,
                cost=1,
                partition=None,
                namespace="default",
                state=farl_state,
                farl=mock_farl,
                error_class=None,
            )

            limit = handler.get_limit_value()
            assert limit.amount == 10
            assert limit.multiples == 1

    @pytest.mark.anyio
    async def test_multi_partition_handler_edge_cases(self, mock_farl):
        """Test MultiPartitionPolicyHandler with edge cases"""
        mock_request = Mock(spec=Request)
        farl_state = FarlState(policy=[], state=[], violated=[])

        # Test with empty partition costs
        handler = MultiPartitionPolicyHandler(
            request=mock_request,
            name="empty_test",
            quota=10,
            quota_unit=None,
            time="minute",
            period=1,
            partition_costs={},
            namespace="default",
            state=farl_state,
            farl=mock_farl,
            error_class=None,
        )

        mock_farl.limiter.hit.return_value = True
        mock_stats = Mock()
        mock_stats.remaining = 5
        mock_stats.reset_time = 60
        mock_farl.limiter.get_window_stats.return_value = mock_stats

        result = await handler()
        assert result is True  # No partitions means success
        assert len(farl_state["policy"]) == 0
        assert len(farl_state["state"]) == 0

    def test_rate_limit_policy_manager_callable_dependencies(self):
        """Test that callable dependencies work correctly"""
        mock_farl = Mock(spec=Farl)

        def dynamic_quota():
            return 15

        def dynamic_namespace():
            return "dynamic"

        manager = RateLimitPolicyManager(farl=mock_farl)
        dependency = manager.create(
            quota=dynamic_quota, namespace=dynamic_namespace, time="hour"
        )

        assert callable(dependency)
        assert len(manager._dependencies) == 1


class TestIntegrationWithFastAPI:
    @pytest.fixture
    def app_with_dependencies(self):
        app = FastAPI()

        # Create a mock farl instance
        mock_farl = Mock(spec=Farl)
        mock_farl.limiter = Mock()

        manager = RateLimitPolicyManager(farl=mock_farl)
        rate_limit_dep = manager.create(quota=5, time="minute")

        @app.get("/limited")
        async def limited_endpoint(request: Request, _=Depends(rate_limit_dep)):
            return {"message": "success"}

        return app

    def test_dependency_injection_in_fastapi(self, app_with_dependencies):
        with TestClient(app_with_dependencies) as client:
            # The app was created with a mocked farl in the manager
            # So we just need to test that the endpoint works
            response = client.get("/limited")
            assert response.status_code == 200
            assert response.json() == {"message": "success"}


@pytest.mark.anyio
async def test_error_logging_on_violation():
    """Test that proper logging occurs when rate limits are violated"""
    from unittest.mock import patch

    mock_farl = Mock(spec=Farl)
    mock_farl.limiter = Mock()
    mock_farl.limiter.hit.return_value = False

    mock_stats = Mock()
    mock_stats.remaining = 0
    mock_stats.reset_time = 60
    mock_farl.limiter.get_window_stats.return_value = mock_stats

    mock_request = Mock(spec=Request)
    farl_state = FarlState(policy=[], state=[], violated=[])

    handler = PolicyHandler(
        request=mock_request,
        name="test_policy",
        quota=10,
        quota_unit=None,
        time="minute",
        period=1,
        cost=1,
        partition="user123",
        namespace="default",
        state=farl_state,
        farl=mock_farl,
        error_class=QuotaExceeded,
    )

    with patch("farl.dependencies.limits.logger") as mock_logger:
        with pytest.raises(QuotaExceeded):
            await handler()

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "Rate limit exceeded" in call_args[0][0]
        assert "test_policy" in call_args[0][1]
