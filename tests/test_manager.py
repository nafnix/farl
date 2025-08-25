from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from farl.constants import STATE_KEY
from farl.manager import RateLimitPolicyManager
from farl.types import AnyFarlProtocol, FarlState, PolicySettings


class TestRateLimitPolicyManager:
    def setup_method(self):
        self.mock_farl = MagicMock(spec=AnyFarlProtocol)
        self.mock_farl.limiter = MagicMock()
        self.manager = RateLimitPolicyManager(farl=self.mock_farl)

    def test_init_default(self):
        """Test initialization with default values"""
        manager = RateLimitPolicyManager()
        assert manager.error_class is not None  # Should default to QuotaExceeded
        assert manager.farl is None
        assert manager._dependencies == []

    def test_init_with_params(self):
        """Test initialization with custom parameters"""
        from farl.exceptions import TemporaryReducedCapacity

        manager = RateLimitPolicyManager(
            error_class=TemporaryReducedCapacity, farl=self.mock_farl
        )
        assert manager.error_class == TemporaryReducedCapacity
        assert manager.farl == self.mock_farl

    def test_get_value_static_method(self):
        """Test _get_value static method"""
        test_value = {"quota": 10, "time": "minute"}
        value_func = RateLimitPolicyManager._get_value(test_value)
        assert value_func() == test_value

    def test_get_farl_state_with_farl_in_manager(self):
        """Test _get_farl_state when farl is set in manager"""
        mock_request = MagicMock(spec=Request)
        mock_request.scope = {"state": {}}

        farl, state = self.manager._get_farl_state(mock_request)

        assert farl == self.mock_farl
        assert isinstance(state, dict)
        assert "policy" in state
        assert "state" in state
        assert "violated" in state

    def test_get_farl_state_with_farl_in_request_state(self):
        """Test _get_farl_state when farl is in request state"""
        mock_request = MagicMock(spec=Request)
        mock_state_farl = MagicMock(spec=AnyFarlProtocol)
        mock_request.scope = {
            "state": {
                STATE_KEY: FarlState(
                    farl=mock_state_farl, policy=[], state=[], violated=[]
                )
            }
        }

        # Create manager without farl
        manager = RateLimitPolicyManager()
        farl, state = manager._get_farl_state(mock_request)

        assert farl == mock_state_farl

    def test_get_farl_state_no_farl_raises_error(self):
        """Test _get_farl_state raises error when no farl found"""
        mock_request = MagicMock(spec=Request)
        mock_request.scope = {"state": {}}

        manager = RateLimitPolicyManager()  # No farl set

        with pytest.raises(ValueError, match="farl instance is required"):
            manager._get_farl_state(mock_request)

    def test_create_with_static_settings(self):
        """Test create method with static policy settings"""
        settings: PolicySettings = {
            "quota": 10,
            "time": "minute",
            "period": 1,
            "cost": 1,
            "namespace": "test",
        }

        dependency = self.manager.create(settings)

        assert callable(dependency)
        assert len(self.manager._dependencies) == 1

    def test_create_with_callable_settings(self):
        """Test create method with callable settings"""

        def get_settings() -> PolicySettings:
            return {"quota": 5, "time": "hour"}

        dependency = self.manager.create(get_settings)

        assert callable(dependency)
        assert len(self.manager._dependencies) == 1

    @pytest.mark.anyio
    async def test_create_dependency_execution_success(self):
        """Test created dependency execution on success"""
        settings: PolicySettings = {
            "quota": 10,
            "time": "minute",
            "period": 1,
            "cost": 1,
            "namespace": "test",
        }

        mock_handler_instance = MagicMock()
        mock_handler_instance.return_value = True
        mock_request = MagicMock(spec=Request)
        mock_request.scope = {"state": {}}
        # Mock successful handler execution
        with patch("farl.manager.PolicySettingsHandler") as mock_hdlr:
            mock_hdlr.return_value = mock_handler_instance

            dependency = self.manager.create(settings, mock_hdlr)
            result = await dependency(mock_request, settings)

            assert result is True
            mock_hdlr.assert_called_once()

    @pytest.mark.anyio
    async def test_create_dependency_execution_async(self):
        """Test created dependency execution with async handler"""
        settings: PolicySettings = {"quota": 10, "time": "minute"}

        mock_handler_instance = AsyncMock(return_value=True)

        mock_request = MagicMock(spec=Request)
        mock_request.scope = {"state": {}}
        # Mock async handler execution
        with patch("farl.manager.PolicySettingsHandler") as mock_hdlr:
            mock_hdlr.return_value = mock_handler_instance

            dependency = self.manager.create(settings, mock_hdlr)

            result = await dependency(mock_request, settings)

            assert result is True

    def test_call_as_decorator(self):
        """Test __call__ method as decorator"""

        def test_function():
            return "test"

        # Mock the rate_limit decorator
        with patch("farl.manager.rate_limit") as mock_rate_limit:
            mock_decorator = MagicMock()
            mock_rate_limit.return_value = mock_decorator

            result = self.manager(test_function)

            mock_rate_limit.assert_called_once_with(self.manager)
            mock_decorator.assert_called_once_with(test_function)
            assert result == mock_decorator.return_value

    def test_call_create_dependency(self):
        """Test __call__ method to create dependency"""
        # First create some dependencies
        settings1: PolicySettings = {"quota": 10, "time": "minute"}
        settings2: PolicySettings = {"quota": 20, "time": "hour"}

        self.manager.create(settings1)
        self.manager.create(settings2)

        dependency = self.manager()

        assert callable(dependency)

        # Check signature has the expected parameters
        import inspect

        sig = inspect.signature(dependency)
        params = list(sig.parameters.keys())

        assert "request" in params
        assert "_ratelimit_0" in params
        assert "_ratelimit_1" in params

    def test_call_with_violations_raises_error(self):
        """Test __call__ created dependency raises error when violations exist"""
        from farl.exceptions import QuotaExceeded

        # Create manager with error class
        manager = RateLimitPolicyManager(error_class=QuotaExceeded, farl=self.mock_farl)

        dependency = manager()

        mock_request = MagicMock(spec=Request)
        mock_request.scope = {
            "state": {
                STATE_KEY: FarlState(
                    policy=[],
                    state=[],
                    violated=[MagicMock(policy="policy1"), MagicMock(policy="policy2")],
                )
            }
        }

        with pytest.raises(QuotaExceeded) as exc_info:
            dependency(mock_request)

        assert exc_info.value.data.model_dump()["violated-policies"] == [
            "policy1",
            "policy2",
        ]

    def test_call_with_violations_no_error_class(self):
        """Test __call__ created dependency with no violations"""
        manager = RateLimitPolicyManager(error_class=None, farl=self.mock_farl)

        dependency = manager()

        mock_request = MagicMock(spec=Request)
        mock_request.scope = {
            "state": {
                STATE_KEY: FarlState(
                    policy=[], state=[], violated=[MagicMock(policy="policy1")]
                )
            }
        }

        # Should not raise error when error_class is None
        dependency(mock_request)

    def test_call_with_no_violations(self):
        """Test __call__ created dependency with no violations"""
        dependency = self.manager()

        mock_request = MagicMock(spec=Request)
        mock_request.scope = {
            "state": {STATE_KEY: FarlState(policy=[], state=[], violated=[])}
        }

        # Should not raise error when no violations
        dependency(mock_request)

    def test_multiple_create_calls(self):
        """Test multiple calls to create method"""
        settings1: PolicySettings = {"quota": 10, "time": "minute"}
        settings2: PolicySettings = {"quota": 20, "time": "hour"}

        dep1 = self.manager.create(settings1)
        dep2 = self.manager.create(settings2)

        assert len(self.manager._dependencies) == 2
        assert dep1 != dep2
        assert callable(dep1)
        assert callable(dep2)
