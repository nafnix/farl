import inspect
from unittest.mock import Mock

import fastapi.params
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from farl.decorators import (
    INJECT_DEP_PREFIX,
    _create_call,
    _exclude_kwds,
    rate_limit,
)


class TestExcludeKwds:
    def test_exclude_kwds_removes_farl_prefixed_keys(self):
        kwds = {
            "normal_param": "value1",
            "__farl_ratelimit_1": "farl_value1",
            "__farl_ratelimit_2": "farl_value2",
            "another_param": "value2",
        }
        _exclude_kwds(kwds)

        assert kwds == {
            "normal_param": "value1",
            "another_param": "value2",
        }

    def test_exclude_kwds_with_no_farl_keys(self):
        kwds = {"param1": "value1", "param2": "value2"}
        original = kwds.copy()
        _exclude_kwds(kwds)

        assert kwds == original

    def test_exclude_kwds_with_empty_dict(self):
        kwds = {}
        _exclude_kwds(kwds)

        assert kwds == {}


class TestCreateCall:
    def test_create_call_with_sync_function(self):
        def sync_fn(x, y, **kwargs):
            return x + y, kwargs

        wrapped_fn = _create_call(sync_fn)

        # Test that it excludes farl keywords
        result = wrapped_fn(1, 2, normal_param="test", __farl_ratelimit_1="farl")

        assert result == (3, {"normal_param": "test"})
        assert inspect.iscoroutinefunction(wrapped_fn) is False

    def test_create_call_with_async_function(self):
        async def async_fn(x, y, **kwargs):
            return x + y, kwargs

        wrapped_fn = _create_call(async_fn)

        assert inspect.iscoroutinefunction(wrapped_fn) is True

    @pytest.mark.anyio
    async def test_create_call_async_function_execution(self):
        async def async_fn(x, y, **kwargs):
            return x + y, kwargs

        wrapped_fn = _create_call(async_fn)

        # Test that it excludes farl keywords
        result = await wrapped_fn(1, 2, normal_param="test", __farl_ratelimit_1="farl")

        assert result == (3, {"normal_param": "test"})

    def test_create_call_preserves_function_metadata(self):
        def original_fn(x, y):
            """Test function"""
            return x + y

        wrapped_fn = _create_call(original_fn)

        assert wrapped_fn.__name__ == "original_fn"
        assert wrapped_fn.__doc__ == "Test function"


class TestRateLimit:
    def test_rate_limit_decorator_adds_dependency_parameter(self):
        # Mock manager
        mock_manager = Mock()
        mock_dependency = Mock()
        mock_manager.return_value = mock_dependency

        def test_fn(x: int, y: int):
            return x + y

        decorated_fn = rate_limit(mock_manager)(test_fn)

        # Check that dependency was created
        mock_manager.assert_called_once()

        # Check that signature was modified
        signature = inspect.signature(decorated_fn)
        params = list(signature.parameters.values())

        # Should have added a farl dependency parameter
        farl_params = [p for p in params if p.name.startswith(INJECT_DEP_PREFIX)]
        assert len(farl_params) == 1

        farl_param = farl_params[0]
        assert farl_param.kind == inspect.Parameter.KEYWORD_ONLY
        assert isinstance(farl_param.default, fastapi.params.Depends)

    def test_rate_limit_with_var_keyword_params(self):
        mock_manager = Mock()
        mock_dependency = Mock()
        mock_manager.return_value = mock_dependency

        def test_fn(x: int, **kwargs):
            return x, kwargs

        decorated_fn = rate_limit(mock_manager)(test_fn)
        signature = inspect.signature(decorated_fn)
        params = list(signature.parameters.values())

        # The farl dependency should be inserted before **kwargs
        assert params[-1].name == "kwargs"  # **kwargs should still be last
        assert params[-1].kind == inspect.Parameter.VAR_KEYWORD

        # Find the farl parameter (should be second to last)
        farl_param = params[-2]
        assert farl_param.name.startswith(INJECT_DEP_PREFIX)

    def test_rate_limit_multiple_decorators(self):
        mock_manager1 = Mock()
        mock_manager2 = Mock()
        mock_dependency1 = Mock()
        mock_dependency2 = Mock()
        mock_manager1.return_value = mock_dependency1
        mock_manager2.return_value = mock_dependency2

        def test_fn(x: int):
            return x

        # Apply multiple rate limit decorators
        decorated_fn = rate_limit(mock_manager1)(test_fn)
        double_decorated_fn = rate_limit(mock_manager2)(decorated_fn)

        signature = inspect.signature(double_decorated_fn)
        params = list(signature.parameters.values())

        # Should have two farl dependency parameters
        farl_params = [p for p in params if p.name.startswith(INJECT_DEP_PREFIX)]
        assert len(farl_params) == 2

        # Parameters should be numbered sequentially
        assert any("_1" in p.name for p in farl_params)
        assert any("_2" in p.name for p in farl_params)

    def test_rate_limit_integration_with_fastapi(self):
        """Test that the decorator works with FastAPI"""

        app = FastAPI()

        # Create a mock manager for testing
        mock_hdlr = Mock()
        mock_manager = Mock(return_value=lambda: mock_hdlr())

        @rate_limit(mock_manager)
        def test_endpoint(value: int):
            return {"value": value}

        app.get("/test")(test_endpoint)

        with TestClient(app) as client:
            response = client.get("/test?value=42")

            # The endpoint should work normally
            assert response.status_code == 200
            assert response.json() == {"value": 42}
            mock_hdlr.assert_called_once()
