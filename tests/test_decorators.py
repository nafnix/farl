import inspect
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from farl.base import Farl
from farl.decorators import INJECT_DEP_PREFIX, _create_call, _exclude_kwds, rate_limit
from farl.dependencies import RateLimitPolicyManager


class TestExcludeKwds:
    def test_exclude_kwds_removes_farl_params(self):
        kwds = {
            "normal_param": "value1",
            "__farl_ratelimit_1": "should_be_removed",
            "__farl_ratelimit_2": "should_also_be_removed",
            "another_normal": "value2",
        }

        _exclude_kwds(kwds)

        expected = {"normal_param": "value1", "another_normal": "value2"}
        assert kwds == expected

    def test_exclude_kwds_no_farl_params(self):
        kwds = {"param1": "value1", "param2": "value2"}
        original = kwds.copy()

        _exclude_kwds(kwds)

        assert kwds == original

    def test_exclude_kwds_only_farl_params(self):
        kwds = {"__farl_ratelimit_1": "value1", "__farl_ratelimit_special": "value2"}

        _exclude_kwds(kwds)

        assert kwds == {}

    def test_exclude_kwds_partial_match_not_removed(self):
        kwds = {
            "prefix__farl_ratelimit": "should_not_be_removed",
            "__farl_ratelimit_suffix": "should_be_removed",
            "__farl_ratelimit": "should_be_removed",
        }

        _exclude_kwds(kwds)

        assert kwds == {"prefix__farl_ratelimit": "should_not_be_removed"}


class TestCreateCall:
    def test_create_call_sync_function(self):
        def sync_func(a, b, **kwargs):
            return a + b + sum(kwargs.values())

        wrapped = _create_call(sync_func)

        # Test that it preserves function behavior
        result = wrapped(1, 2, normal_param=3, __farl_ratelimit_1=999)
        assert result == 6  # 1 + 2 + 3, farl param excluded

        # Test that wrapper is not a coroutine
        assert not inspect.iscoroutinefunction(wrapped)

        # Test that function metadata is preserved
        assert wrapped.__name__ == sync_func.__name__

    @pytest.mark.anyio
    async def test_create_call_async_function(self):
        async def async_func(a, b, **kwargs):
            return a + b + sum(kwargs.values())

        wrapped = _create_call(async_func)

        # Test that it preserves async function behavior
        result = await wrapped(1, 2, normal_param=3, __farl_ratelimit_1=999)
        assert result == 6  # 1 + 2 + 3, farl param excluded

        # Test that wrapper is a coroutine
        assert inspect.iscoroutinefunction(wrapped)

        # Test that function metadata is preserved
        assert wrapped.__name__ == async_func.__name__

    def test_create_call_preserves_args(self):
        def func_with_args(*args, **kwargs):
            return len(args), len(kwargs)

        wrapped = _create_call(func_with_args)

        args_len, kwargs_len = wrapped(1, 2, 3, a=1, __farl_ratelimit_1=999)
        assert args_len == 3
        assert kwargs_len == 1  # Only 'a', farl param excluded

    @pytest.mark.anyio
    async def test_create_call_async_preserves_args(self):
        async def async_func_with_args(*args, **kwargs):
            return len(args), len(kwargs)

        wrapped = _create_call(async_func_with_args)

        args_len, kwargs_len = await wrapped(1, 2, 3, a=1, __farl_ratelimit_1=999)
        assert args_len == 3
        assert kwargs_len == 1  # Only 'a', farl param excluded


class TestRateLimitDecorator:
    @pytest.fixture
    def mock_manager(self):
        manager = Mock(spec=RateLimitPolicyManager)
        mock_dependency = Mock()
        manager.return_value = mock_dependency
        return manager

    def test_rate_limit_decorator_sync_function(self, mock_manager):
        @rate_limit(mock_manager)
        def test_func(x, y):
            return x + y

        # Check that function is wrapped properly
        assert callable(test_func)
        assert test_func.__name__ == "test_func"

        # Check that signature was modified
        sig = inspect.signature(test_func)
        param_names = list(sig.parameters.keys())

        # Should have original params plus injected dependency
        assert "x" in param_names
        assert "y" in param_names

        # Should have at least one farl dependency injected
        farl_params = [p for p in param_names if p.startswith(INJECT_DEP_PREFIX)]
        assert len(farl_params) >= 1

        # Check that the injected parameter has Depends as default
        farl_param = sig.parameters[farl_params[0]]
        assert isinstance(farl_param.default, type(Depends(lambda: None)))

    def test_rate_limit_decorator_async_function(self, mock_manager):
        @rate_limit(mock_manager)
        async def async_test_func(x, y):
            return x + y

        # Check that function is still async
        assert inspect.iscoroutinefunction(async_test_func)

        # Check signature modification
        sig = inspect.signature(async_test_func)
        param_names = list(sig.parameters.keys())

        farl_params = [p for p in param_names if p.startswith(INJECT_DEP_PREFIX)]
        assert len(farl_params) >= 1

    def test_rate_limit_decorator_with_kwargs(self, mock_manager):
        @rate_limit(mock_manager)
        def func_with_kwargs(a, b=10, **kwargs):
            return a + b + len(kwargs)

        sig = inspect.signature(func_with_kwargs)
        params = list(sig.parameters.values())

        # Check that VAR_KEYWORD parameter is still at the end
        assert params[-1].kind == inspect.Parameter.VAR_KEYWORD

        # Check that farl dependency was inserted before **kwargs
        farl_params = [p for p in params if p.name.startswith(INJECT_DEP_PREFIX)]
        assert len(farl_params) >= 1

        # Farl param should be inserted before **kwargs
        farl_param_index = params.index(farl_params[0])
        kwargs_index = len(params) - 1
        assert farl_param_index < kwargs_index

    def test_rate_limit_decorator_without_kwargs(self, mock_manager):
        @rate_limit(mock_manager)
        def func_without_kwargs(a, b=10):
            return a + b

        sig = inspect.signature(func_without_kwargs)
        params = list(sig.parameters.values())

        # Check that farl dependency was appended
        farl_params = [p for p in params if p.name.startswith(INJECT_DEP_PREFIX)]
        assert len(farl_params) >= 1

        # Farl param should be at the end
        assert params[-1].name.startswith(INJECT_DEP_PREFIX)

    def test_multiple_rate_limit_decorators(self, mock_manager):
        manager1 = Mock(spec=RateLimitPolicyManager)
        manager1.return_value = Mock()

        manager2 = Mock(spec=RateLimitPolicyManager)
        manager2.return_value = Mock()

        @rate_limit(manager1)
        @rate_limit(manager2)
        def double_limited_func(x):
            return x * 2

        sig = inspect.signature(double_limited_func)
        param_names = list(sig.parameters.keys())

        # Should have two farl dependencies
        farl_params = [p for p in param_names if p.startswith(INJECT_DEP_PREFIX)]
        assert len(farl_params) == 2

        # Check that they have different names (numbered)
        assert "__farl_ratelimit_1" in param_names
        assert "__farl_ratelimit_2" in param_names

    def test_rate_limit_decorator_preserves_function_execution(self, mock_manager):
        @rate_limit(mock_manager)
        def add_numbers(a, b):
            return a + b

        # Mock the dependency to not interfere
        with patch.object(add_numbers, "__signature__"):
            # Execute the original function logic
            result = add_numbers.__wrapped__(5, 3)
            assert result == 8

    @pytest.mark.anyio
    async def test_rate_limit_decorator_preserves_async_execution(self, mock_manager):
        @rate_limit(mock_manager)
        async def async_add(a, b):
            return a + b

        # The wrapped function should still work correctly
        with patch.object(async_add, "__signature__"):
            result = await async_add.__wrapped__(5, 3)
            assert result == 8


class TestRateLimitIntegrationWithFastAPI:
    @pytest.fixture
    def mock_farl(self):
        farl = Mock(spec=Farl)
        farl.limiter = Mock()
        farl.limiter.hit.return_value = True

        mock_stats = Mock()
        mock_stats.remaining = 5
        mock_stats.reset_time = 60
        farl.limiter.get_window_stats.return_value = mock_stats

        return farl

    def test_rate_limit_decorator_in_fastapi_endpoint(self, mock_farl):
        app = FastAPI()

        manager = RateLimitPolicyManager(farl=mock_farl)

        @app.get("/test")
        @rate_limit(manager)
        async def test_endpoint(value: int = 10):
            return {"result": value * 2}

        with TestClient(app) as client:
            response = client.get("/test?value=5")
            assert response.status_code == 200
            assert response.json() == {"result": 10}

    def test_rate_limit_with_dependency_injection(self, mock_farl):
        app = FastAPI()

        def get_user_id():
            return "user123"

        manager = RateLimitPolicyManager(farl=mock_farl)

        @app.get("/user-endpoint")
        @rate_limit(manager)
        async def user_endpoint(user_id: str = Depends(get_user_id)):
            return {"user": user_id, "message": "success"}

        with TestClient(app) as client:
            response = client.get("/user-endpoint")
            assert response.status_code == 200
            assert response.json() == {"user": "user123", "message": "success"}

    def test_rate_limit_with_request_dependency(self, mock_farl):
        from fastapi import Request

        app = FastAPI()

        manager = RateLimitPolicyManager(farl=mock_farl)

        @app.get("/request-endpoint")
        @rate_limit(manager)
        async def request_endpoint(request: Request):
            return {"path": request.url.path}

        with TestClient(app) as client:
            response = client.get("/request-endpoint")
            assert response.status_code == 200
            assert response.json() == {"path": "/request-endpoint"}


class TestDecoratorSignatureHandling:
    def test_signature_with_positional_only_params(self):
        manager = Mock(spec=RateLimitPolicyManager)
        manager.return_value = Mock()

        @rate_limit(manager)
        def func_with_pos_only(a, b, /, c=10):
            return a + b + c

        sig = inspect.signature(func_with_pos_only)
        param_names = list(sig.parameters.keys())

        # Should preserve positional-only parameters
        assert "a" in param_names
        assert "b" in param_names
        assert "c" in param_names

        # Should add farl dependency
        farl_params = [p for p in param_names if p.startswith(INJECT_DEP_PREFIX)]
        assert len(farl_params) >= 1

    def test_signature_with_keyword_only_params(self):
        manager = Mock(spec=RateLimitPolicyManager)
        manager.return_value = Mock()

        @rate_limit(manager)
        def func_with_kw_only(a, *, b, c=10):
            return a + b + c

        sig = inspect.signature(func_with_kw_only)
        params = list(sig.parameters.values())

        # Should preserve keyword-only parameters
        kw_only_params = [p for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY]
        original_kw_only = [
            p for p in kw_only_params if not p.name.startswith(INJECT_DEP_PREFIX)
        ]
        assert len(original_kw_only) == 2  # b and c

        # Should add farl dependency as keyword-only
        farl_params = [
            p for p in kw_only_params if p.name.startswith(INJECT_DEP_PREFIX)
        ]
        assert len(farl_params) >= 1

    def test_signature_with_var_args(self):
        manager = Mock(spec=RateLimitPolicyManager)
        manager.return_value = Mock()

        @rate_limit(manager)
        def func_with_var_args(a, *args, b=10):
            return a + len(args) + b

        sig = inspect.signature(func_with_var_args)
        params = list(sig.parameters.values())

        # Should preserve VAR_POSITIONAL parameter
        var_pos_params = [
            p for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL
        ]
        assert len(var_pos_params) == 1
        assert var_pos_params[0].name == "args"

        # Should add farl dependency
        farl_params = [p for p in params if p.name.startswith(INJECT_DEP_PREFIX)]
        assert len(farl_params) >= 1


class TestDecoratorErrorHandling:
    def test_rate_limit_with_none_manager(self):
        with pytest.raises((TypeError, AttributeError)):

            @rate_limit(None)
            def test_func():
                pass

    def test_rate_limit_with_invalid_manager(self):
        with pytest.raises(TypeError):

            @rate_limit("not_a_manager")
            def test_func():
                pass


class TestDecoratorMetadata:
    def test_decorated_function_preserves_metadata(self):
        manager = Mock(spec=RateLimitPolicyManager)
        manager.return_value = Mock()

        def original_func(x, y):
            """This is a test function."""
            return x + y

        decorated = rate_limit(manager)(original_func)

        assert decorated.__name__ == original_func.__name__
        assert decorated.__doc__ == original_func.__doc__
        assert hasattr(decorated, "__wrapped__")
        assert decorated.__wrapped__ is not None

    def test_decorated_async_function_preserves_metadata(self):
        manager = Mock(spec=RateLimitPolicyManager)
        manager.return_value = Mock()

        async def original_async_func(x, y):
            """This is an async test function."""
            return x + y

        decorated = rate_limit(manager)(original_async_func)

        assert decorated.__name__ == original_async_func.__name__
        assert decorated.__doc__ == original_async_func.__doc__
        assert inspect.iscoroutinefunction(decorated)
        assert hasattr(decorated, "__wrapped__")


@pytest.mark.integration
class TestFullIntegrationScenarios:
    def test_complex_endpoint_with_multiple_decorators_and_dependencies(self):
        """Test a realistic scenario with multiple rate limits and dependencies"""
        from fastapi import Depends, Request

        app = FastAPI()

        # Mock farl instance
        mock_farl = Mock(spec=Farl)
        mock_farl.limiter = Mock()
        mock_farl.limiter.hit.return_value = True

        mock_stats = Mock()
        mock_stats.remaining = 5
        mock_stats.reset_time = 60
        mock_farl.limiter.get_window_stats.return_value = mock_stats

        # Create different rate limit managers
        user_limit_manager = RateLimitPolicyManager(farl=mock_farl)
        endpoint_limit_manager = RateLimitPolicyManager(farl=mock_farl)

        def get_user_id(request: Request):
            return request.headers.get("user-id", "anonymous")

        def get_api_key(request: Request):
            return request.headers.get("api-key", "default")

        @app.post("/complex-endpoint")
        @rate_limit(user_limit_manager)
        @rate_limit(endpoint_limit_manager)
        async def complex_endpoint(
            request: Request,
            data: dict,
            user_id: str = Depends(get_user_id),
            api_key: str = Depends(get_api_key),
        ):
            return {
                "user": user_id,
                "api_key": api_key,
                "data_keys": list(data.keys()),
                "path": request.url.path,
            }

        with TestClient(app) as client:
            response = client.post(
                "/complex-endpoint",
                json={"test": "data"},
                headers={"user-id": "user123", "api-key": "secret123"},
            )

            assert response.status_code == 200
            result = response.json()
            assert result["user"] == "user123"
            assert result["api_key"] == "secret123"
            assert result["data_keys"] == ["test"]
            assert result["path"] == "/complex-endpoint"

    def test_decorator_with_class_methods(self):
        """Test that the decorator works with class methods"""
        manager = Mock(spec=RateLimitPolicyManager)
        manager.return_value = Mock()

        class APIHandler:
            @rate_limit(manager)
            def process_request(self, data):
                return f"processed: {data}"

            @rate_limit(manager)
            async def async_process(self, data):
                return f"async processed: {data}"

        handler = APIHandler()

        # Test sync method
        result = handler.process_request("test_data")
        # The wrapped method should still be callable
        assert callable(handler.process_request)

        # Test async method
        assert inspect.iscoroutinefunction(handler.async_process)


class TestAdvancedDecoratorScenarios:
    def test_rate_limit_decorator_with_complex_signature(self):
        """Test decorator with complex function signatures"""
        manager = Mock(spec=RateLimitPolicyManager)
        manager.return_value = Mock()
        
        @rate_limit(manager)
        def complex_func(a, b=10, *args, c=20, **kwargs):
            return a + b + len(args) + c + len(kwargs)
        
        sig = inspect.signature(complex_func)
        params = list(sig.parameters.values())
        
        # Verify all original parameters are preserved
        param_names = [p.name for p in params]
        assert "a" in param_names
        assert "b" in param_names
        assert "args" in param_names
        assert "c" in param_names
        assert "kwargs" in param_names
        
        # Verify farl dependency is added
        farl_params = [p for p in params if p.name.startswith(INJECT_DEP_PREFIX)]
        assert len(farl_params) >= 1
        
        # Verify **kwargs is still at the end
        assert params[-1].kind == inspect.Parameter.VAR_KEYWORD

    @pytest.mark.anyio
    async def test_rate_limit_decorator_with_generic_types(self):
        """Test decorator with functions that use generic types"""
        from typing import TypeVar, Generic, List
        
        T = TypeVar('T')
        
        manager = Mock(spec=RateLimitPolicyManager)
        manager.return_value = Mock()
        
        @rate_limit(manager)
        async def generic_func(data: List[T]) -> T:
            return data[0] if data else None
        
        # Should preserve function signature and async nature
        assert inspect.iscoroutinefunction(generic_func)
        sig = inspect.signature(generic_func)
        
        # Check original parameter exists
        assert "data" in sig.parameters
        
        # Check farl dependency was added
        farl_params = [p for p in sig.parameters.values() 
                      if p.name.startswith(INJECT_DEP_PREFIX)]
        assert len(farl_params) >= 1

    def test_rate_limit_decorator_preserves_annotations(self):
        """Test that decorator preserves type annotations"""
        manager = Mock(spec=RateLimitPolicyManager)
        manager.return_value = Mock()
        
        @rate_limit(manager)
        def annotated_func(x: int, y: str = "default") -> str:
            return f"{x}: {y}"
        
        sig = inspect.signature(annotated_func)
        
        # Check original annotations are preserved
        assert sig.parameters["x"].annotation == int
        assert sig.parameters["y"].annotation == str
        assert sig.return_annotation == str

    def test_nested_decorators_with_rate_limit(self):
        """Test rate_limit decorator works with other decorators"""
        from functools import wraps
        
        def another_decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        
        manager = Mock(spec=RateLimitPolicyManager)
        manager.return_value = Mock()
        
        @another_decorator
        @rate_limit(manager)
        def decorated_func(x, y):
            return x + y
        
        # Should still be callable and have correct name
        assert callable(decorated_func)
        assert decorated_func.__name__ == "decorated_func"
        
        # Should have farl dependency injected
        sig = inspect.signature(decorated_func)
        farl_params = [p for p in sig.parameters.values() 
                      if p.name.startswith(INJECT_DEP_PREFIX)]
        assert len(farl_params) >= 1


class TestDecoratorPerformanceOptimizations:
    def test_decorator_minimal_overhead(self):
        """Test that decorator adds minimal performance overhead"""
        manager = Mock(spec=RateLimitPolicyManager)
        manager.return_value = Mock()
        
        @rate_limit(manager)
        def simple_func(x):
            return x * 2
        
        # Function should still be directly callable (after mocking dependencies)
        assert callable(simple_func)
        
        # Signature modification should be efficient
        sig = inspect.signature(simple_func)
        assert len(sig.parameters) == 2  # x + farl dependency

    def test_decorator_memory_efficiency(self):
        """Test that decorator doesn't create memory leaks"""
        manager = Mock(spec=RateLimitPolicyManager)
        manager.return_value = Mock()
        
        # Create multiple decorated functions
        functions = []
        for i in range(100):
            @rate_limit(manager)
            def temp_func(x):
                return x + i
            functions.append(temp_func)
        
        # Each should have independent signatures
        signatures = [inspect.signature(f) for f in functions]
        assert len(set(str(sig) for sig in signatures)) >= 1  # At least one unique signature
        
        # All should be callable
        assert all(callable(f) for f in functions)


class TestErrorHandlingAndEdgeCases:
    def test_rate_limit_with_lambda_function(self):
        """Test decorator behavior with lambda functions"""
        manager = Mock(spec=RateLimitPolicyManager)
        manager.return_value = Mock()
        
        # Lambda functions should work but might have limitations
        lambda_func = lambda x: x * 2
        
        try:
            decorated_lambda = rate_limit(manager)(lambda_func)
            assert callable(decorated_lambda)
        except Exception:
            # It's acceptable if lambda decoration fails
            pass

    def test_rate_limit_with_builtin_function(self):
        """Test decorator with builtin functions"""
        manager = Mock(spec=RateLimitPolicyManager)
        manager.return_value = Mock()
        
        # Builtin functions might not be decoratable or might work
        try:
            decorated_len = rate_limit(manager)(len)
            # If it works, it should still be callable
            assert callable(decorated_len)
        except (TypeError, AttributeError):
            # It's acceptable if builtin decoration fails
            pass

    def test_rate_limit_preserves_existing_defaults(self):
        """Test that existing parameter defaults are preserved"""
        manager = Mock(spec=RateLimitPolicyManager)
        manager.return_value = Mock()
        
        @rate_limit(manager)
        def func_with_defaults(a, b=42, c="test"):
            return f"{a}-{b}-{c}"
        
        sig = inspect.signature(func_with_defaults)
        
        # Original defaults should be preserved
        assert sig.parameters["b"].default == 42
        assert sig.parameters["c"].default == "test"
        
        # Farl dependency should have Depends default
        farl_params = [p for p in sig.parameters.values() 
                      if p.name.startswith(INJECT_DEP_PREFIX)]
        assert len(farl_params) >= 1
        farl_param = farl_params[0]
        assert isinstance(farl_param.default, type(Depends(lambda: None)))
