from unittest.mock import patch

import pytest
from pydantic import RedisDsn

from farl.base import AsyncFarl, Farl


class TestFarlInitialization:
    """Test various initialization scenarios for the Farl class."""

    def test_farl_default_initialization(self):
        """Test Farl with default memory storage."""
        farl = Farl()

        assert farl.key is None
        assert farl.cost is None
        assert farl.policy is None
        assert farl.namespace is None
        assert hasattr(farl, "limiter")
        # Should use memory storage by default
        assert farl.limiter.storage.__class__.__name__ == "MemoryStorage"

    def test_farl_with_namespace(self):
        """Test Farl initialization with namespace."""
        namespace = "test_namespace"
        farl = Farl(namespace=namespace)

        assert farl.namespace == namespace

    def test_farl_with_custom_attributes(self):
        """Test Farl initialization with custom key, cost, and policy functions."""

        def custom_key():
            return "custom_key"

        def custom_cost():
            return 5

        def custom_policy():
            return {"amount": 10, "time": "minute"}

        farl = Farl(
            rate_limit_key=custom_key,
            rate_limit_cost=custom_cost,
            rate_limit_policy=custom_policy,
        )

        assert farl.key == custom_key
        assert farl.cost == custom_cost
        assert farl.policy == custom_policy

    @pytest.mark.parametrize(
        "strategy", ["fixed-window", "moving-window", "sliding-window-counter"]
    )
    def test_farl_strategies(self, strategy):
        """Test Farl initialization with different rate limiting strategies."""
        farl = Farl(strategy=strategy)

        # Map strategy names to expected limiter class names
        expected_limiter_names = {
            "fixed-window": "FixedWindowRateLimiter",
            "moving-window": "MovingWindowRateLimiter",
            "sliding-window-counter": "SlidingWindowCounterRateLimiter",
        }

        assert farl.limiter.__class__.__name__ == expected_limiter_names[strategy]

    def test_farl_invalid_strategy(self):
        """Test Farl initialization with invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported strategy: invalid_strategy"):
            Farl(strategy="invalid_strategy")


class TestAsyncFarlInitialization:
    """Test various initialization scenarios for the AsyncFarl class."""

    def test_async_farl_default_initialization(self):
        """Test AsyncFarl with default memory storage."""
        farl = AsyncFarl()

        assert farl.key is None
        assert farl.cost is None
        assert farl.policy is None
        assert farl.namespace is None
        assert hasattr(farl, "limiter")
        # Should use async memory storage by default
        assert farl.limiter.storage.__class__.__name__ == "MemoryStorage"

    def test_async_farl_with_namespace(self):
        """Test AsyncFarl initialization with namespace."""
        namespace = "async_test_namespace"
        farl = AsyncFarl(namespace=namespace)

        assert farl.namespace == namespace

    def test_async_farl_with_custom_attributes(self):
        """Test AsyncFarl initialization with custom key, cost, and policy functions."""

        def custom_key():
            return "async_custom_key"

        def custom_cost():
            return 3

        def custom_policy():
            return {"amount": 5, "time": "hour"}

        farl = AsyncFarl(
            rate_limit_key=custom_key,
            rate_limit_cost=custom_cost,
            rate_limit_policy=custom_policy,
        )

        assert farl.key == custom_key
        assert farl.cost == custom_cost
        assert farl.policy == custom_policy

    @pytest.mark.parametrize(
        "strategy", ["fixed-window", "moving-window", "sliding-window-counter"]
    )
    def test_async_farl_strategies(self, strategy):
        """Test AsyncFarl initialization with different rate limiting strategies."""
        farl = AsyncFarl(strategy=strategy)

        # Map strategy names to expected async limiter class names
        expected_limiter_names = {
            "fixed-window": "FixedWindowRateLimiter",
            "moving-window": "MovingWindowRateLimiter",
            "sliding-window-counter": "SlidingWindowCounterRateLimiter",
        }

        assert farl.limiter.__class__.__name__ == expected_limiter_names[strategy]

    def test_async_farl_invalid_strategy(self):
        """Test AsyncFarl initialization with invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported strategy: bad_strategy"):
            AsyncFarl(strategy="bad_strategy")


class TestImportErrorHandling:
    """Test behavior when Redis is not available."""

    @patch("farl.base.RedisConnectionPool", None)
    @patch("farl.base.AsyncRedisConnectionPool", None)
    def test_redis_not_available(self):
        """Test that classes can still be instantiated when Redis is not available."""
        # These should work fine with memory storage
        farl = Farl()
        async_farl = AsyncFarl()

        assert farl is not None
        assert async_farl is not None
        assert farl.limiter.storage.__class__.__name__ == "MemoryStorage"
        assert async_farl.limiter.storage.__class__.__name__ == "MemoryStorage"
