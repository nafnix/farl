import pytest

from farl.base import AsyncFarl, Farl


class TestFarl:
    def test_farl_init_memory_storage(self):
        """Test Farl initialization with memory storage (default)"""
        farl = Farl()

        assert hasattr(farl, "limiter")
        assert farl.limiter is not None
        # Should use memory storage by default

    def test_farl_init_fixed_window_strategy(self):
        """Test Farl with fixed-window strategy"""
        from limits.strategies import FixedWindowRateLimiter

        farl = Farl(strategy="fixed-window")

        assert hasattr(farl, "limiter")
        # Check that it's using the correct strategy class name
        assert isinstance(farl.limiter, FixedWindowRateLimiter)

    def test_farl_init_moving_window_strategy(self):
        """Test Farl with moving-window strategy"""
        from limits.strategies import MovingWindowRateLimiter

        farl = Farl(strategy="moving-window")

        assert hasattr(farl, "limiter")
        assert isinstance(farl.limiter, MovingWindowRateLimiter)

    def test_farl_init_sliding_window_strategy(self):
        """Test Farl with sliding-window-counter strategy"""
        from limits.strategies import SlidingWindowCounterRateLimiter

        farl = Farl(strategy="sliding-window-counter")

        assert hasattr(farl, "limiter")
        assert isinstance(farl.limiter, SlidingWindowCounterRateLimiter)

    def test_farl_init_invalid_strategy(self):
        """Test Farl initialization with invalid strategy"""
        with pytest.raises(ValueError, match="Unsupported strategy"):
            Farl(strategy="invalid-strategy")  # pyright: ignore[reportArgumentType]


class TestAsyncFarl:
    def test_async_farl_init_memory_storage(self):
        """Test AsyncFarl initialization with memory storage (default)"""
        farl = AsyncFarl()

        assert hasattr(farl, "limiter")
        assert farl.limiter is not None

    def test_async_farl_init_fixed_window_strategy(self):
        """Test AsyncFarl with fixed-window strategy"""
        farl = AsyncFarl(strategy="fixed-window")

        assert hasattr(farl, "limiter")
        assert "FixedWindow" in farl.limiter.__class__.__name__

    def test_async_farl_init_moving_window_strategy(self):
        """Test AsyncFarl with moving-window strategy"""
        farl = AsyncFarl(strategy="moving-window")

        assert hasattr(farl, "limiter")
        assert "MovingWindow" in farl.limiter.__class__.__name__

    def test_async_farl_init_sliding_window_strategy(self):
        """Test AsyncFarl with sliding-window-counter strategy"""
        farl = AsyncFarl(strategy="sliding-window-counter")

        assert hasattr(farl, "limiter")
        assert "SlidingWindow" in farl.limiter.__class__.__name__

    def test_async_farl_init_invalid_strategy(self):
        """Test AsyncFarl initialization with invalid strategy"""
        with pytest.raises(ValueError, match="Unsupported strategy"):
            AsyncFarl(strategy="invalid-strategy")  # pyright: ignore[reportArgumentType]
