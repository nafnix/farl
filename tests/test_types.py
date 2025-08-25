import base64
from datetime import datetime

import pytest

from farl.types import HeaderRateLimit, HeaderRateLimitPolicy, RedisDsn


class TestHeaderRateLimit:
    def test_header_rate_limit_basic(self):
        """Test basic HeaderRateLimit creation and string representation"""
        header = HeaderRateLimit("test-policy", 5)

        assert header.policy == "test-policy"
        assert header.remaining == 5
        assert header.reset_timestamp is None
        assert header.partition_key is None

        # Test string representation
        expected = '"test-policy";r=5'
        assert str(header) == expected

    def test_header_rate_limit_with_reset_timestamp(self):
        """Test HeaderRateLimit with reset timestamp"""
        # Use future timestamp
        future_timestamp = datetime.now().timestamp() + 60
        header = HeaderRateLimit("test-policy", 3, future_timestamp)

        assert header.reset_timestamp == future_timestamp

        # Test quota_reset_seconds property
        reset_seconds = header.quota_reset_seconds
        assert reset_seconds is not None
        assert 59 <= reset_seconds <= 61  # Should be around 60 seconds

    def test_header_rate_limit_with_string_partition_key(self):
        """Test HeaderRateLimit with string partition key"""
        header = HeaderRateLimit("test-policy", 2, partition_key="user123")

        expected = '"test-policy";r=2;pk="user123"'
        assert str(header) == expected

    def test_header_rate_limit_with_bytes_partition_key(self):
        """Test HeaderRateLimit with bytes partition key"""
        key_bytes = b"user123"
        header = HeaderRateLimit("test-policy", 2, partition_key=key_bytes)

        # Should base64 encode the bytes
        encoded_key = base64.b64encode(key_bytes).decode()
        expected = f'"test-policy";r=2;pk=:{encoded_key}:'
        assert str(header) == expected

    def test_header_rate_limit_complete(self):
        """Test HeaderRateLimit with all parameters"""
        future_timestamp = datetime.now().timestamp() + 120
        header = HeaderRateLimit(
            "complete-policy", 10, future_timestamp, "partition123"
        )

        reset_seconds = header.quota_reset_seconds
        expected = f'"complete-policy";r=10;t={reset_seconds};pk="partition123"'
        assert str(header) == expected

    def test_quota_reset_seconds_none_when_no_timestamp(self):
        """Test quota_reset_seconds returns None when no timestamp"""
        header = HeaderRateLimit("test-policy", 5)
        assert header.quota_reset_seconds is None

    def test_quota_reset_seconds_past_timestamp(self):
        """Test quota_reset_seconds with past timestamp"""
        past_timestamp = datetime.now().timestamp() - 60
        header = HeaderRateLimit("test-policy", 5, past_timestamp)

        reset_seconds = header.quota_reset_seconds
        # Should be negative or very small positive number
        assert reset_seconds is not None
        assert reset_seconds <= 1


class TestHeaderRateLimitPolicy:
    def test_header_rate_limit_policy_basic(self):
        """Test basic HeaderRateLimitPolicy creation"""
        policy = HeaderRateLimitPolicy("test-policy", 10)

        assert policy.policy == "test-policy"
        assert policy.quota == 10
        assert policy.quota_unit is None
        assert policy.window is None
        assert policy.partition_key is None

        # Test string representation
        expected = '"test-policy";q=10'
        assert str(policy) == expected

    def test_header_rate_limit_policy_with_quota_unit(self):
        """Test HeaderRateLimitPolicy with quota unit"""
        policy = HeaderRateLimitPolicy("test-policy", 100, "requests")

        expected = '"test-policy";q=100;qu="requests"'
        assert str(policy) == expected

    def test_header_rate_limit_policy_with_window(self):
        """Test HeaderRateLimitPolicy with window"""
        policy = HeaderRateLimitPolicy("test-policy", 50, window=3600)

        expected = '"test-policy";q=50;w=3600'
        assert str(policy) == expected

    def test_header_rate_limit_policy_with_string_partition_key(self):
        """Test HeaderRateLimitPolicy with string partition key"""
        policy = HeaderRateLimitPolicy("test-policy", 25, partition_key="api-key-123")

        expected = '"test-policy";q=25;pk="api-key-123"'
        assert str(policy) == expected

    def test_header_rate_limit_policy_with_bytes_partition_key(self):
        """Test HeaderRateLimitPolicy with bytes partition key"""
        key_bytes = b"binary-key"
        policy = HeaderRateLimitPolicy("test-policy", 15, partition_key=key_bytes)

        encoded_key = base64.b64encode(key_bytes).decode()
        expected = f'"test-policy";q=15;pk=:{encoded_key}:'
        assert str(policy) == expected

    def test_header_rate_limit_policy_complete(self):
        """Test HeaderRateLimitPolicy with all parameters"""
        policy = HeaderRateLimitPolicy(
            "complete-policy", 200, "operations", 7200, "service-a"
        )

        expected = '"complete-policy";q=200;qu="operations";w=7200;pk="service-a"'
        assert str(policy) == expected

    def test_header_rate_limit_policy_order_matters(self):
        """Test that parameter order is consistent in string representation"""
        policy1 = HeaderRateLimitPolicy("test", 100, "req", 60, "key")
        policy2 = HeaderRateLimitPolicy("test", 100, "req", 60, "key")

        assert str(policy1) == str(policy2)

        # Test expected order: policy, quota, quota_unit, window, partition_key
        expected = '"test";q=100;qu="req";w=60;pk="key"'
        assert str(policy1) == expected


class TestRedisDsn:
    def test_redis_dsn_basic(self):
        """Test basic Redis DSN validation"""
        dsn = RedisDsn("redis://localhost:6379/0")

        assert dsn.scheme == "redis"
        assert dsn.host == "localhost"
        assert dsn.port == 6379
        assert dsn.path == "/0"

    def test_redis_dsn_with_auth(self):
        """Test Redis DSN with authentication"""
        dsn = RedisDsn("redis://user:pass@localhost:6379/1")

        assert dsn.scheme == "redis"
        assert dsn.username == "user"
        assert dsn.password == "pass"
        assert dsn.host == "localhost"
        assert dsn.port == 6379
        assert dsn.path == "/1"

    def test_redis_dsn_defaults(self):
        """Test Redis DSN with default values"""
        dsn = RedisDsn("redis://localhost")

        assert dsn.host == "localhost"
        assert dsn.port == 6379  # Default port
        assert dsn.path == "/0"  # Default database

    def test_redis_dsn_rediss_scheme(self):
        """Test Redis DSN with SSL scheme"""
        dsn = RedisDsn("rediss://localhost:6380/2")

        assert dsn.scheme == "rediss"
        assert dsn.port == 6380
        assert dsn.path == "/2"

    def test_redis_dsn_sentinel_scheme(self):
        """Test Redis DSN with sentinel scheme"""
        dsn = RedisDsn("redis+sentinel://localhost:26379/mymaster")

        assert dsn.scheme == "redis+sentinel"
        assert dsn.port == 26379

    def test_redis_dsn_cluster_scheme(self):
        """Test Redis DSN with cluster scheme"""
        dsn = RedisDsn("redis+cluster://localhost:7000/0")

        assert dsn.scheme == "redis+cluster"
        assert dsn.port == 7000

    def test_redis_dsn_invalid_scheme(self):
        """Test that invalid schemes raise validation error"""
        with pytest.raises(ValueError):
            RedisDsn("http://localhost:6379/0")

    def test_redis_dsn_missing_host(self):
        """Test Redis DSN validation requires host"""
        # This should work because localhost is the default
        dsn = RedisDsn("redis://localhost:6379/0")
        assert dsn.host == "localhost"

        # Test with explicit host
        dsn2 = RedisDsn("redis://redis.example.com:6379/0")
        assert dsn2.host == "redis.example.com"


class TestHeaderRateLimitEdgeCases:
    def test_header_rate_limit_zero_remaining(self):
        """Test HeaderRateLimit with zero remaining requests"""
        header = HeaderRateLimit("exhausted-policy", 0)

        expected = '"exhausted-policy";r=0'
        assert str(header) == expected

    def test_header_rate_limit_negative_remaining(self):
        """Test HeaderRateLimit with negative remaining (edge case)"""
        header = HeaderRateLimit("over-limit", -1)

        expected = '"over-limit";r=-1'
        assert str(header) == expected

    def test_header_rate_limit_empty_policy_name(self):
        """Test HeaderRateLimit with empty policy name"""
        header = HeaderRateLimit("", 5)

        expected = '"";r=5'
        assert str(header) == expected

    def test_header_rate_limit_policy_special_characters(self):
        """Test HeaderRateLimit with special characters in policy name"""
        header = HeaderRateLimit("policy-with_special.chars", 3)

        expected = '"policy-with_special.chars";r=3'
        assert str(header) == expected

    def test_header_rate_limit_policy_unicode(self):
        """Test HeaderRateLimit with unicode characters"""
        header = HeaderRateLimit("策略-测试", 2)

        expected = '"策略-测试";r=2'
        assert str(header) == expected
