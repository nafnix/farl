from farl.constants import HEADER_RATELIMIT, HEADER_RATELIMIT_POLICY
from farl.exceptions import (
    AbnormalUsageDetected,
    FarlError,
    QuotaExceeded,
    TemporaryReducedCapacity,
)
from farl.openapi import openapi


class TestOpenapi:
    def test_openapi_with_farl_error(self):
        """Test openapi function with base FarlError"""
        result = openapi(FarlError)

        assert "model" in result
        assert "headers" in result
        assert result["model"] == FarlError

        headers = result["headers"]
        assert HEADER_RATELIMIT_POLICY in headers
        assert HEADER_RATELIMIT in headers

    def test_openapi_with_quota_exceeded(self):
        """Test openapi function with QuotaExceeded exception"""
        result = openapi(QuotaExceeded)

        assert result["model"] == QuotaExceeded

        headers = result["headers"]
        assert len(headers) == 2
        assert HEADER_RATELIMIT_POLICY in headers
        assert HEADER_RATELIMIT in headers

    def test_openapi_with_abnormal_usage_detected(self):
        """Test openapi function with AbnormalUsageDetected exception"""
        result = openapi(AbnormalUsageDetected)

        assert result["model"] == AbnormalUsageDetected

    def test_openapi_with_temporary_reduced_capacity(self):
        """Test openapi function with TemporaryReducedCapacity exception"""
        result = openapi(TemporaryReducedCapacity)

        assert result["model"] == TemporaryReducedCapacity

    def test_openapi_header_structure(self):
        """Test that openapi headers have correct structure"""
        result = openapi(FarlError)
        headers = result["headers"]

        # Test RateLimit-Policy header structure
        policy_header = headers[HEADER_RATELIMIT_POLICY]
        assert "schema" in policy_header
        assert policy_header["schema"]["type"] == "string"
        assert policy_header["schema"]["title"] == "RateLimit-Policy"

        # Test RateLimit header structure
        ratelimit_header = headers[HEADER_RATELIMIT]
        assert "schema" in ratelimit_header
        assert ratelimit_header["schema"]["type"] == "string"
        assert ratelimit_header["schema"]["title"] == "RateLimit"

    def test_openapi_return_type(self):
        """Test that openapi returns a dictionary with expected keys"""
        result = openapi(FarlError)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"model", "headers"}

    def test_openapi_headers_constants(self):
        """Test that openapi uses the correct header constants"""
        result = openapi(FarlError)
        headers = result["headers"]

        # Verify that the actual constants are used as keys
        assert HEADER_RATELIMIT_POLICY == "RateLimit-Policy"
        assert HEADER_RATELIMIT == "RateLimit"

        # Verify they match the headers
        assert "RateLimit-Policy" in headers
        assert "RateLimit" in headers

    def test_openapi_schema_consistency(self):
        """Test that different exception types return consistent schema structure"""
        exceptions = [
            FarlError,
            QuotaExceeded,
            AbnormalUsageDetected,
            TemporaryReducedCapacity,
        ]

        schemas = [openapi(exc) for exc in exceptions]

        # All schemas should have the same header structure
        for schema in schemas:
            assert "headers" in schema
            headers = schema["headers"]
            assert len(headers) == 2
            assert HEADER_RATELIMIT_POLICY in headers
            assert HEADER_RATELIMIT in headers

            # All headers should have the same schema structure
            for header_name in [HEADER_RATELIMIT_POLICY, HEADER_RATELIMIT]:
                header = headers[header_name]
                assert "schema" in header
                assert header["schema"]["type"] == "string"
                assert "title" in header["schema"]

    def test_openapi_with_custom_exception_class(self):
        """Test openapi with a custom exception class"""

        class CustomFarlError(FarlError):
            status = 429
            type = "custom_error"
            title = "Custom Rate Limit Error"

        result = openapi(CustomFarlError)

        assert result["model"] == CustomFarlError
        # Headers should be the same regardless of exception type
        assert len(result["headers"]) == 2
