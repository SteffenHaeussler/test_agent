"""
Tests for RateLimitException and rate limit error handling.

Following TDD principles - create failing tests first.
"""

from src.agent.exceptions import RateLimitException


class TestRateLimitException:
    """Test RateLimitException functionality."""

    def test_rate_limit_exception_creation(self):
        """Should create RateLimitException with message and context."""
        context = {
            "key": "user123",
            "limit": 60,
            "remaining": 0,
            "reset_time": 1609459200.0,
        }

        exception = RateLimitException(
            message="Rate limit exceeded for user123", context=context
        )

        assert str(exception) == "Rate limit exceeded for user123"
        assert exception.context == context
        assert exception.context["key"] == "user123"
        assert exception.context["limit"] == 60
        assert exception.context["remaining"] == 0

    def test_rate_limit_exception_inheritance(self):
        """Should inherit from ValidationException."""
        from src.agent.exceptions import ValidationException

        exception = RateLimitException("Rate limited")

        assert isinstance(exception, ValidationException)
        assert isinstance(exception, Exception)

    def test_rate_limit_exception_with_retry_after(self):
        """Should include retry_after information in context."""
        context = {
            "key": "user123",
            "limit": 60,
            "remaining": 0,
            "retry_after": 30,
            "reset_time": 1609459200.0,
        }

        exception = RateLimitException(
            message="Rate limit exceeded. Try again in 30 seconds.", context=context
        )

        assert exception.context["retry_after"] == 30
        assert "Try again in 30 seconds" in str(exception)

    def test_rate_limit_exception_sanitizes_sensitive_data(self):
        """Should sanitize sensitive data from context."""
        context = {
            "rate_limit_key": "user123",
            "api_key": "secret_key_123",
            "limit": 60,
            "remaining": 0,
        }

        exception = RateLimitException(message="Rate limit exceeded", context=context)

        sanitized = exception.get_sanitized_context()
        assert sanitized["api_key"] == "[FILTERED]"
        assert sanitized["rate_limit_key"] == "user123"
        assert sanitized["limit"] == 60
