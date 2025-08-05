"""
Tests for rate limiting middleware integration.

Following TDD principles - create failing tests first.
"""

import pytest
from unittest.mock import patch

from src.agent.domain.commands import Question, SQLQuestion
from src.agent.exceptions import RateLimitException
from src.agent.validators.rate_limit_middleware import (
    RateLimitMiddleware,
    RateLimitConfig,
)


class TestRateLimitConfig:
    """Test rate limit configuration."""

    def test_config_creation_with_defaults(self):
        """Should create config with default values."""
        config = RateLimitConfig()

        assert config.default_capacity == 60
        assert config.default_refill_rate == 1.0
        assert config.enabled is True
        assert config.cleanup_interval == 300
        assert config.per_session_limits == {}
        assert config.per_command_limits == {}

    def test_config_creation_with_custom_values(self):
        """Should create config with custom values."""
        per_session = {"premium": {"capacity": 120, "refill_rate": 2.0}}
        per_command = {"Question": {"capacity": 30, "refill_rate": 0.5}}

        config = RateLimitConfig(
            default_capacity=100,
            default_refill_rate=2.0,
            enabled=False,
            cleanup_interval=600,
            per_session_limits=per_session,
            per_command_limits=per_command,
        )

        assert config.default_capacity == 100
        assert config.default_refill_rate == 2.0
        assert config.enabled is False
        assert config.cleanup_interval == 600
        assert config.per_session_limits == per_session
        assert config.per_command_limits == per_command

    def test_config_from_env(self):
        """Should create config from environment variables."""
        env_vars = {
            "RATE_LIMIT_ENABLED": "true",
            "RATE_LIMIT_DEFAULT_CAPACITY": "120",
            "RATE_LIMIT_DEFAULT_REFILL_RATE": "2.0",
            "RATE_LIMIT_CLEANUP_INTERVAL": "600",
        }

        with patch.dict("os.environ", env_vars):
            config = RateLimitConfig.from_env()

        assert config.enabled is True
        assert config.default_capacity == 120
        assert config.default_refill_rate == 2.0
        assert config.cleanup_interval == 600


class TestRateLimitMiddleware:
    """Test rate limiting middleware."""

    def test_middleware_creation(self):
        """Should create middleware with config."""
        config = RateLimitConfig(default_capacity=60, default_refill_rate=1.0)
        middleware = RateLimitMiddleware(config)

        assert middleware.config == config
        assert middleware.rate_limiter is not None
        assert middleware.enabled is True

    def test_middleware_disabled_when_config_disabled(self):
        """Should disable middleware when config disabled."""
        config = RateLimitConfig(enabled=False)
        middleware = RateLimitMiddleware(config)

        assert middleware.enabled is False

    def test_get_rate_limit_key_for_session_id(self):
        """Should generate rate limit key from session ID."""
        config = RateLimitConfig()
        middleware = RateLimitMiddleware(config)

        command = Question(question="test", q_id="session123")
        key = middleware._get_rate_limit_key(command)

        assert key == "session:session123"

    def test_get_rate_limit_parameters_uses_defaults(self):
        """Should use default rate limit parameters."""
        config = RateLimitConfig(default_capacity=60, default_refill_rate=1.0)
        middleware = RateLimitMiddleware(config)

        command = Question(question="test", q_id="session123")
        capacity, refill_rate = middleware._get_rate_limit_parameters(command)

        assert capacity == 60
        assert refill_rate == 1.0

    def test_get_rate_limit_parameters_uses_per_command_limits(self):
        """Should use per-command rate limits when configured."""
        per_command = {"Question": {"capacity": 30, "refill_rate": 0.5}}
        config = RateLimitConfig(
            default_capacity=60, default_refill_rate=1.0, per_command_limits=per_command
        )
        middleware = RateLimitMiddleware(config)

        command = Question(question="test", q_id="session123")
        capacity, refill_rate = middleware._get_rate_limit_parameters(command)

        assert capacity == 30
        assert refill_rate == 0.5

    def test_check_rate_limit_allows_within_limit(self):
        """Should allow requests within rate limit."""
        config = RateLimitConfig(default_capacity=5, default_refill_rate=1.0)
        middleware = RateLimitMiddleware(config)

        command = Question(question="test", q_id="session123")

        # Should allow multiple requests within limit
        middleware.check_rate_limit(command)
        middleware.check_rate_limit(command)
        middleware.check_rate_limit(command)

    def test_check_rate_limit_raises_exception_over_limit(self):
        """Should raise RateLimitException when over limit."""
        config = RateLimitConfig(default_capacity=2, default_refill_rate=0.1)
        middleware = RateLimitMiddleware(config)

        command = Question(question="test", q_id="session123")

        # Use up the rate limit
        middleware.check_rate_limit(command)
        middleware.check_rate_limit(command)

        # Should raise exception on next request
        with pytest.raises(RateLimitException) as exc_info:
            middleware.check_rate_limit(command)

        exception = exc_info.value
        assert "Rate limit exceeded" in str(exception)
        assert exception.context["rate_limit_key"] == "session:session123"
        assert exception.context["limit"] == 2
        assert exception.context["remaining"] == 0
        assert "retry_after" in exception.context

    def test_check_rate_limit_disabled_middleware_allows_all(self):
        """Should allow all requests when middleware disabled."""
        config = RateLimitConfig(
            enabled=False, default_capacity=1, default_refill_rate=0.1
        )
        middleware = RateLimitMiddleware(config)

        command = Question(question="test", q_id="session123")

        # Should allow many requests even though capacity is 1
        for _ in range(10):
            middleware.check_rate_limit(command)

    def test_get_rate_limit_headers(self):
        """Should provide rate limit headers for HTTP responses."""
        config = RateLimitConfig(default_capacity=10, default_refill_rate=1.0)
        middleware = RateLimitMiddleware(config)

        command = Question(question="test", q_id="session123")

        # Use some of the rate limit
        middleware.check_rate_limit(command)
        middleware.check_rate_limit(command)

        headers = middleware.get_rate_limit_headers(command)

        assert headers["X-RateLimit-Limit"] == "10"
        assert headers["X-RateLimit-Remaining"] == "8"
        assert "X-RateLimit-Reset" in headers

    @pytest.mark.asyncio
    async def test_async_check_rate_limit(self):
        """Should support async rate limit checking."""
        config = RateLimitConfig(default_capacity=5, default_refill_rate=1.0)
        middleware = RateLimitMiddleware(config)

        command = Question(question="test", q_id="session123")

        # Should work with async
        await middleware.async_check_rate_limit(command)
        await middleware.async_check_rate_limit(command)

        # Should maintain state across async calls
        headers = middleware.get_rate_limit_headers(command)
        assert headers["X-RateLimit-Remaining"] == "3"

    @pytest.mark.asyncio
    async def test_async_check_rate_limit_raises_exception(self):
        """Should raise exception in async when over limit."""
        config = RateLimitConfig(default_capacity=1, default_refill_rate=0.1)
        middleware = RateLimitMiddleware(config)

        command = Question(question="test", q_id="session123")

        # Use up the rate limit
        await middleware.async_check_rate_limit(command)

        # Should raise exception on next request
        with pytest.raises(RateLimitException):
            await middleware.async_check_rate_limit(command)

    def test_different_sessions_have_separate_limits(self):
        """Should maintain separate limits for different sessions."""
        config = RateLimitConfig(default_capacity=2, default_refill_rate=0.1)
        middleware = RateLimitMiddleware(config)

        command1 = Question(question="test", q_id="session1")
        command2 = Question(question="test", q_id="session2")

        # Exhaust limit for session1
        middleware.check_rate_limit(command1)
        middleware.check_rate_limit(command1)

        with pytest.raises(RateLimitException):
            middleware.check_rate_limit(command1)

        # session2 should still have full limit
        middleware.check_rate_limit(command2)
        middleware.check_rate_limit(command2)

    def test_reset_rate_limit(self):
        """Should reset rate limit for a session."""
        config = RateLimitConfig(default_capacity=2, default_refill_rate=0.1)
        middleware = RateLimitMiddleware(config)

        command = Question(question="test", q_id="session123")

        # Exhaust limit
        middleware.check_rate_limit(command)
        middleware.check_rate_limit(command)

        with pytest.raises(RateLimitException):
            middleware.check_rate_limit(command)

        # Reset should restore full capacity
        middleware.reset_rate_limit(command)
        middleware.check_rate_limit(command)  # Should not raise

    def test_middleware_with_different_command_types(self):
        """Should handle different command types appropriately."""
        per_command = {
            "Question": {"capacity": 10, "refill_rate": 1.0},
            "SQLQuestion": {"capacity": 5, "refill_rate": 0.5},
        }
        config = RateLimitConfig(
            default_capacity=20, default_refill_rate=2.0, per_command_limits=per_command
        )
        middleware = RateLimitMiddleware(config)

        question_cmd = Question(question="test", q_id="session123")
        sql_cmd = SQLQuestion(question="test", q_id="session123")

        # Same session, different command types should use different limits
        # but share the same rate limit key (session-based)

        # Use up Question limit (10 requests)
        for _ in range(10):
            middleware.check_rate_limit(question_cmd)

        # SQLQuestion should be limited by the same bucket since they share session
        with pytest.raises(RateLimitException):
            middleware.check_rate_limit(sql_cmd)
