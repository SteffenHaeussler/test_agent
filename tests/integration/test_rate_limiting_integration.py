"""
Integration tests for the complete rate limiting system.

These tests verify the full integration of:
- Rate limiter core
- Middleware
- Configuration
- Exception handling
"""

import pytest
from unittest.mock import patch

from src.agent.domain.commands import Question, SQLQuestion
from src.agent.exceptions import RateLimitException
from src.agent.validators.rate_limit_middleware import (
    RateLimitMiddleware,
    RateLimitConfig,
)
from src.agent.config import get_rate_limit_config


class TestRateLimitingIntegration:
    """Integration tests for the complete rate limiting system."""

    def test_end_to_end_rate_limiting_flow(self):
        """Test complete rate limiting flow from config to enforcement."""
        # Set up environment for testing
        env_vars = {
            "RATE_LIMIT_ENABLED": "true",
            "RATE_LIMIT_DEFAULT_CAPACITY": "3",
            "RATE_LIMIT_DEFAULT_REFILL_RATE": "0.1",
            "RATE_LIMIT_QUESTION_CAPACITY": "2",
            "RATE_LIMIT_QUESTION_REFILL_RATE": "0.05",
        }

        with patch.dict("os.environ", env_vars):
            # Load configuration
            config_dict = get_rate_limit_config()

            # Create middleware from config
            config = RateLimitConfig(
                enabled=config_dict["enabled"],
                default_capacity=config_dict["default_capacity"],
                default_refill_rate=config_dict["default_refill_rate"],
                per_command_limits=config_dict["per_command_limits"],
            )
            middleware = RateLimitMiddleware(config)

            # Test Question command with custom limits
            question_cmd = Question(question="test question", q_id="session123")

            # Should allow requests within limit (2 for Question)
            middleware.check_rate_limit(question_cmd)
            middleware.check_rate_limit(question_cmd)

            # Should block third request
            with pytest.raises(RateLimitException) as exc_info:
                middleware.check_rate_limit(question_cmd)

            exception = exc_info.value
            assert "Rate limit exceeded" in str(exception)
            assert exception.context["limit"] == 2
            assert exception.context["remaining"] == 0
            assert "retry_after" in exception.context

    def test_integration_with_different_command_types(self):
        """Test rate limiting with different command types sharing sessions."""
        config_dict = {
            "enabled": True,
            "default_capacity": 10,
            "default_refill_rate": 1.0,
            "per_command_limits": {
                "Question": {"capacity": 5, "refill_rate": 0.5},
                "SQLQuestion": {"capacity": 3, "refill_rate": 0.3},
            },
        }

        config = RateLimitConfig(
            enabled=config_dict["enabled"],
            default_capacity=config_dict["default_capacity"],
            default_refill_rate=config_dict["default_refill_rate"],
            per_command_limits=config_dict["per_command_limits"],
        )
        middleware = RateLimitMiddleware(config)

        # Different command types for same session share the same bucket
        question_cmd = Question(question="test", q_id="session456")
        sql_cmd = SQLQuestion(question="SELECT * FROM test", q_id="session456")

        # Use the bucket with Question command (5 capacity)
        for _ in range(5):
            middleware.check_rate_limit(question_cmd)

        # SQL command should also be blocked (same session bucket)
        with pytest.raises(RateLimitException):
            middleware.check_rate_limit(sql_cmd)

    def test_rate_limit_headers_integration(self):
        """Test rate limit headers reflect actual state."""
        config = RateLimitConfig(default_capacity=5, default_refill_rate=1.0)
        middleware = RateLimitMiddleware(config)

        command = Question(question="test", q_id="session_headers")

        # Initial state
        headers = middleware.get_rate_limit_headers(command)
        assert headers["X-RateLimit-Limit"] == "5"
        assert headers["X-RateLimit-Remaining"] == "5"

        # After using some requests
        middleware.check_rate_limit(command)
        middleware.check_rate_limit(command)

        headers = middleware.get_rate_limit_headers(command)
        assert headers["X-RateLimit-Remaining"] == "3"

        # After hitting limit
        middleware.check_rate_limit(command)  # 2 remaining
        middleware.check_rate_limit(command)  # 1 remaining
        middleware.check_rate_limit(command)  # 0 remaining

        headers = middleware.get_rate_limit_headers(command)
        assert headers["X-RateLimit-Remaining"] == "0"

    @pytest.mark.asyncio
    async def test_async_integration(self):
        """Test async rate limiting integration."""
        config = RateLimitConfig(default_capacity=3, default_refill_rate=0.1)
        middleware = RateLimitMiddleware(config)

        command = Question(question="async test", q_id="async_session")

        # Should work with async
        await middleware.async_check_rate_limit(command)
        await middleware.async_check_rate_limit(command)
        await middleware.async_check_rate_limit(command)

        # Should raise exception when over limit
        with pytest.raises(RateLimitException):
            await middleware.async_check_rate_limit(command)

    def test_disabled_rate_limiting_integration(self):
        """Test that disabled rate limiting allows unlimited requests."""
        env_vars = {"RATE_LIMIT_ENABLED": "false"}

        with patch.dict("os.environ", env_vars):
            config_dict = get_rate_limit_config()
            config = RateLimitConfig(
                enabled=config_dict["enabled"],
                default_capacity=1,  # Very low limit
                default_refill_rate=0.01,  # Very slow refill
            )
            middleware = RateLimitMiddleware(config)

            command = Question(question="test", q_id="disabled_session")

            # Should allow many requests even with low limits
            for _ in range(100):
                middleware.check_rate_limit(command)  # Should not raise

    def test_rate_limit_reset_integration(self):
        """Test rate limit reset functionality."""
        config = RateLimitConfig(default_capacity=2, default_refill_rate=0.1)
        middleware = RateLimitMiddleware(config)

        command = Question(question="test", q_id="reset_session")

        # Exhaust the limit
        middleware.check_rate_limit(command)
        middleware.check_rate_limit(command)

        with pytest.raises(RateLimitException):
            middleware.check_rate_limit(command)

        # Reset should restore access
        middleware.reset_rate_limit(command)
        middleware.check_rate_limit(command)  # Should not raise

    def test_multiple_sessions_isolation(self):
        """Test that different sessions are properly isolated."""
        config = RateLimitConfig(default_capacity=2, default_refill_rate=0.1)
        middleware = RateLimitMiddleware(config)

        cmd1 = Question(question="test", q_id="session_a")
        cmd2 = Question(question="test", q_id="session_b")
        cmd3 = Question(question="test", q_id="session_c")

        # Exhaust session_a
        middleware.check_rate_limit(cmd1)
        middleware.check_rate_limit(cmd1)
        with pytest.raises(RateLimitException):
            middleware.check_rate_limit(cmd1)

        # session_b and session_c should still work
        middleware.check_rate_limit(cmd2)
        middleware.check_rate_limit(cmd2)
        middleware.check_rate_limit(cmd3)
        middleware.check_rate_limit(cmd3)

        # Now they should be exhausted too
        with pytest.raises(RateLimitException):
            middleware.check_rate_limit(cmd2)
        with pytest.raises(RateLimitException):
            middleware.check_rate_limit(cmd3)

    def test_configuration_validation_integration(self):
        """Test that invalid configurations are handled gracefully."""
        # Test with invalid values that should use defaults
        env_vars = {
            "RATE_LIMIT_ENABLED": "true",
            "RATE_LIMIT_DEFAULT_CAPACITY": "invalid",
            "RATE_LIMIT_DEFAULT_REFILL_RATE": "not_a_float",
        }

        with patch.dict("os.environ", env_vars):
            config_dict = get_rate_limit_config()

            # Should use safe defaults
            assert config_dict["default_capacity"] == 60
            assert config_dict["default_refill_rate"] == 1.0

            # Middleware should work with these defaults
            config = RateLimitConfig(
                enabled=config_dict["enabled"],
                default_capacity=config_dict["default_capacity"],
                default_refill_rate=config_dict["default_refill_rate"],
            )
            middleware = RateLimitMiddleware(config)

            command = Question(question="test", q_id="validation_session")
            middleware.check_rate_limit(command)  # Should not raise

    def test_exception_context_completeness(self):
        """Test that rate limit exceptions contain all necessary context."""
        config = RateLimitConfig(default_capacity=1, default_refill_rate=0.1)
        middleware = RateLimitMiddleware(config)

        command = Question(question="test", q_id="exception_session")

        # Use up the limit
        middleware.check_rate_limit(command)

        # Capture the exception
        with pytest.raises(RateLimitException) as exc_info:
            middleware.check_rate_limit(command)

        exception = exc_info.value
        context = exception.context

        # Verify all expected context fields are present
        assert "rate_limit_key" in context
        assert "limit" in context
        assert "remaining" in context
        assert "retry_after" in context
        assert "reset_time" in context
        assert "command_type" in context

        assert context["rate_limit_key"] == "session:exception_session"
        assert context["limit"] == 1
        assert context["remaining"] == 0
        assert context["command_type"] == "Question"
        assert isinstance(context["retry_after"], int)
        assert isinstance(context["reset_time"], float)

    def test_logging_integration(self):
        """Test that rate limiting produces appropriate log messages."""
        config = RateLimitConfig(default_capacity=1, default_refill_rate=0.1)
        middleware = RateLimitMiddleware(config)

        command = Question(question="test", q_id="logging_session")

        with patch("src.agent.validators.rate_limit_middleware.logger") as mock_logger:
            # Normal request should log debug
            middleware.check_rate_limit(command)
            mock_logger.debug.assert_called()

            # Rate limited request should log warning
            try:
                middleware.check_rate_limit(command)
            except RateLimitException:
                pass

            mock_logger.warning.assert_called()

            # Reset should log info
            middleware.reset_rate_limit(command)
            mock_logger.info.assert_called()
