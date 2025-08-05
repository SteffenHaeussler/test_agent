"""
Test suite for retry utility with exponential backoff.

This test suite follows TDD principles:
1. Write failing tests first that define the expected behavior
2. Tests cover both sync and async retry decorators
3. Tests proper exception handling and retry logic
4. Tests exponential backoff timing
5. Tests configuration parameters
6. Tests integration with database exception hierarchy
"""

from unittest.mock import Mock, patch, AsyncMock
import pytest

from src.agent.utils.retry import (
    RetryConfig,
    with_retry,
    with_async_retry,
    is_retryable_exception,
)
from src.agent.exceptions import (
    DatabaseConnectionException,
    DatabaseQueryException,
    DatabaseTransactionException,
    InputValidationException,
)


class TestRetryConfig:
    """Test RetryConfig data class."""

    def test_default_config_values(self):
        """Test that RetryConfig has sensible defaults."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_config_values(self):
        """Test that custom config values are set correctly."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False,
        )

        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False

    def test_calculate_delay_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=False)

        assert config.calculate_delay(0) == 1.0  # 1 * 2^0
        assert config.calculate_delay(1) == 2.0  # 1 * 2^1
        assert config.calculate_delay(2) == 4.0  # 1 * 2^2
        assert config.calculate_delay(3) == 8.0  # 1 * 2^3

    def test_calculate_delay_respects_max_delay(self):
        """Test that calculated delay respects max_delay."""
        config = RetryConfig(initial_delay=1.0, max_delay=5.0, jitter=False)

        # Should cap at max_delay
        assert config.calculate_delay(10) == 5.0

    def test_calculate_delay_with_jitter(self):
        """Test that jitter adds randomness to delay."""
        config = RetryConfig(initial_delay=2.0, jitter=True)

        delay1 = config.calculate_delay(1)
        delay2 = config.calculate_delay(1)

        # With jitter, delays should be different (probabilistically)
        # Base delay is 4.0, jitter should make it between 2.0 and 4.0
        assert 2.0 <= delay1 <= 4.0
        assert 2.0 <= delay2 <= 4.0


class TestRetryableExceptionClassification:
    """Test classification of exceptions as retryable or not."""

    def test_connection_exceptions_are_retryable(self):
        """Test that connection exceptions are retryable."""
        exc = DatabaseConnectionException("Connection failed")
        assert is_retryable_exception(exc) is True

    def test_timeout_exceptions_are_retryable(self):
        """Test that timeout-related exceptions are retryable."""
        # Test with timeout in message
        exc = DatabaseQueryException("Query timeout after 30 seconds")
        assert is_retryable_exception(exc) is True

        # Test with different timeout message
        exc = DatabaseConnectionException("Connection timed out")
        assert is_retryable_exception(exc) is True

    def test_temporary_unavailable_exceptions_are_retryable(self):
        """Test that temporary unavailable exceptions are retryable."""
        exc = DatabaseConnectionException("Service temporarily unavailable")
        assert is_retryable_exception(exc) is True

        exc = DatabaseQueryException("Database is temporarily unavailable")
        assert is_retryable_exception(exc) is True

    def test_syntax_errors_are_not_retryable(self):
        """Test that syntax errors are not retryable."""
        exc = DatabaseQueryException("SQL syntax error near 'FROM'")
        assert is_retryable_exception(exc) is False

    def test_permission_errors_are_not_retryable(self):
        """Test that permission errors are not retryable."""
        exc = DatabaseQueryException("Permission denied for table users")
        assert is_retryable_exception(exc) is False

    def test_integrity_violations_are_not_retryable(self):
        """Test that data integrity violations are not retryable."""
        exc = DatabaseTransactionException("UNIQUE constraint failed")
        assert is_retryable_exception(exc) is False

        exc = DatabaseTransactionException("FOREIGN KEY constraint failed")
        assert is_retryable_exception(exc) is False

    def test_validation_exceptions_are_not_retryable(self):
        """Test that validation exceptions are not retryable."""
        exc = InputValidationException("Invalid input format")
        assert is_retryable_exception(exc) is False

    def test_unknown_database_exceptions_are_retryable_by_default(self):
        """Test that unknown database exceptions are retryable by default."""
        exc = DatabaseQueryException("Unknown database error")
        assert is_retryable_exception(exc) is True


class TestSyncRetryDecorator:
    """Test synchronous retry decorator functionality."""

    def test_with_retry_succeeds_on_first_attempt(self):
        """Test that successful functions are not retried."""
        mock_func = Mock(return_value="success")

        @with_retry()
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_with_retry_retries_on_retryable_exception(self):
        """Test that retryable exceptions trigger retries."""
        mock_func = Mock()
        mock_func.side_effect = [
            DatabaseConnectionException("Connection failed"),
            DatabaseConnectionException("Connection failed"),
            "success",
        ]

        @with_retry(RetryConfig(max_retries=3, initial_delay=0.01))
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_with_retry_raises_after_max_retries(self):
        """Test that retry gives up after max_retries attempts."""
        original_exception = DatabaseConnectionException("Connection failed")
        mock_func = Mock(side_effect=original_exception)

        @with_retry(RetryConfig(max_retries=2, initial_delay=0.01))
        def test_func():
            return mock_func()

        with pytest.raises(DatabaseConnectionException) as exc_info:
            test_func()

        # Should have tried 3 times total (initial + 2 retries)
        assert mock_func.call_count == 3
        # Should raise the original exception
        assert exc_info.value is original_exception

    def test_with_retry_does_not_retry_non_retryable_exceptions(self):
        """Test that non-retryable exceptions are not retried."""
        original_exception = DatabaseQueryException("SQL syntax error")
        mock_func = Mock(side_effect=original_exception)

        @with_retry(RetryConfig(max_retries=3, initial_delay=0.01))
        def test_func():
            return mock_func()

        with pytest.raises(DatabaseQueryException) as exc_info:
            test_func()

        # Should only try once
        assert mock_func.call_count == 1
        assert exc_info.value is original_exception

    @patch("time.sleep")
    def test_with_retry_applies_exponential_backoff(self, mock_sleep):
        """Test that retry applies exponential backoff delays."""
        mock_func = Mock()
        mock_func.side_effect = [
            DatabaseConnectionException("Connection failed"),
            DatabaseConnectionException("Connection failed"),
            "success",
        ]

        config = RetryConfig(
            max_retries=2, initial_delay=1.0, exponential_base=2.0, jitter=False
        )

        @with_retry(config)
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 3

        # Should have slept with exponential backoff
        expected_delays = [1.0, 2.0]  # 1*2^0, 1*2^1
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays

    @patch("src.agent.utils.retry.logger")
    def test_with_retry_logs_retry_attempts(self, mock_logger):
        """Test that retry attempts are logged with proper context."""
        mock_func = Mock()
        mock_func.side_effect = [
            DatabaseConnectionException("Connection failed"),
            "success",
        ]

        @with_retry(RetryConfig(max_retries=1, initial_delay=0.01))
        def test_func():
            return mock_func()

        result = test_func()

        assert result == "success"
        # Should log the retry attempt
        mock_logger.warning.assert_called_once()
        log_message = mock_logger.warning.call_args[0][0]
        assert "failed on attempt 1" in log_message.lower()


class TestAsyncRetryDecorator:
    """Test asynchronous retry decorator functionality."""

    @pytest.mark.asyncio
    async def test_with_async_retry_succeeds_on_first_attempt(self):
        """Test that successful async functions are not retried."""
        mock_func = AsyncMock(return_value="success")

        @with_async_retry()
        async def test_func():
            return await mock_func()

        result = await test_func()

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_with_async_retry_retries_on_retryable_exception(self):
        """Test that retryable exceptions trigger async retries."""
        mock_func = AsyncMock()
        mock_func.side_effect = [
            DatabaseConnectionException("Connection failed"),
            DatabaseConnectionException("Connection failed"),
            "success",
        ]

        @with_async_retry(RetryConfig(max_retries=3, initial_delay=0.01))
        async def test_func():
            return await mock_func()

        result = await test_func()

        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_with_async_retry_raises_after_max_retries(self):
        """Test that async retry gives up after max_retries attempts."""
        original_exception = DatabaseConnectionException("Connection failed")
        mock_func = AsyncMock(side_effect=original_exception)

        @with_async_retry(RetryConfig(max_retries=2, initial_delay=0.01))
        async def test_func():
            return await mock_func()

        with pytest.raises(DatabaseConnectionException) as exc_info:
            await test_func()

        # Should have tried 3 times total (initial + 2 retries)
        assert mock_func.call_count == 3
        # Should raise the original exception
        assert exc_info.value is original_exception

    @pytest.mark.asyncio
    async def test_with_async_retry_does_not_retry_non_retryable_exceptions(self):
        """Test that non-retryable exceptions are not retried in async context."""
        original_exception = DatabaseQueryException("SQL syntax error")
        mock_func = AsyncMock(side_effect=original_exception)

        @with_async_retry(RetryConfig(max_retries=3, initial_delay=0.01))
        async def test_func():
            return await mock_func()

        with pytest.raises(DatabaseQueryException) as exc_info:
            await test_func()

        # Should only try once
        assert mock_func.call_count == 1
        assert exc_info.value is original_exception

    @pytest.mark.asyncio
    @patch("asyncio.sleep")
    async def test_with_async_retry_applies_exponential_backoff(self, mock_sleep):
        """Test that async retry applies exponential backoff delays."""
        mock_func = AsyncMock()
        mock_func.side_effect = [
            DatabaseConnectionException("Connection failed"),
            DatabaseConnectionException("Connection failed"),
            "success",
        ]

        config = RetryConfig(
            max_retries=2, initial_delay=1.0, exponential_base=2.0, jitter=False
        )

        @with_async_retry(config)
        async def test_func():
            return await mock_func()

        result = await test_func()

        assert result == "success"
        assert mock_func.call_count == 3

        # Should have slept with exponential backoff
        expected_delays = [1.0, 2.0]  # 1*2^0, 1*2^1
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays

    @pytest.mark.asyncio
    @patch("src.agent.utils.retry.logger")
    async def test_with_async_retry_logs_retry_attempts(self, mock_logger):
        """Test that async retry attempts are logged with proper context."""
        mock_func = AsyncMock()
        mock_func.side_effect = [
            DatabaseConnectionException("Connection failed"),
            "success",
        ]

        @with_async_retry(RetryConfig(max_retries=1, initial_delay=0.01))
        async def test_func():
            return await mock_func()

        result = await test_func()

        assert result == "success"
        # Should log the retry attempt
        mock_logger.warning.assert_called_once()
        log_message = mock_logger.warning.call_args[0][0]
        assert "failed on attempt 1" in log_message.lower()


class TestRetryDecoratorWithFunctionArguments:
    """Test that retry decorators preserve function signatures and arguments."""

    def test_with_retry_preserves_function_arguments(self):
        """Test that decorated function receives all arguments correctly."""

        @with_retry(RetryConfig(max_retries=1, initial_delay=0.01))
        def test_func(arg1, arg2, kwarg1=None):
            return f"{arg1}-{arg2}-{kwarg1}"

        result = test_func("a", "b", kwarg1="c")
        assert result == "a-b-c"

    @pytest.mark.asyncio
    async def test_with_async_retry_preserves_function_arguments(self):
        """Test that decorated async function receives all arguments correctly."""

        @with_async_retry(RetryConfig(max_retries=1, initial_delay=0.01))
        async def test_func(arg1, arg2, kwarg1=None):
            return f"{arg1}-{arg2}-{kwarg1}"

        result = await test_func("a", "b", kwarg1="c")
        assert result == "a-b-c"

    def test_with_retry_preserves_function_metadata(self):
        """Test that decorated function preserves original function metadata."""

        @with_retry()
        def test_func():
            """Test function docstring."""
            return "test"

        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test function docstring."


class TestRetryDecoratorErrorContexts:
    """Test that retry decorators preserve error contexts and exception chaining."""

    def test_with_retry_preserves_exception_context(self):
        """Test that retry preserves exception context after all retries fail."""
        original_context = {"operation": "test", "attempt": 1}
        original_exception = DatabaseConnectionException(
            "Connection failed", context=original_context
        )

        mock_func = Mock(side_effect=original_exception)

        @with_retry(RetryConfig(max_retries=1, initial_delay=0.01))
        def test_func():
            return mock_func()

        with pytest.raises(DatabaseConnectionException) as exc_info:
            test_func()

        # Should preserve original context
        assert exc_info.value.context == original_context
        assert exc_info.value is original_exception

    @pytest.mark.asyncio
    async def test_with_async_retry_preserves_exception_context(self):
        """Test that async retry preserves exception context after all retries fail."""
        original_context = {"operation": "test", "attempt": 1}
        original_exception = DatabaseConnectionException(
            "Connection failed", context=original_context
        )

        mock_func = AsyncMock(side_effect=original_exception)

        @with_async_retry(RetryConfig(max_retries=1, initial_delay=0.01))
        async def test_func():
            return await mock_func()

        with pytest.raises(DatabaseConnectionException) as exc_info:
            await test_func()

        # Should preserve original context
        assert exc_info.value.context == original_context
        assert exc_info.value is original_exception
