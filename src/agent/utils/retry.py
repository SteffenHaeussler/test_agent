"""
Retry utility with exponential backoff for the agentic AI framework.

This module provides decorators for adding retry logic to both synchronous and
asynchronous functions. It includes:

- Configurable exponential backoff with optional jitter
- Smart exception classification (retryable vs non-retryable)
- Proper logging of retry attempts
- Preservation of function signatures and exception contexts
- Integration with the framework's exception hierarchy

Features:
- Exponential backoff with configurable base and maximum delay
- Jitter to prevent thundering herd problems
- Retry only on transient/retryable exceptions
- Comprehensive logging for observability
- Support for both sync and async functions
"""

import asyncio
import functools
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from loguru import logger

from src.agent.exceptions import (
    DatabaseException,
    ValidationException,
)

# Type variables for generic decorator support
F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Any])


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay in seconds between retries (default: 60.0)
        exponential_base: Base for exponential backoff calculation (default: 2.0)
        jitter: Whether to add random jitter to delays (default: True)
    """

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given retry attempt using exponential backoff.

        Args:
            attempt: The retry attempt number (0-based)

        Returns:
            Delay in seconds for this attempt
        """
        # Calculate exponential delay: initial_delay * (exponential_base ^ attempt)
        delay = self.initial_delay * (self.exponential_base**attempt)

        # Respect maximum delay
        delay = min(delay, self.max_delay)

        # Add jitter if enabled (0.5 to 1.0 multiplier)
        if self.jitter:
            jitter_factor = 0.5 + (random.random() * 0.5)
            delay *= jitter_factor

        return delay


def is_retryable_exception(exception: Exception) -> bool:
    """
    Determine if an exception should trigger a retry attempt.

    This function implements smart exception classification:
    - Retryable: Connection failures, timeouts, temporary unavailability
    - Non-retryable: Syntax errors, permission denied, data integrity violations

    Args:
        exception: The exception to classify

    Returns:
        True if the exception should trigger a retry, False otherwise
    """
    # Validation exceptions are never retryable
    if isinstance(exception, ValidationException):
        return False

    # For database exceptions, check the message content
    if isinstance(exception, DatabaseException):
        message = str(exception).lower()

        # Non-retryable conditions (permanent failures)
        non_retryable_patterns = [
            "syntax error",
            "permission denied",
            "access denied",
            "authorization failed",
            "authentication failed",
            "unique constraint",
            "foreign key constraint",
            "check constraint",
            "not null constraint",
            "integrity constraint",
            "duplicate key",
            "invalid column",
            "table does not exist",
            "column does not exist",
            "invalid sql",
            "malformed",
        ]

        for pattern in non_retryable_patterns:
            if pattern in message:
                return False

        # Retryable conditions (transient failures)
        retryable_patterns = [
            "connection",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "service unavailable",
            "too many connections",
            "connection refused",
            "network",
            "host unreachable",
            "connection reset",
            "connection lost",
        ]

        for pattern in retryable_patterns:
            if pattern in message:
                return True

        # Database exceptions are retryable by default (unknown transient issues)
        return True

    # Non-database exceptions are not retryable by default
    return False


def with_retry(config: Optional[RetryConfig] = None) -> Callable[[F], F]:
    """
    Decorator to add retry logic with exponential backoff to synchronous functions.

    Args:
        config: RetryConfig instance with retry parameters. If None, uses defaults.

    Returns:
        Decorated function with retry logic

    Example:
        @with_retry(RetryConfig(max_retries=5, initial_delay=0.5))
        def database_operation():
            # This will be retried up to 5 times on retryable exceptions
            return execute_query("SELECT * FROM users")
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(config.max_retries + 1):  # +1 for initial attempt
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # If this is the last attempt, re-raise the exception
                    if attempt == config.max_retries:
                        raise

                    # Check if this exception should trigger a retry
                    if not is_retryable_exception(e):
                        logger.debug(
                            f"Exception is not retryable, not retrying: {type(e).__name__}: {e}"
                        )
                        raise

                    # Calculate delay and log retry attempt
                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}, "
                        f"retrying in {delay:.2f} seconds. "
                        f"Error: {type(e).__name__}: {e}"
                    )

                    # Wait before retrying
                    time.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def with_async_retry(
    config: Optional[RetryConfig] = None,
) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator to add retry logic with exponential backoff to asynchronous functions.

    Args:
        config: RetryConfig instance with retry parameters. If None, uses defaults.

    Returns:
        Decorated async function with retry logic

    Example:
        @with_async_retry(RetryConfig(max_retries=5, initial_delay=0.5))
        async def async_database_operation():
            # This will be retried up to 5 times on retryable exceptions
            return await execute_query_async("SELECT * FROM users")
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(config.max_retries + 1):  # +1 for initial attempt
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # If this is the last attempt, re-raise the exception
                    if attempt == config.max_retries:
                        raise

                    # Check if this exception should trigger a retry
                    if not is_retryable_exception(e):
                        logger.debug(
                            f"Exception is not retryable, not retrying: {type(e).__name__}: {e}"
                        )
                        raise

                    # Calculate delay and log retry attempt
                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"Async function {func.__name__} failed on attempt {attempt + 1}, "
                        f"retrying in {delay:.2f} seconds. "
                        f"Error: {type(e).__name__}: {e}"
                    )

                    # Wait before retrying (async)
                    await asyncio.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


# Convenience functions for common retry scenarios
def with_database_retry(
    max_retries: int = 3, initial_delay: float = 1.0
) -> Callable[[F], F]:
    """
    Convenience decorator for database operations with sensible defaults.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
    """
    config = RetryConfig(max_retries=max_retries, initial_delay=initial_delay)
    return with_retry(config)


def with_async_database_retry(
    max_retries: int = 3, initial_delay: float = 1.0
) -> Callable[[AsyncF], AsyncF]:
    """
    Convenience decorator for async database operations with sensible defaults.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
    """
    config = RetryConfig(max_retries=max_retries, initial_delay=initial_delay)
    return with_async_retry(config)
