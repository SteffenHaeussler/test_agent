"""
Rate limiting middleware for the agentic AI framework.

This module provides:
- Rate limiting middleware that integrates with the message bus
- Configuration management for rate limits
- Session-based rate limiting
- Support for different rate limits per command type
- HTTP-style rate limit headers
"""

import os
from typing import Any, Dict
from dataclasses import dataclass

from loguru import logger

from src.agent.domain.commands import Command
from src.agent.exceptions import RateLimitException
from src.agent.utils.rate_limiter import RateLimiter, InMemoryStorage


@dataclass
class RateLimitConfig:
    """
    Configuration for rate limiting.

    Args:
        enabled: Whether rate limiting is enabled
        default_capacity: Default token bucket capacity
        default_refill_rate: Default token refill rate (tokens per second)
        cleanup_interval: Seconds between bucket cleanup
        per_session_limits: Per-session rate limit overrides
        per_command_limits: Per-command type rate limit overrides
    """

    enabled: bool = True
    default_capacity: int = 60
    default_refill_rate: float = 1.0
    cleanup_interval: int = 300
    per_session_limits: Dict[str, Dict[str, Any]] = None
    per_command_limits: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default empty dictionaries."""
        if self.per_session_limits is None:
            self.per_session_limits = {}
        if self.per_command_limits is None:
            self.per_command_limits = {}

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        """
        Create configuration from environment variables.

        Environment variables:
        - RATE_LIMIT_ENABLED: Enable/disable rate limiting (default: true)
        - RATE_LIMIT_DEFAULT_CAPACITY: Default capacity (default: 60)
        - RATE_LIMIT_DEFAULT_REFILL_RATE: Default refill rate (default: 1.0)
        - RATE_LIMIT_CLEANUP_INTERVAL: Cleanup interval in seconds (default: 300)

        Returns:
            RateLimitConfig instance
        """
        return cls(
            enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            default_capacity=int(os.getenv("RATE_LIMIT_DEFAULT_CAPACITY", "60")),
            default_refill_rate=float(
                os.getenv("RATE_LIMIT_DEFAULT_REFILL_RATE", "1.0")
            ),
            cleanup_interval=int(os.getenv("RATE_LIMIT_CLEANUP_INTERVAL", "300")),
        )


class RateLimitMiddleware:
    """
    Rate limiting middleware for command processing.

    Features:
    - Session-based rate limiting using command q_id
    - Configurable rate limits per command type
    - Async support
    - HTTP-style rate limit headers
    - Automatic cleanup of old buckets
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiting middleware.

        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.enabled = config.enabled

        if self.enabled:
            storage = InMemoryStorage(cleanup_interval=config.cleanup_interval)
            self.rate_limiter = RateLimiter(storage)
        else:
            self.rate_limiter = None

    def check_rate_limit(self, command: Command) -> None:
        """
        Check rate limit for a command and raise exception if exceeded.

        Args:
            command: Command to check rate limit for

        Raises:
            RateLimitException: If rate limit is exceeded
        """
        if not self.enabled:
            return

        rate_limit_key = self._get_rate_limit_key(command)
        capacity, refill_rate = self._get_rate_limit_parameters(command)

        logger.debug(
            f"Checking rate limit for {rate_limit_key} "
            f"(capacity: {capacity}, refill_rate: {refill_rate})"
        )

        allowed = self.rate_limiter.check_rate_limit(
            key=rate_limit_key, capacity=capacity, refill_rate=refill_rate
        )

        if not allowed:
            rate_info = self.rate_limiter.get_rate_limit_info(
                key=rate_limit_key, capacity=capacity, refill_rate=refill_rate
            )

            retry_after = int((capacity - rate_info["remaining"]) / refill_rate)

            context = {
                "rate_limit_key": rate_limit_key,
                "limit": capacity,
                "remaining": rate_info["remaining"],
                "retry_after": retry_after,
                "reset_time": rate_info["reset_time"],
                "command_type": type(command).__name__,
            }

            logger.warning(f"Rate limit exceeded for {rate_limit_key}", extra=context)

            raise RateLimitException(
                message=f"Rate limit exceeded for {rate_limit_key}. "
                f"Try again in {retry_after} seconds.",
                context=context,
            )

        logger.debug(f"Rate limit check passed for {rate_limit_key}")

    async def async_check_rate_limit(self, command: Command) -> None:
        """
        Async version of rate limit check.

        Args:
            command: Command to check rate limit for

        Raises:
            RateLimitException: If rate limit is exceeded
        """
        if not self.enabled:
            return

        rate_limit_key = self._get_rate_limit_key(command)
        capacity, refill_rate = self._get_rate_limit_parameters(command)

        allowed = await self.rate_limiter.async_check_rate_limit(
            key=rate_limit_key, capacity=capacity, refill_rate=refill_rate
        )

        if not allowed:
            rate_info = self.rate_limiter.get_rate_limit_info(
                key=rate_limit_key, capacity=capacity, refill_rate=refill_rate
            )

            retry_after = int((capacity - rate_info["remaining"]) / refill_rate)

            context = {
                "rate_limit_key": rate_limit_key,
                "limit": capacity,
                "remaining": rate_info["remaining"],
                "retry_after": retry_after,
                "reset_time": rate_info["reset_time"],
                "command_type": type(command).__name__,
            }

            raise RateLimitException(
                message=f"Rate limit exceeded for {rate_limit_key}. "
                f"Try again in {retry_after} seconds.",
                context=context,
            )

    def get_rate_limit_headers(self, command: Command) -> Dict[str, str]:
        """
        Get HTTP-style rate limit headers for a command.

        Args:
            command: Command to get headers for

        Returns:
            Dictionary of rate limit headers
        """
        if not self.enabled:
            return {}

        rate_limit_key = self._get_rate_limit_key(command)
        capacity, refill_rate = self._get_rate_limit_parameters(command)

        rate_info = self.rate_limiter.get_rate_limit_info(
            key=rate_limit_key, capacity=capacity, refill_rate=refill_rate
        )

        return {
            "X-RateLimit-Limit": str(capacity),
            "X-RateLimit-Remaining": str(rate_info["remaining"]),
            "X-RateLimit-Reset": str(int(rate_info["reset_time"])),
        }

    def reset_rate_limit(self, command: Command) -> None:
        """
        Reset rate limit for a command's session.

        Args:
            command: Command to reset rate limit for
        """
        if not self.enabled:
            return

        rate_limit_key = self._get_rate_limit_key(command)
        self.rate_limiter.reset_rate_limit(rate_limit_key)

        logger.info(f"Rate limit reset for {rate_limit_key}")

    def _get_rate_limit_key(self, command: Command) -> str:
        """
        Generate rate limit key for a command.

        Uses the command's q_id as session identifier.

        Args:
            command: Command to generate key for

        Returns:
            Rate limit key string
        """
        return f"session:{command.q_id}"

    def _get_rate_limit_parameters(self, command: Command) -> tuple[int, float]:
        """
        Get rate limit parameters for a command.

        Checks per-command limits first, then falls back to defaults.

        Args:
            command: Command to get parameters for

        Returns:
            Tuple of (capacity, refill_rate)
        """
        command_type = type(command).__name__

        # Check for per-command limits
        if command_type in self.config.per_command_limits:
            limits = self.config.per_command_limits[command_type]
            return limits["capacity"], limits["refill_rate"]

        # Use default limits
        return self.config.default_capacity, self.config.default_refill_rate

    @classmethod
    def from_env(cls) -> "RateLimitMiddleware":
        """
        Create rate limiting middleware from environment configuration.

        Returns:
            RateLimitMiddleware with environment-based configuration
        """
        config = RateLimitConfig.from_env()
        return cls(config)
