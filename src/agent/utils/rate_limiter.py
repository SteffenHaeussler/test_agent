"""
Rate limiting implementation using token bucket algorithm.

This module provides:
- Token bucket algorithm for rate limiting
- In-memory storage backend with automatic cleanup
- Async-compatible rate limiting
- Thread-safe implementation
"""

import asyncio
import threading
import time
from typing import Dict, Any


class TokenBucket:
    """
    Token bucket implementation for rate limiting.

    The token bucket algorithm allows bursts of requests up to the bucket
    capacity, then limits requests to the refill rate.

    Args:
        capacity: Maximum number of tokens in the bucket
        refill_rate: Rate at which tokens are added (tokens per second)
    """

    def __init__(self, capacity: int, refill_rate: float):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if refill_rate <= 0.0:
            raise ValueError("Refill rate must be positive")

        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)  # Start with full bucket
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)

        self.last_refill = now


class InMemoryStorage:
    """
    In-memory storage backend for rate limiting buckets.

    Features:
    - Thread-safe bucket access
    - Automatic cleanup of empty buckets
    - Configurable cleanup interval
    """

    def __init__(self, cleanup_interval: int = 300):
        """
        Initialize storage.

        Args:
            cleanup_interval: Seconds between cleanup runs
        """
        self.cleanup_interval = cleanup_interval
        self._buckets: Dict[str, TokenBucket] = {}
        self._last_cleanup = time.time()
        self._lock = threading.Lock()

    def get_bucket(self, key: str, capacity: int, refill_rate: float) -> TokenBucket:
        """
        Get or create a token bucket for the given key.

        Args:
            key: Unique identifier for the bucket
            capacity: Bucket capacity if creating new bucket
            refill_rate: Refill rate if creating new bucket

        Returns:
            TokenBucket instance
        """
        with self._lock:
            self._cleanup_if_needed()

            if key not in self._buckets:
                self._buckets[key] = TokenBucket(capacity, refill_rate)

            return self._buckets[key]

    def _cleanup_if_needed(self) -> None:
        """Clean up empty buckets if cleanup interval has passed."""
        now = time.time()
        if now - self._last_cleanup >= self.cleanup_interval:
            self._cleanup()
            self._last_cleanup = now

    def _cleanup(self) -> None:
        """Remove empty buckets to free memory."""
        keys_to_remove = []

        for key, bucket in self._buckets.items():
            # Remove buckets that are empty and haven't been refilled recently
            if bucket.tokens <= 0 and (time.time() - bucket.last_refill) > 60:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._buckets[key]


class RateLimiter:
    """
    Main rate limiter class with token bucket algorithm.

    Supports:
    - Configurable rate limits per key
    - Async and sync operations
    - Rate limit information queries
    - Rate limit resets
    """

    def __init__(self, storage: InMemoryStorage):
        """
        Initialize rate limiter.

        Args:
            storage: Storage backend for buckets
        """
        self.storage = storage

    def check_rate_limit(
        self, key: str, capacity: int, refill_rate: float, tokens: int = 1
    ) -> bool:
        """
        Check if request is within rate limit and consume tokens.

        Args:
            key: Unique identifier for rate limit (e.g., user_id, ip_address)
            capacity: Maximum tokens in bucket
            refill_rate: Rate of token refill (tokens per second)
            tokens: Number of tokens to consume

        Returns:
            True if within rate limit, False if rate limited
        """
        bucket = self.storage.get_bucket(key, capacity, refill_rate)
        return bucket.consume(tokens)

    async def async_check_rate_limit(
        self, key: str, capacity: int, refill_rate: float, tokens: int = 1
    ) -> bool:
        """
        Async version of rate limit check.

        Args:
            key: Unique identifier for rate limit
            capacity: Maximum tokens in bucket
            refill_rate: Rate of token refill (tokens per second)
            tokens: Number of tokens to consume

        Returns:
            True if within rate limit, False if rate limited
        """
        # Run synchronous check in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.check_rate_limit, key, capacity, refill_rate, tokens
        )

    def get_rate_limit_info(
        self, key: str, capacity: int, refill_rate: float
    ) -> Dict[str, Any]:
        """
        Get current rate limit information for a key.

        Args:
            key: Unique identifier for rate limit
            capacity: Bucket capacity
            refill_rate: Token refill rate

        Returns:
            Dictionary with rate limit information
        """
        bucket = self.storage.get_bucket(key, capacity, refill_rate)

        # Trigger refill to get current state
        with bucket._lock:
            bucket._refill()

            # Calculate when bucket will be full again
            tokens_needed = capacity - bucket.tokens
            reset_time = (
                time.time() + (tokens_needed / refill_rate)
                if tokens_needed > 0
                else time.time()
            )

            return {
                "remaining": int(bucket.tokens),
                "capacity": capacity,
                "refill_rate": refill_rate,
                "reset_time": reset_time,
            }

    def reset_rate_limit(self, key: str) -> None:
        """
        Reset rate limit for a key by removing its bucket.

        Args:
            key: Unique identifier for rate limit
        """
        with self.storage._lock:
            if key in self.storage._buckets:
                del self.storage._buckets[key]
