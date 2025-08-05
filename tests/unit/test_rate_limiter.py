"""
Tests for the rate limiting system.

Following TDD principles:
1. Write failing tests first
2. Implement minimum code to pass tests
3. Refactor while keeping tests green
"""

import pytest
from unittest.mock import patch

from src.agent.utils.rate_limiter import TokenBucket, RateLimiter, InMemoryStorage


class TestTokenBucket:
    """Test token bucket algorithm implementation."""

    def test_token_bucket_creation_with_valid_parameters(self):
        """Should create token bucket with valid parameters."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)

        assert bucket.capacity == 10
        assert bucket.refill_rate == 5.0
        assert bucket.tokens == 10  # Should start full
        assert bucket.last_refill is not None

    def test_token_bucket_creation_with_invalid_capacity(self):
        """Should raise ValueError for invalid capacity."""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            TokenBucket(capacity=0, refill_rate=5.0)

        with pytest.raises(ValueError, match="Capacity must be positive"):
            TokenBucket(capacity=-1, refill_rate=5.0)

    def test_token_bucket_creation_with_invalid_refill_rate(self):
        """Should raise ValueError for invalid refill rate."""
        with pytest.raises(ValueError, match="Refill rate must be positive"):
            TokenBucket(capacity=10, refill_rate=0.0)

        with pytest.raises(ValueError, match="Refill rate must be positive"):
            TokenBucket(capacity=10, refill_rate=-1.0)

    def test_consume_tokens_when_available(self):
        """Should consume tokens when available."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)

        # Should be able to consume tokens
        assert bucket.consume(3) is True
        assert abs(bucket.tokens - 7) < 0.01  # Account for small float precision

        # Should be able to consume remaining tokens
        assert bucket.consume(7) is True
        assert bucket.tokens < 0.01  # Account for small float precision

    def test_consume_tokens_when_insufficient(self):
        """Should reject consumption when insufficient tokens."""
        bucket = TokenBucket(capacity=5, refill_rate=2.0)

        # Consume all tokens
        assert bucket.consume(5) is True
        assert bucket.tokens < 0.01  # Account for small float precision

        # Should reject when no tokens left
        assert bucket.consume(1) is False
        assert bucket.tokens < 0.01  # Account for small float precision

    def test_token_refill_over_time(self):
        """Should refill tokens over time."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)  # 5 tokens per second

        # Consume all tokens
        assert bucket.consume(10) is True
        assert bucket.tokens < 0.01

        # Mock time progression (1 second = 5 tokens)
        with patch("src.agent.utils.rate_limiter.time.time") as mock_time:
            initial_time = 1000.0  # Use fixed time for consistency
            mock_time.return_value = initial_time
            bucket.last_refill = initial_time

            # Advance time by 1 second
            mock_time.return_value = initial_time + 1.0

            # Should refill 5 tokens
            bucket._refill()
            assert abs(bucket.tokens - 5.0) < 0.01

    def test_token_refill_does_not_exceed_capacity(self):
        """Should not refill beyond capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)

        # Start with full bucket
        assert abs(bucket.tokens - 10) < 0.01

        # Mock time progression
        with patch("src.agent.utils.rate_limiter.time.time") as mock_time:
            initial_time = 1000.0
            mock_time.return_value = initial_time
            bucket.last_refill = initial_time

            # Advance time by 2 seconds (would add 20 tokens)
            mock_time.return_value = initial_time + 2.0

            # Should cap at capacity
            bucket._refill()
            assert abs(bucket.tokens - 10) < 0.01

    def test_consume_triggers_refill(self):
        """Should automatically refill before consuming."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)

        # Consume all tokens
        bucket.consume(10)
        assert bucket.tokens < 0.01

        # Mock time advancement
        with patch("src.agent.utils.rate_limiter.time.time") as mock_time:
            initial_time = 1000.0
            mock_time.return_value = initial_time
            bucket.last_refill = initial_time

            # Advance time by 0.5 seconds (should add 5 tokens)
            mock_time.return_value = initial_time + 0.5

            # Should refill and allow consumption
            assert bucket.consume(3) is True
            assert abs(bucket.tokens - 2) < 0.01  # 5 refilled - 3 consumed


class TestInMemoryStorage:
    """Test in-memory storage backend for rate limiting."""

    def test_storage_creation(self):
        """Should create storage with cleanup interval."""
        storage = InMemoryStorage(cleanup_interval=300)

        assert storage.cleanup_interval == 300
        assert storage._buckets == {}
        assert storage._last_cleanup is not None

    def test_get_bucket_creates_new_bucket(self):
        """Should create new bucket for unknown key."""
        storage = InMemoryStorage()

        bucket = storage.get_bucket("test_key", capacity=5, refill_rate=2.0)

        assert isinstance(bucket, TokenBucket)
        assert bucket.capacity == 5
        assert bucket.refill_rate == 2.0

    def test_get_bucket_returns_existing_bucket(self):
        """Should return existing bucket for known key."""
        storage = InMemoryStorage()

        # Get bucket first time
        bucket1 = storage.get_bucket("test_key", capacity=5, refill_rate=2.0)
        bucket1.consume(2)  # Modify state

        # Get bucket second time
        bucket2 = storage.get_bucket("test_key", capacity=5, refill_rate=2.0)

        assert bucket1 is bucket2
        assert bucket2.tokens == 3  # State preserved

    def test_cleanup_removes_empty_buckets(self):
        """Should remove empty buckets during cleanup."""
        storage = InMemoryStorage(cleanup_interval=1)

        # Create bucket and empty it
        bucket = storage.get_bucket("test_key", capacity=5, refill_rate=0.1)
        bucket.consume(5)

        # Mock time to trigger cleanup
        with patch("src.agent.utils.rate_limiter.time.time") as mock_time:
            initial_time = 1000.0
            mock_time.return_value = initial_time
            storage._last_cleanup = initial_time
            bucket.last_refill = initial_time - 61  # Make bucket old enough for cleanup

            # Advance time to trigger cleanup
            mock_time.return_value = initial_time + 2

            # Force cleanup
            storage._cleanup_if_needed()

            # Empty bucket should be removed
            assert "test_key" not in storage._buckets


class TestRateLimiter:
    """Test rate limiter main class."""

    def test_rate_limiter_creation(self):
        """Should create rate limiter with storage backend."""
        storage = InMemoryStorage()
        limiter = RateLimiter(storage)

        assert limiter.storage is storage

    def test_check_rate_limit_allows_within_limit(self):
        """Should allow requests within rate limit."""
        storage = InMemoryStorage()
        limiter = RateLimiter(storage)

        # Should allow multiple requests within limit
        assert limiter.check_rate_limit("user1", capacity=5, refill_rate=1.0) is True
        assert limiter.check_rate_limit("user1", capacity=5, refill_rate=1.0) is True
        assert limiter.check_rate_limit("user1", capacity=5, refill_rate=1.0) is True

    def test_check_rate_limit_blocks_over_limit(self):
        """Should block requests over rate limit."""
        storage = InMemoryStorage()
        limiter = RateLimiter(storage)

        # Consume all tokens
        for _ in range(5):
            assert (
                limiter.check_rate_limit("user1", capacity=5, refill_rate=0.1) is True
            )

        # Next request should be blocked
        assert limiter.check_rate_limit("user1", capacity=5, refill_rate=0.1) is False

    def test_different_keys_have_separate_limits(self):
        """Should maintain separate limits for different keys."""
        storage = InMemoryStorage()
        limiter = RateLimiter(storage)

        # Exhaust limit for user1
        for _ in range(3):
            assert (
                limiter.check_rate_limit("user1", capacity=3, refill_rate=0.1) is True
            )
        assert limiter.check_rate_limit("user1", capacity=3, refill_rate=0.1) is False

        # user2 should still have full limit
        assert limiter.check_rate_limit("user2", capacity=3, refill_rate=0.1) is True

    @pytest.mark.asyncio
    async def test_async_check_rate_limit(self):
        """Should support async rate limit checking."""
        storage = InMemoryStorage()
        limiter = RateLimiter(storage)

        # Should work with async
        result = await limiter.async_check_rate_limit(
            "user1", capacity=5, refill_rate=1.0
        )
        assert result is True

        # Should maintain state across async calls
        for _ in range(4):
            result = await limiter.async_check_rate_limit(
                "user1", capacity=5, refill_rate=1.0
            )
            assert result is True

        # Should block on limit
        result = await limiter.async_check_rate_limit(
            "user1", capacity=5, refill_rate=1.0
        )
        assert result is False

    def test_get_rate_limit_info(self):
        """Should provide rate limit information."""
        storage = InMemoryStorage()
        limiter = RateLimiter(storage)

        # Use some tokens
        limiter.check_rate_limit("user1", capacity=10, refill_rate=2.0)
        limiter.check_rate_limit("user1", capacity=10, refill_rate=2.0)
        limiter.check_rate_limit("user1", capacity=10, refill_rate=2.0)

        info = limiter.get_rate_limit_info("user1", capacity=10, refill_rate=2.0)

        assert info["remaining"] == 7
        assert info["capacity"] == 10
        assert info["refill_rate"] == 2.0
        assert "reset_time" in info

    def test_reset_rate_limit(self):
        """Should reset rate limit for a key."""
        storage = InMemoryStorage()
        limiter = RateLimiter(storage)

        # Exhaust limit
        for _ in range(5):
            limiter.check_rate_limit("user1", capacity=5, refill_rate=0.1)

        assert limiter.check_rate_limit("user1", capacity=5, refill_rate=0.1) is False

        # Reset should restore full capacity
        limiter.reset_rate_limit("user1")
        assert limiter.check_rate_limit("user1", capacity=5, refill_rate=0.1) is True
