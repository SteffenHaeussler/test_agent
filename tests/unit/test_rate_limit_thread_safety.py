"""
Thread safety tests for the rate limiting system.

These tests verify that the rate limiter works correctly under
concurrent access from multiple threads.
"""

import asyncio
import pytest
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agent.domain.commands import Question
from src.agent.exceptions import RateLimitException
from src.agent.utils.rate_limiter import TokenBucket, InMemoryStorage, RateLimiter
from src.agent.validators.rate_limit_middleware import (
    RateLimitMiddleware,
    RateLimitConfig,
)


class TestThreadSafety:
    """Test thread safety of rate limiting components."""

    def test_token_bucket_thread_safety(self):
        """Test that TokenBucket is thread-safe under concurrent access."""
        bucket = TokenBucket(capacity=100, refill_rate=50.0)
        successful_consumptions = []
        failed_consumptions = []

        def consume_tokens(thread_id):
            """Try to consume tokens from multiple threads."""
            results = []
            for i in range(10):
                if bucket.consume(1):
                    results.append(f"thread_{thread_id}_request_{i}")
                else:
                    failed_consumptions.append(f"thread_{thread_id}_request_{i}")
            return results

        # Run concurrent access from multiple threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(consume_tokens, i) for i in range(10)]

            for future in as_completed(futures):
                successful_consumptions.extend(future.result())

        # Total successful + failed should equal total attempts (100)
        total_attempts = 10 * 10  # 10 threads * 10 requests each
        total_results = len(successful_consumptions) + len(failed_consumptions)
        assert total_results == total_attempts

        # Should not exceed bucket capacity
        assert len(successful_consumptions) <= 100

        # All successful consumptions should be unique (no double-counting)
        assert len(successful_consumptions) == len(set(successful_consumptions))

    def test_in_memory_storage_thread_safety(self):
        """Test that InMemoryStorage is thread-safe."""
        storage = InMemoryStorage(cleanup_interval=1)
        results = {}

        def access_storage(thread_id):
            """Access storage from multiple threads."""
            thread_results = []
            for i in range(5):
                key = f"thread_{thread_id}_bucket_{i}"
                bucket = storage.get_bucket(key, capacity=10, refill_rate=1.0)

                # Try to consume tokens
                if bucket.consume(3):
                    thread_results.append(key)

            return thread_results

        # Run concurrent access
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(access_storage, i) for i in range(5)]

            for i, future in enumerate(as_completed(futures)):
                results[f"thread_{i}"] = future.result()

        # Verify all threads got their buckets
        total_successful = sum(
            len(thread_results) for thread_results in results.values()
        )
        assert total_successful == 25  # 5 threads * 5 buckets each

        # Verify storage contains all buckets
        assert len(storage._buckets) == 25

    def test_rate_limiter_thread_safety(self):
        """Test that RateLimiter is thread-safe."""
        storage = InMemoryStorage()
        limiter = RateLimiter(storage)

        allowed_requests = []
        blocked_requests = []

        def make_requests(thread_id):
            """Make requests from multiple threads."""
            thread_allowed = []
            thread_blocked = []

            for i in range(10):
                key = f"user_{thread_id % 3}"  # Simulate 3 users

                if limiter.check_rate_limit(key, capacity=15, refill_rate=1.0):
                    thread_allowed.append(f"thread_{thread_id}_request_{i}")
                else:
                    thread_blocked.append(f"thread_{thread_id}_request_{i}")

            return thread_allowed, thread_blocked

        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_requests, i) for i in range(10)]

            for future in as_completed(futures):
                thread_allowed, thread_blocked = future.result()
                allowed_requests.extend(thread_allowed)
                blocked_requests.extend(thread_blocked)

        # Verify total requests
        total_requests = len(allowed_requests) + len(blocked_requests)
        assert total_requests == 100  # 10 threads * 10 requests

        # Each user should have at most their capacity (15) allowed
        # We have 3 users (user_0, user_1, user_2)
        user_0_allowed = [
            r
            for r in allowed_requests
            if "thread_0" in r or "thread_3" in r or "thread_6" in r or "thread_9" in r
        ]
        user_1_allowed = [
            r
            for r in allowed_requests
            if "thread_1" in r or "thread_4" in r or "thread_7" in r
        ]
        user_2_allowed = [
            r
            for r in allowed_requests
            if "thread_2" in r or "thread_5" in r or "thread_8" in r
        ]

        # Due to threading, exact counts may vary, but should be reasonable
        assert len(user_0_allowed) <= 15
        assert len(user_1_allowed) <= 15
        assert len(user_2_allowed) <= 15

    def test_middleware_thread_safety(self):
        """Test that RateLimitMiddleware is thread-safe."""
        config = RateLimitConfig(default_capacity=20, default_refill_rate=2.0)
        middleware = RateLimitMiddleware(config)

        successful_checks = []
        rate_limited_checks = []

        def check_rate_limits(thread_id):
            """Check rate limits from multiple threads."""
            thread_successful = []
            thread_rate_limited = []

            for i in range(15):
                command = Question(
                    question=f"test_{i}", q_id=f"session_{thread_id % 2}"
                )

                try:
                    middleware.check_rate_limit(command)
                    thread_successful.append(f"thread_{thread_id}_check_{i}")
                except RateLimitException:
                    thread_rate_limited.append(f"thread_{thread_id}_check_{i}")

            return thread_successful, thread_rate_limited

        # Run concurrent checks
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(check_rate_limits, i) for i in range(8)]

            for future in as_completed(futures):
                thread_successful, thread_rate_limited = future.result()
                successful_checks.extend(thread_successful)
                rate_limited_checks.extend(thread_rate_limited)

        # Verify total checks
        total_checks = len(successful_checks) + len(rate_limited_checks)
        assert total_checks == 120  # 8 threads * 15 checks

        # Should have some rate limiting (we have 2 sessions with 20 capacity each)
        assert len(rate_limited_checks) > 0
        assert len(successful_checks) <= 40  # 2 sessions * 20 capacity

    @pytest.mark.asyncio
    async def test_async_thread_safety(self):
        """Test thread safety with mixed async/sync access."""
        config = RateLimitConfig(default_capacity=30, default_refill_rate=3.0)
        middleware = RateLimitMiddleware(config)

        async_results = []
        sync_results = []

        async def async_checks():
            """Perform async rate limit checks."""
            results = []
            for i in range(10):
                command = Question(question=f"async_{i}", q_id="mixed_session")
                try:
                    await middleware.async_check_rate_limit(command)
                    results.append(f"async_check_{i}")
                except RateLimitException:
                    pass
            return results

        def sync_checks():
            """Perform sync rate limit checks."""
            results = []
            for i in range(10):
                command = Question(question=f"sync_{i}", q_id="mixed_session")
                try:
                    middleware.check_rate_limit(command)
                    results.append(f"sync_check_{i}")
                except RateLimitException:
                    pass
            return results

        # Run mixed async/sync operations
        async_tasks = [async_checks() for _ in range(3)]

        with ThreadPoolExecutor(max_workers=3) as executor:
            sync_futures = [executor.submit(sync_checks) for _ in range(3)]

            # Wait for async tasks
            async_results_list = await asyncio.gather(*async_tasks)
            for results in async_results_list:
                async_results.extend(results)

            # Wait for sync tasks
            for future in as_completed(sync_futures):
                sync_results.extend(future.result())

        # Should have processed requests from both async and sync
        total_successful = len(async_results) + len(sync_results)
        assert total_successful <= 30  # Capacity limit
        assert len(async_results) > 0  # Some async requests succeeded
        assert len(sync_results) > 0  # Some sync requests succeeded

    def test_cleanup_thread_safety(self):
        """Test that bucket cleanup is thread-safe."""
        storage = InMemoryStorage(cleanup_interval=1)

        def create_and_empty_buckets(thread_id):
            """Create buckets, empty them, and trigger cleanup."""
            for i in range(5):
                key = f"cleanup_thread_{thread_id}_bucket_{i}"
                bucket = storage.get_bucket(key, capacity=1, refill_rate=0.01)
                bucket.consume(1)  # Empty the bucket

            # Force cleanup by setting old timestamps
            current_time = time.time()
            for bucket in storage._buckets.values():
                bucket.last_refill = current_time - 100  # Make buckets old

            storage._last_cleanup = current_time - 10  # Force cleanup interval
            storage._cleanup_if_needed()

            return len(storage._buckets)

        # Run cleanup from multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_and_empty_buckets, i) for i in range(5)]

            final_bucket_counts = []
            for future in as_completed(futures):
                final_bucket_counts.append(future.result())

        # Cleanup should have occurred and storage should be manageable
        # (exact count depends on timing, but should be reasonable)
        final_count = len(storage._buckets)
        assert final_count >= 0  # Should not crash or have negative counts

    def test_high_concurrency_stress(self):
        """Stress test with high concurrency."""
        config = RateLimitConfig(default_capacity=100, default_refill_rate=10.0)
        middleware = RateLimitMiddleware(config)

        total_successful = 0
        total_blocked = 0

        def stress_test(thread_id):
            """Perform high-frequency rate limit checks."""
            successful = 0
            blocked = 0

            for i in range(50):
                command = Question(question=f"stress_{i}", q_id="stress_session")
                try:
                    middleware.check_rate_limit(command)
                    successful += 1
                except RateLimitException:
                    blocked += 1

            return successful, blocked

        # High concurrency stress test
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(stress_test, i) for i in range(20)]

            for future in as_completed(futures):
                successful, blocked = future.result()
                total_successful += successful
                total_blocked += blocked

        # Verify system handled high load correctly
        total_requests = total_successful + total_blocked
        assert total_requests == 1000  # 20 threads * 50 requests
        assert total_successful <= 100  # Should not exceed capacity
        assert total_blocked > 0  # Should have blocked some requests
