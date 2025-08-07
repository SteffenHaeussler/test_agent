import hashlib
import json
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
import redis.exceptions

from src.agent.adapters.cache import CacheManager, CacheMetrics, CacheStrategy


class TestCacheManager:
    """Test suite for CacheManager following TDD principles."""

    @pytest.fixture
    def cache_config(self):
        """Fixture providing cache configuration."""
        return {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "password": None,
            "max_connections": 10,
            "decode_responses": True,
        }

    @pytest_asyncio.fixture
    async def cache_manager(self, cache_config):
        """Fixture providing a CacheManager instance."""
        manager = CacheManager(cache_config)
        # Don't actually initialize Redis for tests
        manager.enabled = True
        manager.redis = AsyncMock()
        yield manager
        await manager.close()

    def test_cache_manager_initialization(self, cache_config):
        """Test that CacheManager initializes with correct configuration."""
        manager = CacheManager(cache_config)

        assert manager.config == cache_config
        assert manager.redis is None  # Not initialized yet
        assert manager.metrics is not None
        assert isinstance(manager.metrics, CacheMetrics)
        assert manager.enabled is True

    @pytest.mark.asyncio
    async def test_cache_manager_initialize_success(self, cache_config):
        """Test successful Redis connection initialization."""
        manager = CacheManager(cache_config)

        with patch("src.agent.adapters.cache.redis.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_from_url.return_value = mock_redis

            await manager.initialize()

            assert manager.redis is not None
            assert manager.enabled is True
            mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_manager_initialize_failure_disables_cache(self, cache_config):
        """Test that cache is disabled when Redis connection fails."""
        manager = CacheManager(cache_config)

        with patch("src.agent.adapters.cache.redis.from_url") as mock_from_url:
            mock_from_url.side_effect = redis.exceptions.ConnectionError(
                "Connection failed"
            )

            await manager.initialize()

            assert manager.redis is None
            assert manager.enabled is False

    def test_generate_cache_key_with_simple_params(self):
        """Test cache key generation for simple parameters."""
        manager = CacheManager({})

        key = manager.generate_cache_key("test_prefix", query="hello", model="gpt-4")

        # Key should be deterministic and include all parameters
        expected_data = {"query": "hello", "model": "gpt-4"}
        expected_hash = hashlib.sha256(
            json.dumps(expected_data, sort_keys=True).encode()
        ).hexdigest()
        expected_key = f"test_prefix:{expected_hash}"

        assert key == expected_key

    def test_generate_cache_key_with_complex_params(self):
        """Test cache key generation for complex nested parameters."""
        manager = CacheManager({})

        complex_params = {
            "query": "complex query",
            "config": {"temperature": 0.7, "max_tokens": 100},
            "user_context": ["context1", "context2"],
        }

        key = manager.generate_cache_key("complex_prefix", **complex_params)

        # Should handle complex nested structures
        assert key.startswith("complex_prefix:")
        assert len(key.split(":")[1]) == 64  # SHA256 hash length

    @pytest.mark.asyncio
    async def test_set_cache_success(self, cache_manager):
        """Test successfully setting a cache value."""
        cache_manager.redis = AsyncMock()

        await cache_manager.set("test_key", "test_value", ttl=300)

        cache_manager.redis.setex.assert_called_once_with(
            "test_key", 300, '"test_value"'
        )

    @pytest.mark.asyncio
    async def test_set_cache_disabled(self):
        """Test that set operations are skipped when cache is disabled."""
        manager = CacheManager({})
        manager.enabled = False

        # Should not raise an exception
        await manager.set("test_key", "test_value")

    @pytest.mark.asyncio
    async def test_get_cache_hit(self, cache_manager):
        """Test successful cache hit."""
        cache_manager.redis = AsyncMock()
        cache_manager.redis.get.return_value = '"test_value"'

        result = await cache_manager.get("test_key")

        assert result == "test_value"
        cache_manager.redis.get.assert_called_once_with("test_key")
        assert cache_manager.metrics.hits == 1

    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache_manager):
        """Test cache miss."""
        cache_manager.redis = AsyncMock()
        cache_manager.redis.get.return_value = None

        result = await cache_manager.get("test_key")

        assert result is None
        cache_manager.redis.get.assert_called_once_with("test_key")
        assert cache_manager.metrics.misses == 1

    @pytest.mark.asyncio
    async def test_get_cache_disabled(self):
        """Test that get operations return None when cache is disabled."""
        manager = CacheManager({})
        manager.enabled = False

        result = await manager.get("test_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_cache_success(self, cache_manager):
        """Test successful cache deletion."""
        cache_manager.redis = AsyncMock()
        cache_manager.redis.delete.return_value = 1

        result = await cache_manager.delete("test_key")

        assert result is True
        cache_manager.redis.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_delete_cache_key_not_found(self, cache_manager):
        """Test cache deletion when key doesn't exist."""
        cache_manager.redis = AsyncMock()
        cache_manager.redis.delete.return_value = 0

        result = await cache_manager.delete("nonexistent_key")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_pattern_success(self, cache_manager):
        """Test successful pattern-based cache deletion."""
        cache_manager.redis = AsyncMock()

        # Mock async iterator
        async def mock_scan_iter(match):
            for key in ["key1", "key2", "key3"]:
                yield key

        cache_manager.redis.scan_iter = mock_scan_iter
        cache_manager.redis.delete.return_value = 3

        result = await cache_manager.delete_pattern("test_*")

        assert result == 3
        cache_manager.redis.delete.assert_called_once_with("key1", "key2", "key3")

    @pytest.mark.asyncio
    async def test_exists_key_found(self, cache_manager):
        """Test checking existence of a key that exists."""
        cache_manager.redis = AsyncMock()
        cache_manager.redis.exists.return_value = 1

        result = await cache_manager.exists("test_key")

        assert result is True
        cache_manager.redis.exists.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_exists_key_not_found(self, cache_manager):
        """Test checking existence of a key that doesn't exist."""
        cache_manager.redis = AsyncMock()
        cache_manager.redis.exists.return_value = 0

        result = await cache_manager.exists("test_key")

        assert result is False

    @pytest.mark.asyncio
    async def test_close_connection(self, cache_manager):
        """Test closing Redis connection."""
        mock_redis = AsyncMock()
        cache_manager.redis = mock_redis

        await cache_manager.close()

        mock_redis.aclose.assert_called_once()
        assert cache_manager.redis is None


class TestCacheMetrics:
    """Test suite for CacheMetrics."""

    def test_metrics_initialization(self):
        """Test that metrics initialize with zero values."""
        metrics = CacheMetrics()

        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.sets == 0
        assert metrics.deletes == 0
        assert metrics.errors == 0

    def test_hit_ratio_with_no_operations(self):
        """Test hit ratio calculation with no operations."""
        metrics = CacheMetrics()

        assert metrics.hit_ratio == 0.0

    def test_hit_ratio_calculation(self):
        """Test hit ratio calculation with operations."""
        metrics = CacheMetrics()
        metrics.record_hit()
        metrics.record_hit()
        metrics.record_miss()

        assert metrics.hit_ratio == 2 / 3

    def test_record_operations(self):
        """Test recording different cache operations."""
        metrics = CacheMetrics()

        metrics.record_hit()
        metrics.record_miss()
        metrics.record_set()
        metrics.record_delete()
        metrics.record_error()

        assert metrics.hits == 1
        assert metrics.misses == 1
        assert metrics.sets == 1
        assert metrics.deletes == 1
        assert metrics.errors == 1

    def test_reset_metrics(self):
        """Test resetting metrics to zero."""
        metrics = CacheMetrics()
        metrics.record_hit()
        metrics.record_miss()
        metrics.record_set()

        metrics.reset()

        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.sets == 0


class TestCacheStrategy:
    """Test suite for CacheStrategy enum and utilities."""

    def test_cache_strategy_values(self):
        """Test that CacheStrategy has expected values."""
        assert CacheStrategy.LLM_RESPONSE.value == "llm_response"
        assert CacheStrategy.DATABASE_QUERY.value == "database_query"
        assert CacheStrategy.RAG_EMBEDDING.value == "rag_embedding"
        assert CacheStrategy.RAG_RETRIEVAL.value == "rag_retrieval"

    def test_get_ttl_for_strategy(self):
        """Test TTL calculation based on strategy."""
        from src.agent.adapters.cache import get_ttl_for_strategy

        # LLM responses should have longer TTL
        assert (
            get_ttl_for_strategy(CacheStrategy.LLM_RESPONSE, complexity="simple")
            == 3600
        )
        assert (
            get_ttl_for_strategy(CacheStrategy.LLM_RESPONSE, complexity="complex")
            == 1800
        )

        # Database queries should have medium TTL
        assert get_ttl_for_strategy(CacheStrategy.DATABASE_QUERY) == 1800

        # RAG operations should have shorter TTL
        assert get_ttl_for_strategy(CacheStrategy.RAG_EMBEDDING) == 7200
        assert get_ttl_for_strategy(CacheStrategy.RAG_RETRIEVAL) == 900
