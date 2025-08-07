"""
Redis-based caching layer for the agentic AI framework.

This module provides a comprehensive caching solution with:
- Async Redis client with connection pooling
- Different caching strategies for various use cases
- Cache key generation with query parameters and model configurations
- Time-based and event-based invalidation
- Performance metrics and monitoring
- Graceful fallback when Redis is unavailable
"""

import asyncio
import hashlib
import json
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import redis.asyncio as redis
import redis.exceptions
from loguru import logger


class CacheStrategy(Enum):
    """Enumeration of different caching strategies."""

    LLM_RESPONSE = "llm_response"
    DATABASE_QUERY = "database_query"
    RAG_EMBEDDING = "rag_embedding"
    RAG_RETRIEVAL = "rag_retrieval"


@dataclass
class CacheMetrics:
    """Tracks cache performance metrics."""

    hits: int = field(default=0)
    misses: int = field(default=0)
    sets: int = field(default=0)
    deletes: int = field(default=0)
    errors: int = field(default=0)

    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total_reads = self.hits + self.misses
        if total_reads == 0:
            return 0.0
        return self.hits / total_reads

    def record_hit(self):
        """Record a cache hit."""
        self.hits += 1

    def record_miss(self):
        """Record a cache miss."""
        self.misses += 1

    def record_set(self):
        """Record a cache set operation."""
        self.sets += 1

    def record_delete(self):
        """Record a cache delete operation."""
        self.deletes += 1

    def record_error(self):
        """Record a cache error."""
        self.errors += 1

    def reset(self):
        """Reset all metrics to zero."""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0


def get_ttl_for_strategy(
    strategy: CacheStrategy, complexity: Optional[str] = None
) -> int:
    """
    Get TTL (time-to-live) in seconds based on caching strategy.

    Args:
        strategy: The caching strategy
        complexity: Optional complexity indicator for LLM responses

    Returns:
        TTL in seconds
    """
    if strategy == CacheStrategy.LLM_RESPONSE:
        if complexity == "complex":
            return 1800  # 30 minutes for complex queries
        return 3600  # 1 hour for simple queries
    elif strategy == CacheStrategy.DATABASE_QUERY:
        return 1800  # 30 minutes for database queries
    elif strategy == CacheStrategy.RAG_EMBEDDING:
        return 7200  # 2 hours for embeddings (relatively stable)
    elif strategy == CacheStrategy.RAG_RETRIEVAL:
        return 900  # 15 minutes for retrieval results
    else:
        return 600  # Default 10 minutes


class CacheManager:
    """
    Redis-based cache manager with async support and comprehensive features.

    Features:
    - Async Redis operations with connection pooling
    - Automatic JSON serialization/deserialization
    - Cache key generation with consistent hashing
    - Performance metrics tracking
    - Graceful degradation when Redis is unavailable
    - Pattern-based cache invalidation
    - TTL support for time-based expiration
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cache manager.

        Args:
            config: Redis configuration dictionary
        """
        self.config = config
        self.redis: Optional[redis.Redis] = None
        self.enabled = True
        self.metrics = CacheMetrics()

    async def initialize(self) -> None:
        """
        Initialize Redis connection with error handling.

        If connection fails, caching will be disabled but the application
        will continue to function normally.
        """
        try:
            # Build Redis URL from config
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 6379)
            db = self.config.get("db", 0)
            password = self.config.get("password")

            url = f"redis://{host}:{port}/{db}"
            if password:
                url = f"redis://:{password}@{host}:{port}/{db}"

            # Create Redis client with connection pooling
            self.redis = redis.from_url(
                url,
                max_connections=self.config.get("max_connections", 10),
                decode_responses=self.config.get("decode_responses", True),
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30,
            )

            # Test connection
            await self.redis.ping()
            logger.info("Redis cache initialized successfully")

        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            logger.warning(f"Failed to connect to Redis, disabling cache: {e}")
            self.enabled = False
            self.redis = None
        except Exception as e:
            logger.error(f"Unexpected error initializing Redis cache: {e}")
            self.enabled = False
            self.redis = None

    def generate_cache_key(self, prefix: str, **kwargs) -> str:
        """
        Generate a consistent cache key from parameters.

        Args:
            prefix: Key prefix to categorize cache entries
            **kwargs: Parameters to include in key generation

        Returns:
            Consistent cache key string
        """
        # Sort parameters for consistent key generation
        key_data = {k: v for k, v in sorted(kwargs.items())}

        # Create deterministic hash from parameters
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()

        return f"{prefix}:{key_hash}"

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/error
        """
        if not self.enabled or not self.redis:
            return None

        try:
            value = await self.redis.get(key)
            if value is not None:
                self.metrics.record_hit()
                return json.loads(value)
            else:
                self.metrics.record_miss()
                return None

        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.metrics.record_error()
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis:
            return False

        try:
            serialized_value = json.dumps(value)

            if ttl:
                await self.redis.setex(key, ttl, serialized_value)
            else:
                await self.redis.set(key, serialized_value)

            self.metrics.record_set()
            return True

        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self.metrics.record_error()
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False if key didn't exist
        """
        if not self.enabled or not self.redis:
            return False

        try:
            result = await self.redis.delete(key)
            self.metrics.record_delete()
            return result > 0

        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            self.metrics.record_error()
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.

        Args:
            pattern: Redis pattern (e.g., "user:*")

        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.redis:
            return 0

        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self.redis.delete(*keys)
                self.metrics.record_delete()
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Cache delete pattern error for pattern {pattern}: {e}")
            self.metrics.record_error()
            return 0

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists, False otherwise
        """
        if not self.enabled or not self.redis:
            return False

        try:
            result = await self.redis.exists(key)
            return result > 0

        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            self.metrics.record_error()
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.aclose()
            self.redis = None
            logger.info("Redis cache connection closed")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.

        Returns:
            Dictionary containing cache metrics
        """
        return {
            "enabled": self.enabled,
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "sets": self.metrics.sets,
            "deletes": self.metrics.deletes,
            "errors": self.metrics.errors,
            "hit_ratio": self.metrics.hit_ratio,
        }

    def reset_metrics(self) -> None:
        """Reset all cache metrics."""
        self.metrics.reset()


def cache_decorator(
    cache_manager: CacheManager, strategy: CacheStrategy, key_prefix: str
):
    """
    Decorator for caching function results.

    Args:
        cache_manager: CacheManager instance
        strategy: Caching strategy to determine TTL
        key_prefix: Prefix for cache keys

    Returns:
        Decorator function
    """

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            cache_key = cache_manager.generate_cache_key(
                key_prefix, func_name=func.__name__, args=args, kwargs=kwargs
            )

            # Try to get from cache first
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            ttl = get_ttl_for_strategy(strategy)
            await cache_manager.set(cache_key, result, ttl)

            return result

        def sync_wrapper(*args, **kwargs):
            # For sync functions, use asyncio.run for cache operations
            async def sync_cache_wrapper():
                cache_key = cache_manager.generate_cache_key(
                    key_prefix, func_name=func.__name__, args=args, kwargs=kwargs
                )

                cached_result = await cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result

                result = func(*args, **kwargs)
                ttl = get_ttl_for_strategy(strategy)
                await cache_manager.set(cache_key, result, ttl)

                return result

            return asyncio.run(sync_cache_wrapper())

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@dataclass
class CacheInvalidationEvent:
    """
    Event object for cache invalidation.
    """

    event_type: str
    affected_tables: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class CacheInvalidator:
    """
    Handles cache invalidation strategies and events.
    """

    def __init__(self, cache_manager: CacheManager):
        """
        Initialize cache invalidator.

        Args:
            cache_manager: The cache manager instance
        """
        self.cache_manager = cache_manager

    async def handle_event(self, event: CacheInvalidationEvent) -> None:
        """
        Handle cache invalidation event.

        Args:
            event: The invalidation event
        """
        logger.info(f"Processing cache invalidation event: {event.event_type}")

        for pattern in event.patterns:
            deleted_count = await self.cache_manager.delete_pattern(pattern)
            logger.debug(
                f"Invalidated {deleted_count} cache entries matching pattern: {pattern}"
            )

    async def invalidate_llm_cache(self, model_id: Optional[str] = None) -> int:
        """
        Invalidate LLM cache entries.

        Args:
            model_id: Optional specific model ID to invalidate

        Returns:
            Number of cache entries deleted
        """
        if model_id:
            pattern = f"llm_response:*{model_id}*"
        else:
            pattern = "llm_response:*"

        deleted_count = await self.cache_manager.delete_pattern(pattern)
        logger.info(f"Invalidated {deleted_count} LLM cache entries")
        return deleted_count

    async def invalidate_database_cache(self, table: Optional[str] = None) -> int:
        """
        Invalidate database cache entries.

        Args:
            table: Optional specific table name to invalidate

        Returns:
            Number of cache entries deleted
        """
        if table:
            pattern = f"database_query:*{table}*"
        else:
            pattern = "database_query:*"

        deleted_count = await self.cache_manager.delete_pattern(pattern)
        logger.info(f"Invalidated {deleted_count} database cache entries")
        return deleted_count

    async def invalidate_rag_cache(self, cache_type: Optional[str] = None) -> int:
        """
        Invalidate RAG cache entries.

        Args:
            cache_type: Optional cache type ('embedding', 'retrieval', 'rerank')

        Returns:
            Number of cache entries deleted
        """
        if cache_type:
            pattern = f"rag_{cache_type}:*"
        else:
            pattern = "rag_*:*"

        deleted_count = await self.cache_manager.delete_pattern(pattern)
        logger.info(f"Invalidated {deleted_count} RAG cache entries")
        return deleted_count

    async def invalidate_all_cache(self) -> int:
        """
        Invalidate all cache entries.

        Returns:
            Number of cache entries deleted
        """
        deleted_count = await self.cache_manager.delete_pattern("*")
        logger.warning(f"Invalidated ALL {deleted_count} cache entries")
        return deleted_count
