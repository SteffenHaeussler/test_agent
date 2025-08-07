import pytest
from unittest.mock import AsyncMock, patch

from src.agent.adapters.cache import CacheStrategy
from src.agent.adapters.llm import LLM
from src.agent.adapters.database import BaseDatabaseAdapter
from src.agent.adapters.rag import BaseRAG
from src.agent.domain.commands import LLMResponseModel


class TestLLMCacheIntegration:
    """Test suite for LLM adapter cache integration."""

    @pytest.fixture
    def cache_manager(self):
        """Mock cache manager for testing."""
        mock_cache = AsyncMock()
        mock_cache.enabled = True
        mock_cache.generate_cache_key.return_value = "test_cache_key"
        return mock_cache

    @pytest.fixture
    def llm_config(self):
        """Configuration for LLM."""
        return {
            "model_id": "gpt-4",
            "temperature": 0.7,
            "timeout": 30.0,
            "max_retries": 2,
        }

    @pytest.mark.asyncio
    async def test_llm_cache_hit_async(self, cache_manager, llm_config):
        """Test LLM cache hit - should return cached result without calling LLM."""
        # Arrange
        cache_manager.get.return_value = {
            "response": "Cached response",
            "chain_of_thought": "Cached thinking",
        }

        llm = LLM(llm_config)
        llm.cache_manager = cache_manager

        # Mock the LLM call to ensure it's not called
        llm._make_llm_call_async = AsyncMock()

        # Act
        with patch.object(llm, "_should_use_cache", return_value=True):
            result = await llm.use_cached_async("test question", LLMResponseModel)

        # Assert
        cache_manager.get.assert_called_once()
        llm._make_llm_call_async.assert_not_called()
        assert result.response == "Cached response"

    @pytest.mark.asyncio
    async def test_llm_cache_miss_async(self, cache_manager, llm_config):
        """Test LLM cache miss - should call LLM and cache result."""
        # Arrange
        cache_manager.get.return_value = None
        llm_response = LLMResponseModel(
            response="Fresh response", chain_of_thought="Fresh thinking"
        )

        llm = LLM(llm_config)
        llm.cache_manager = cache_manager
        llm._make_llm_call_async = AsyncMock(return_value=llm_response)

        # Act
        with patch.object(llm, "_should_use_cache", return_value=True):
            result = await llm.use_cached_async("test question", LLMResponseModel)

        # Assert
        cache_manager.get.assert_called_once()
        llm._make_llm_call_async.assert_called_once()
        cache_manager.set.assert_called_once()
        assert result.response == "Fresh response"

    @pytest.mark.asyncio
    async def test_llm_cache_disabled(self, llm_config):
        """Test LLM operation when cache is disabled."""
        # Arrange
        cache_manager = AsyncMock()
        cache_manager.enabled = False

        llm_response = LLMResponseModel(
            response="Direct response", chain_of_thought="Direct thinking"
        )

        llm = LLM(llm_config)
        llm.cache_manager = cache_manager
        llm._make_llm_call_async = AsyncMock(return_value=llm_response)

        # Act
        with patch.object(llm, "_should_use_cache", return_value=False):
            await llm.use_cached_async("test question", LLMResponseModel)

        # Assert
        cache_manager.get.assert_not_called()
        llm._make_llm_call_async.assert_called_once()
        cache_manager.set.assert_not_called()

    def test_llm_cache_key_generation(self, cache_manager, llm_config):
        """Test cache key generation includes relevant parameters."""
        # Arrange
        llm = LLM(llm_config)
        llm.cache_manager = cache_manager

        # Act
        llm._generate_cache_key("test question", LLMResponseModel, {"extra": "param"})

        # Assert
        cache_manager.generate_cache_key.assert_called_once()
        call_args = cache_manager.generate_cache_key.call_args

        # Should include model parameters and question
        assert "llm_response" in call_args[0]  # prefix
        assert "question" in call_args[1]
        assert "model_id" in call_args[1]
        assert "temperature" in call_args[1]


class TestDatabaseCacheIntegration:
    """Test suite for database adapter cache integration."""

    @pytest.fixture
    def cache_manager(self):
        """Mock cache manager for testing."""
        mock_cache = AsyncMock()
        mock_cache.enabled = True
        mock_cache.generate_cache_key.return_value = "test_db_cache_key"
        return mock_cache

    @pytest.fixture
    def db_config(self):
        """Configuration for database."""
        return {
            "host": "localhost",
            "port": 5432,
            "database": "testdb",
            "username": "testuser",
            "password": "testpass",
        }

    @pytest.mark.asyncio
    async def test_database_query_cache_hit(self, cache_manager, db_config):
        """Test database query cache hit - should return cached result."""
        # Arrange
        cached_result = [{"id": 1, "name": "cached"}]
        cache_manager.get.return_value = cached_result

        db = BaseDatabaseAdapter(db_config)
        db.cache_manager = cache_manager

        # Mock the actual database query
        db._execute_query_async = AsyncMock()

        # Act
        with patch.object(db, "_should_use_cache", return_value=True):
            result = await db.execute_cached_query_async("SELECT * FROM users")

        # Assert
        cache_manager.get.assert_called_once()
        db._execute_query_async.assert_not_called()
        assert result == cached_result

    @pytest.mark.asyncio
    async def test_database_query_cache_miss(self, cache_manager, db_config):
        """Test database query cache miss - should execute query and cache result."""
        # Arrange
        cache_manager.get.return_value = None
        query_result = {"data": [{"id": 1, "name": "fresh"}]}

        db = BaseDatabaseAdapter(db_config)
        db.cache_manager = cache_manager

        # Mock the execute_query_async method
        with patch.object(
            db, "execute_query_async", return_value=query_result
        ) as mock_execute:
            with patch.object(db, "_should_use_cache", return_value=True):
                result = await db.execute_cached_query_async("SELECT * FROM users")

        # Assert
        cache_manager.get.assert_called_once()
        mock_execute.assert_called_once_with("SELECT * FROM users", None)
        cache_manager.set.assert_called_once()
        assert result == query_result

    @pytest.mark.asyncio
    async def test_database_cache_invalidation_on_write(self, cache_manager, db_config):
        """Test cache invalidation when performing write operations."""
        # Arrange
        db = BaseDatabaseAdapter(db_config)
        db.cache_manager = cache_manager
        query_result = {"data": []}

        # Mock the execute_query_async method
        with patch.object(
            db, "execute_query_async", return_value=query_result
        ) as mock_execute:
            # Act
            await db.execute_write_query_async("UPDATE users SET name = 'updated'")

        # Assert - Should execute query and invalidate cache
        mock_execute.assert_called_once_with("UPDATE users SET name = 'updated'", None)
        cache_manager.delete_pattern.assert_called()
        # Pattern should include table name
        call_args = cache_manager.delete_pattern.call_args[0][0]
        assert "users" in call_args

    def test_database_cache_key_includes_query_params(self, cache_manager, db_config):
        """Test database cache key generation includes query and parameters."""
        # Arrange
        db = BaseDatabaseAdapter(db_config)
        db.cache_manager = cache_manager

        # Act
        db._generate_cache_key("SELECT * FROM users WHERE id = ?", [123])

        # Assert
        cache_manager.generate_cache_key.assert_called_once()
        call_args = cache_manager.generate_cache_key.call_args

        assert "database_query" in call_args[0]  # prefix
        assert "query" in call_args[1]
        assert "params" in call_args[1]


class TestRAGCacheIntegration:
    """Test suite for RAG adapter cache integration."""

    @pytest.fixture
    def cache_manager(self):
        """Mock cache manager for testing."""
        mock_cache = AsyncMock()
        mock_cache.enabled = True
        mock_cache.generate_cache_key.return_value = "test_rag_cache_key"
        return mock_cache

    @pytest.fixture
    def rag_config(self):
        """Configuration for RAG."""
        return {
            "embedding_url": "http://localhost:8080/embed",
            "ranking_url": "http://localhost:8080/rank",
            "retrieval_url": "http://localhost:8080/retrieve",
            "n_retrieval_candidates": 20,
            "n_ranking_candidates": 10,
            "retrieval_table": "test_table",
        }

    @pytest.mark.asyncio
    async def test_rag_embedding_cache_hit(self, cache_manager, rag_config):
        """Test RAG embedding cache hit - should return cached embedding."""
        # Arrange
        cached_embedding = {"embedding": [0.1, 0.2, 0.3]}
        cache_manager.get.return_value = cached_embedding

        rag = BaseRAG(rag_config)
        rag.cache_manager = cache_manager

        # Mock the embed_async method
        with patch.object(rag, "embed_async", return_value=None) as mock_embed:
            with patch.object(rag, "_should_use_cache", return_value=True):
                result = await rag.embed_cached_async("test text")

        # Assert
        cache_manager.get.assert_called_once()
        mock_embed.assert_not_called()
        assert result == cached_embedding

    @pytest.mark.asyncio
    async def test_rag_embedding_cache_miss(self, cache_manager, rag_config):
        """Test RAG embedding cache miss - should compute and cache embedding."""
        # Arrange
        cache_manager.get.return_value = None
        fresh_embedding = {"embedding": [0.4, 0.5, 0.6]}

        rag = BaseRAG(rag_config)
        rag.cache_manager = cache_manager

        # Mock the embed_async method
        with patch.object(
            rag, "embed_async", return_value=fresh_embedding
        ) as mock_embed:
            with patch.object(rag, "_should_use_cache", return_value=True):
                result = await rag.embed_cached_async("test text")

        # Assert
        cache_manager.get.assert_called_once()
        mock_embed.assert_called_once()
        cache_manager.set.assert_called_once()
        assert result == fresh_embedding

    @pytest.mark.asyncio
    async def test_rag_retrieval_cache_hit(self, cache_manager, rag_config):
        """Test RAG retrieval cache hit - should return cached results."""
        # Arrange
        cached_results = {
            "results": [{"id": "doc1", "content": "cached content", "score": 0.9}]
        }
        cache_manager.get.return_value = cached_results

        rag = BaseRAG(rag_config)
        rag.cache_manager = cache_manager

        # Act
        embedding = [0.1, 0.2, 0.3]
        with patch.object(rag, "retrieve_async", return_value=None) as mock_retrieve:
            with patch.object(rag, "_should_use_cache", return_value=True):
                result = await rag.retrieve_cached_async(embedding)

        # Assert
        cache_manager.get.assert_called_once()
        mock_retrieve.assert_not_called()
        assert result == cached_results

    @pytest.mark.asyncio
    async def test_rag_rerank_cache_handling(self, cache_manager, rag_config):
        """Test RAG reranking with cache - should cache rerank results."""
        # Arrange
        cache_manager.get.return_value = None
        rerank_result = {"score": 0.85, "reasoning": "relevant match"}

        rag = BaseRAG(rag_config)
        rag.cache_manager = cache_manager

        # Act
        with patch.object(
            rag, "rerank_async", return_value=rerank_result
        ) as mock_rerank:
            with patch.object(rag, "_should_use_cache", return_value=True):
                result = await rag.rerank_cached_async("question", "document content")

        # Assert
        cache_manager.get.assert_called_once()
        mock_rerank.assert_called_once()
        cache_manager.set.assert_called_once()
        assert result == rerank_result

    def test_rag_cache_key_generation(self, cache_manager, rag_config):
        """Test RAG cache key generation includes relevant parameters."""
        # Arrange
        rag = BaseRAG(rag_config)
        rag.cache_manager = cache_manager

        # Test embedding cache key
        rag._generate_embedding_cache_key("test text")
        cache_manager.generate_cache_key.assert_called()

        # Test retrieval cache key
        rag._generate_retrieval_cache_key([0.1, 0.2, 0.3])

        # Should have been called multiple times
        assert cache_manager.generate_cache_key.call_count >= 2


class TestCacheInvalidationStrategies:
    """Test suite for cache invalidation strategies."""

    @pytest.fixture
    def cache_manager(self):
        """Mock cache manager for testing."""
        mock_cache = AsyncMock()
        mock_cache.enabled = True
        return mock_cache

    @pytest.mark.asyncio
    async def test_time_based_cache_expiration(self, cache_manager):
        """Test that cache entries expire based on TTL."""
        # This would be handled by Redis TTL, but we test the TTL calculation
        from src.agent.adapters.cache import get_ttl_for_strategy

        # Different strategies should have different TTLs
        llm_ttl = get_ttl_for_strategy(CacheStrategy.LLM_RESPONSE)
        db_ttl = get_ttl_for_strategy(CacheStrategy.DATABASE_QUERY)
        rag_ttl = get_ttl_for_strategy(CacheStrategy.RAG_EMBEDDING)

        assert llm_ttl > 0
        assert db_ttl > 0
        assert rag_ttl > 0

        # LLM responses should have longer TTL than retrieval
        assert llm_ttl > get_ttl_for_strategy(CacheStrategy.RAG_RETRIEVAL)

    @pytest.mark.asyncio
    async def test_event_based_cache_invalidation(self, cache_manager):
        """Test event-based cache invalidation on data updates."""
        # Simulate database update event
        from src.agent.adapters.cache import CacheInvalidationEvent

        event = CacheInvalidationEvent(
            event_type="database_update",
            affected_tables=["users", "orders"],
            patterns=["database_query:*users*", "database_query:*orders*"],
        )

        # Mock event handler
        invalidator = CacheInvalidator(cache_manager)
        await invalidator.handle_event(event)

        # Should call delete_pattern for each affected pattern
        assert cache_manager.delete_pattern.call_count == len(event.patterns)

    @pytest.mark.asyncio
    async def test_manual_cache_invalidation(self, cache_manager):
        """Test manual cache invalidation endpoints."""
        from src.agent.adapters.cache import CacheInvalidator

        invalidator = CacheInvalidator(cache_manager)

        # Test invalidating all LLM caches
        await invalidator.invalidate_llm_cache()
        cache_manager.delete_pattern.assert_called_with("llm_response:*")

        # Test invalidating specific database table caches
        await invalidator.invalidate_database_cache(table="users")
        cache_manager.delete_pattern.assert_called_with("database_query:*users*")


# Mock classes for testing
class CacheInvalidationEvent:
    def __init__(self, event_type, affected_tables, patterns):
        self.event_type = event_type
        self.affected_tables = affected_tables
        self.patterns = patterns


class CacheInvalidator:
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager

    async def handle_event(self, event):
        for pattern in event.patterns:
            await self.cache_manager.delete_pattern(pattern)

    async def invalidate_llm_cache(self):
        await self.cache_manager.delete_pattern("llm_response:*")

    async def invalidate_database_cache(self, table=None):
        if table:
            await self.cache_manager.delete_pattern(f"database_query:*{table}*")
        else:
            await self.cache_manager.delete_pattern("database_query:*")
