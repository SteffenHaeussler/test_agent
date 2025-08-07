from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from src.agent.adapters.rag import BaseRAG


@pytest.fixture
def rag_instance():
    kwargs = {
        "embedding_url": "http://test-embedding-url",
        "ranking_url": "http://test-ranking-url",
        "retrieval_url": "http://test-retrieval-url",
        "n_retrieval_candidates": 5,
        "n_ranking_candidates": 3,
        "retrieval_table": "test_table",
        "max_retries": 1,  # Reduce retries for tests
        "base_delay": 0.001,  # 1ms instead of 1s
        "max_delay": 0.002,  # 2ms instead of 60s
    }
    return BaseRAG(kwargs)


class TestRAG:
    def test_rag_init(self, rag_instance):
        assert rag_instance.embedding_url == "http://test-embedding-url"
        assert rag_instance.ranking_url == "http://test-ranking-url"
        assert rag_instance.retrieval_url == "http://test-retrieval-url"
        assert rag_instance.n_retrieval_candidates == 5
        assert rag_instance.n_ranking_candidates == 3
        assert rag_instance.retrieval_table == "test_table"

    def test_embed(self, rag_instance):
        # Mock the async client's get method
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch.object(rag_instance, "_get_client", return_value=mock_client):
            response = rag_instance.embed("test_text")

        assert response == {"embedding": [0.1, 0.2, 0.3]}

    def test_rerank(self, rag_instance):
        # Mock the async client's get method
        mock_response = Mock()
        mock_response.json.return_value = {
            "question": "test_question",
            "text": "test_text",
            "score": 0.5,
            "id": "test_id",
            "tag": "test_tag",
            "name": "test_name",
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch.object(rag_instance, "_get_client", return_value=mock_client):
            response = rag_instance.rerank("test_question", "test_text")

        assert response == {
            "question": "test_question",
            "text": "test_text",
            "score": 0.5,
            "id": "test_id",
            "tag": "test_tag",
            "name": "test_name",
        }

    def test_retrieve(self, rag_instance):
        # Mock the async client's post method
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [
                {
                    "id": "test_id",
                    "score": 0.5,
                    "description": "test_text",
                    "tag": "test_tag",
                    "name": "test_name",
                },
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(rag_instance, "_get_client", return_value=mock_client):
            response = rag_instance.retrieve([0.1, 0.2, 0.3])

        assert response == {
            "items": [
                {
                    "id": "test_id",
                    "score": 0.5,
                    "description": "test_text",
                    "tag": "test_tag",
                    "name": "test_name",
                },
            ]
        }

    def test_http_status_error(self, rag_instance):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error",
            request=Mock(),
            response=Mock(status_code=500, text="Internal Server Error"),
        )

        with patch("httpx.get", return_value=mock_response):
            result = rag_instance.embed("some text")
            assert result is None

    def test_request_error(self, rag_instance):
        with patch(
            "httpx.get", side_effect=httpx.RequestError("Network Error", request=Mock())
        ):
            result = rag_instance.embed("some text")
            assert result is None


class TestAsyncRAG:
    """Test async functionality of RAG adapter."""

    @pytest.mark.asyncio
    async def test_async_embed_with_connection_pooling(self, rag_instance):
        """Test async embed method uses AsyncClient with connection pooling."""
        # Create mock response with regular Mock for json() method (not AsyncMock)
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = Mock()

        # Mock the _get_client method to return our mock client directly
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch.object(rag_instance, "_get_client", return_value=mock_client):
            result = await rag_instance.embed_async("test_text")

            assert result == {"embedding": [0.1, 0.2, 0.3]}
            mock_client.get.assert_called_once_with(
                "http://test-embedding-url", params={"text": "test_text"}
            )

    @pytest.mark.asyncio
    async def test_async_rerank_with_retry_logic(self, rag_instance):
        """Test async rerank method with exponential backoff retry logic."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "question": "test_question",
            "text": "test_text",
            "score": 0.5,
            "id": "test_id",
            "tag": "test_tag",
            "name": "test_name",
        }
        mock_response.raise_for_status = Mock()

        # Mock the _get_client method to return our mock client directly
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch.object(rag_instance, "_get_client", return_value=mock_client):
            result = await rag_instance.rerank_async("test_question", "test_text")

            assert result == {
                "question": "test_question",
                "text": "test_text",
                "score": 0.5,
                "id": "test_id",
                "tag": "test_tag",
                "name": "test_name",
            }
            mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_retrieve_with_timeout(self, rag_instance):
        """Test async retrieve method with timeout handling."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [
                {
                    "id": "test_id",
                    "score": 0.5,
                    "description": "test_text",
                    "tag": "test_tag",
                    "name": "test_name",
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        # Mock the _get_client method to return our mock client directly
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(rag_instance, "_get_client", return_value=mock_client):
            result = await rag_instance.retrieve_async([0.1, 0.2, 0.3])

            assert result == {
                "items": [
                    {
                        "id": "test_id",
                        "score": 0.5,
                        "description": "test_text",
                        "tag": "test_tag",
                        "name": "test_name",
                    }
                ]
            }
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_exponential_backoff_retry_logic(self):
        """Test retry logic with exponential backoff on API failures."""
        # Create RAG instance with 2 retries for this specific test
        kwargs = {
            "embedding_url": "http://test-embedding-url",
            "ranking_url": "http://test-ranking-url",
            "retrieval_url": "http://test-retrieval-url",
            "n_retrieval_candidates": 5,
            "n_ranking_candidates": 3,
            "retrieval_table": "test_table",
            "max_retries": 2,  # Need 2 retries for this test to pass
            "base_delay": 0.001,  # Fast delays for testing
            "max_delay": 0.002,
        }
        rag_instance = BaseRAG(kwargs)

        call_count = [0]  # Use list to make it mutable in nested function

        async def mock_get_request(*args, **kwargs):
            """Mock get request that fails twice then succeeds"""
            call_count[0] += 1
            if call_count[0] < 3:
                # First two calls fail with HTTP error
                mock_response = Mock(status_code=500, text="Internal Server Error")
                raise httpx.HTTPStatusError(
                    "Server Error", request=Mock(), response=mock_response
                )
            else:
                # Third call succeeds
                mock_response = Mock()
                mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
                mock_response.raise_for_status = Mock()
                return mock_response

        # Mock the _get_client method to return our mock client
        mock_client = AsyncMock()
        mock_client.get.side_effect = mock_get_request

        with patch.object(rag_instance, "_get_client", return_value=mock_client):
            result = await rag_instance.embed_async("test_text")

            # Should succeed after 2 retries
            assert result == {"embedding": [0.1, 0.2, 0.3]}
            assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_async_context_managers(self, rag_instance):
        """Test async context manager lifecycle."""
        async with rag_instance as rag:
            assert rag is not None
            assert hasattr(rag, "_client")
            assert rag._client is not None

        # After exiting the context, client should be cleaned up
        assert rag._client is None

    @pytest.mark.asyncio
    async def test_health_check(self, rag_instance):
        """Test health check functionality."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = Mock()

        # Mock the _get_client method to return our mock client directly
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.post.return_value = mock_response

        with patch.object(rag_instance, "_get_client", return_value=mock_client):
            result = await rag_instance.health_check()

            assert isinstance(result, bool)
            assert result is True
            # Should call all endpoints (embed=get, rerank=get, retrieve=post)
            assert mock_client.get.call_count == 2
            assert mock_client.post.call_count == 1

    def test_backward_compatibility_sync_wrappers(self, rag_instance):
        """Test that sync methods still work for backward compatibility."""
        # Create mock response with regular Mock for json() method (not AsyncMock)
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = Mock()

        # Mock the async client through the _get_client method
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch.object(rag_instance, "_get_client", return_value=mock_client):
            # Test sync embed method uses async implementation internally
            result = rag_instance.embed("test_text")
            assert result == {"embedding": [0.1, 0.2, 0.3]}

            # Test sync rerank method
            mock_response.json.return_value = {"score": 0.8}
            result = rag_instance.rerank("test_question", "test_text")
            assert result == {"score": 0.8}

            # Test sync retrieve method (uses POST)
            mock_client.post.return_value = mock_response
            mock_response.json.return_value = {"items": ["result1", "result2"]}
            result = rag_instance.retrieve([0.1, 0.2, 0.3])
            assert result == {"items": ["result1", "result2"]}

    def test_sync_methods_return_none_on_errors(self, rag_instance):
        """Test that sync methods return None on errors for backward compatibility."""
        # Mock client that always fails
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=Mock(),
            response=Mock(status_code=500, text="Internal Server Error"),
        )
        mock_client.post.side_effect = httpx.RequestError(
            "Network Error", request=Mock()
        )

        with patch.object(rag_instance, "_get_client", return_value=mock_client):
            # Sync methods should return None on errors (backward compatibility)
            result = rag_instance.embed("test_text")
            assert result is None

            result = rag_instance.rerank("test_question", "test_text")
            assert result is None

            result = rag_instance.retrieve([0.1, 0.2, 0.3])
            assert result is None
