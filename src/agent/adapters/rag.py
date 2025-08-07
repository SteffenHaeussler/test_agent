import asyncio
from abc import ABC
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from src.agent.exceptions import RAGSystemException
from src.agent.adapters.cache import CacheManager, CacheStrategy, get_ttl_for_strategy


class AbstractModel(ABC):
    """
    AbstractModel is an abstract base class for all RAG models.

    Methods:
        - embed(self, text: str) -> Dict[str, List[float]]: Embed the text (sync).
        - embed_async(self, text: str) -> Dict[str, List[float]]: Embed the text (async).
        - rerank(self, question: str, text: str) -> Dict[str, str]: Rerank the text (sync).
        - rerank_async(self, question: str, text: str) -> Dict[str, str]: Rerank the text (async).
        - retrieve(self, embedding: list[float]) -> Dict[str, List[str]]: Retrieve the text (sync).
        - retrieve_async(self, embedding: list[float]) -> Dict[str, List[str]]: Retrieve the text (async).
        - health_check(self) -> bool: Check if the RAG service is healthy (async).
    """

    def __init__(self) -> None:
        pass

    def __enter__(self) -> "AbstractModel":
        """Enter the synchronous context manager."""
        # For backward compatibility, provide a synchronous interface
        return self

    def __exit__(
        self, exc_type: Optional[Any], exc_val: Optional[Any], exc_tb: Optional[Any]
    ) -> None:
        """Exit the synchronous context manager."""
        pass

    async def __aenter__(self) -> "AbstractModel":
        """Enter the async context manager."""
        return self

    async def __aexit__(
        self, exc_type: Optional[Any], exc_val: Optional[Any], exc_tb: Optional[Any]
    ) -> None:
        """Exit the async context manager."""
        pass

    def embed(self, text: str) -> Optional[Dict[str, List[float]]]:
        pass

    async def embed_async(self, text: str) -> Optional[Dict[str, List[float]]]:
        pass

    def rerank(self, question: str, text: str) -> Optional[Dict[str, Any]]:
        pass

    async def rerank_async(self, question: str, text: str) -> Optional[Dict[str, Any]]:
        pass

    def retrieve(self, embedding: List[float]) -> Optional[Dict[str, Any]]:
        pass

    async def retrieve_async(self, embedding: List[float]) -> Optional[Dict[str, Any]]:
        pass

    async def health_check(self) -> bool:
        """
        Check if the RAG service is healthy.

        Returns:
            bool: True if healthy, False otherwise
        """
        pass


class BaseRAG(AbstractModel):
    """
    Async-first RAG adapter with retry logic, timeout handling, and backward compatibility.

    Features:
    - Async HTTP API calls with httpx.AsyncClient
    - Exponential backoff retry logic
    - Timeout handling for HTTP requests
    - Connection pooling support
    - Health check functionality
    - Backward compatibility with sync wrapper

    Methods:
        - embed(self, text: str) -> Dict[str, List[float]]: Embed the text (sync wrapper).
        - embed_async(self, text: str) -> Dict[str, List[float]]: Embed the text (async).
        - rerank(self, question: str, text: str) -> Dict[str, str]: Rerank the text (sync wrapper).
        - rerank_async(self, question: str, text: str) -> Dict[str, str]: Rerank the text (async).
        - retrieve(self, embedding: list[float]) -> Dict[str, List[str]]: Retrieve the text (sync wrapper).
        - retrieve_async(self, embedding: list[float]) -> Dict[str, List[str]]: Retrieve the text (async).
        - call_api(self, api_url, body={}, method="get") -> None: Call the API (sync wrapper).
        - call_api_async(self, api_url, body={}, method="get") -> None: Call the API (async).
        - health_check(self) -> bool: Check if the RAG service is healthy (async).
    """

    def __init__(self, kwargs: Dict[str, Any]) -> None:
        """
        Initialize the BaseRAG model with async support.

        Args:
            kwargs: Dict[str, Any]: Configuration parameters including:
                - embedding_url: URL for embedding API
                - ranking_url: URL for ranking API
                - retrieval_url: URL for retrieval API
                - n_retrieval_candidates: Number of retrieval candidates
                - n_ranking_candidates: Number of ranking candidates
                - retrieval_table: Table name for retrieval
                - timeout: Request timeout in seconds (default: 30.0)
                - max_retries: Maximum retry attempts (default: 3)
                - base_delay: Base delay for exponential backoff (default: 1.0)
                - max_delay: Maximum delay for exponential backoff (default: 60.0)
                - max_connections: Max connections in pool (default: 10)
                - max_keepalive_connections: Max keepalive connections (default: 5)
        """
        super().__init__()
        self.kwargs = kwargs

        # API endpoints
        self.embedding_url = kwargs["embedding_url"]
        self.n_retrieval_candidates = int(kwargs["n_retrieval_candidates"])
        self.n_ranking_candidates = int(kwargs["n_ranking_candidates"])
        self.ranking_url = kwargs["ranking_url"]
        self.retrieval_url = kwargs["retrieval_url"]
        self.retrieval_table = kwargs["retrieval_table"]

        # Async configuration
        self.timeout = kwargs.get("timeout", 30.0)
        self.max_retries = kwargs.get("max_retries", 3)
        self.base_delay = kwargs.get("base_delay", 1.0)
        self.max_delay = kwargs.get("max_delay", 60.0)
        self.max_connections = kwargs.get("max_connections", 10)
        self.max_keepalive_connections = kwargs.get("max_keepalive_connections", 5)

        # HTTP client will be initialized on demand
        self._client: Optional[httpx.AsyncClient] = None

        # Cache configuration
        self.cache_manager: Optional[CacheManager] = kwargs.get("cache_manager")
        self.cache_enabled = kwargs.get("cache_enabled", True)

    async def __aenter__(self) -> "BaseRAG":
        """Enter the async context manager and initialize HTTP client."""
        await self._ensure_client()
        return self

    async def __aexit__(
        self, exc_type: Optional[Any], exc_val: Optional[Any], exc_tb: Optional[Any]
    ) -> None:
        """Exit the async context manager and cleanup HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self) -> None:
        """Ensure async HTTP client is initialized."""
        if self._client is None:
            # Create async client with connection pooling
            limits = httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_keepalive_connections,
            )
            self._client = httpx.AsyncClient(
                limits=limits,
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
            )
            logger.debug("Initialized async HTTP client with connection pooling")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get the async HTTP client, initializing if necessary."""
        await self._ensure_client()
        return self._client

    def call_api(
        self, api_url: str, body: Dict[str, Any] = {}, method: str = "get"
    ) -> Optional[httpx.Response]:
        """
        Synchronous wrapper for backward compatibility.

        This method runs the async call_api_async in a new event loop.

        Args:
            api_url: str: The API URL.
            body: Dict: The body of the request.
            method: str: The method of the request.

        Returns:
            response: httpx.Response: The response from the API.
        """
        try:
            return self._run_async_method(self.call_api_async(api_url, body, method))
        except Exception as e:
            # For backward compatibility with old direct call_api usage, return None on any error
            logger.debug(f"API call to {api_url} failed: {e}")
            return None

    async def call_api_async(
        self, api_url: str, body: Dict[str, Any] = {}, method: str = "get"
    ) -> Optional[httpx.Response]:
        """
        Calls the RAG API asynchronously with retry logic and timeout handling.

        Args:
            api_url: str: The API URL.
            body: Dict: The body of the request.
            method: str: The method of the request.

        Returns:
            response: httpx.Response: The response from the API.

        Raises:
            RAGSystemException: If the API call fails after all retries.
        """
        client = await self._get_client()
        retry_count = 0
        last_exception = None

        while retry_count <= self.max_retries:
            try:
                # Execute API call with timeout
                if method == "get":
                    response = await client.get(api_url, params=body)
                elif method == "post":
                    response = await client.post(api_url, json=body)
                else:
                    raise ValueError("Invalid method")

                response.raise_for_status()
                logger.debug(
                    f"RAG API call to {api_url} successful after {retry_count} retries"
                )
                return response

            except httpx.TimeoutException as e:
                last_exception = e
                retry_count += 1

                if retry_count <= self.max_retries:
                    delay = min(
                        self.base_delay * (2 ** (retry_count - 1)), self.max_delay
                    )
                    logger.warning(
                        f"RAG API call timeout (attempt {retry_count}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    await asyncio.sleep(delay)

            except httpx.HTTPStatusError as e:
                last_exception = e
                retry_count += 1

                if retry_count <= self.max_retries:
                    delay = min(
                        self.base_delay * (2 ** (retry_count - 1)), self.max_delay
                    )
                    logger.warning(
                        f"RAG API call HTTP error (attempt {retry_count}/{self.max_retries + 1}): "
                        f"{e.response.status_code} - {e.response.text}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    await asyncio.sleep(delay)

            except httpx.RequestError as e:
                last_exception = e
                retry_count += 1

                if retry_count <= self.max_retries:
                    delay = min(
                        self.base_delay * (2 ** (retry_count - 1)), self.max_delay
                    )
                    logger.warning(
                        f"RAG API call request error (attempt {retry_count}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    await asyncio.sleep(delay)

            except Exception as e:
                last_exception = e
                retry_count += 1

                if retry_count <= self.max_retries:
                    delay = min(
                        self.base_delay * (2 ** (retry_count - 1)), self.max_delay
                    )
                    logger.warning(
                        f"RAG API call unexpected error (attempt {retry_count}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(f"RAG API call to {api_url} failed after {retry_count} attempts")
        context = {
            "api_url": api_url,
            "method": method,
            "timeout_seconds": self.timeout,
            "retry_count": retry_count,
            "max_retries": self.max_retries,
            "operation": "api_call",
        }

        raise RAGSystemException(
            f"RAG API call failed after {retry_count} attempts: {last_exception}",
            context=context,
            original_exception=last_exception,
        )

    def embed(self, text: str) -> Optional[Dict[str, List[float]]]:
        """
        Synchronous wrapper for backward compatibility.

        This method runs the async embed_async in a new event loop.

        Args:
            text: str: The text to embed.

        Returns:
            Optional[Dict[str, List[float]]]: The response from the embedding API, or None if failed.
        """
        try:
            return self._run_async_method(self.embed_async(text))
        except RAGSystemException:
            # For backward compatibility, return None on errors after all retries exhausted
            return None
        except Exception as e:
            # For any other unexpected errors, also return None for backward compatibility
            logger.debug(f"Embed call failed: {e}")
            return None

    def rerank(self, question: str, text: str) -> Optional[Dict[str, Any]]:
        """
        Synchronous wrapper for backward compatibility.

        This method runs the async rerank_async in a new event loop.

        Args:
            question: str: The question to rerank the text.
            text: str: The text to rerank.

        Returns:
            Optional[Dict[str, Any]]: The response from the ranking API, or None if failed.
        """
        try:
            return self._run_async_method(self.rerank_async(question, text))
        except RAGSystemException:
            # For backward compatibility, return None on errors after all retries exhausted
            return None
        except Exception as e:
            # For any other unexpected errors, also return None for backward compatibility
            logger.debug(f"Rerank call failed: {e}")
            return None

    def retrieve(self, embedding: List[float]) -> Optional[Dict[str, Any]]:
        """
        Synchronous wrapper for backward compatibility.

        This method runs the async retrieve_async in a new event loop.

        Args:
            embedding: List[float]: The embedding to retrieve the text.

        Returns:
            Optional[Dict[str, Any]]: The response from the retrieval API, or None if failed.
        """
        try:
            return self._run_async_method(self.retrieve_async(embedding))
        except RAGSystemException:
            # For backward compatibility, return None on errors after all retries exhausted
            return None
        except Exception as e:
            # For any other unexpected errors, also return None for backward compatibility
            logger.debug(f"Retrieve call failed: {e}")
            return None

    def _run_async_method(self, coro: Any) -> Any:
        """
        Run an async method in a new event loop for backward compatibility.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If there's already a running loop, we need to use a thread
                import concurrent.futures

                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(coro)
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(run_in_thread).result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    async def embed_async(self, text: str) -> Optional[Dict[str, List[float]]]:
        """
        Embed the text asynchronously.

        Args:
            text: str: The text to embed.

        Returns:
            Optional[Dict[str, List[float]]]: The response from the embedding API, or None if failed.
        """
        response = await self.call_api_async(self.embedding_url, {"text": text})
        return response.json() if response else None

    async def rerank_async(self, question: str, text: str) -> Optional[Dict[str, Any]]:
        """
        Rerank the text asynchronously.

        Args:
            question: str: The question to rerank the text.
            text: str: The text to rerank.

        Returns:
            Optional[Dict[str, Any]]: The response from the ranking API, or None if failed.
        """
        response = await self.call_api_async(
            self.ranking_url,
            {"text": text, "question": question, "table": self.retrieval_table},
        )
        return response.json() if response else None

    async def retrieve_async(self, embedding: List[float]) -> Optional[Dict[str, Any]]:
        """
        Retrieve the text asynchronously.

        Args:
            embedding: List[float]: The embedding to retrieve the text.

        Returns:
            Optional[Dict[str, Any]]: The response from the retrieval API, or None if failed.
        """
        response = await self.call_api_async(
            self.retrieval_url,
            {
                "embedding": embedding,
                "n_items": self.n_retrieval_candidates,
                "table": self.retrieval_table,
            },
            method="post",
        )
        return response.json() if response else None

    async def health_check(self) -> bool:
        """
        Check if the RAG service is healthy by making test calls to all endpoints.

        Returns:
            bool: True if all services are healthy, False otherwise
        """
        try:
            # Test embedding endpoint
            embedding_response = await self.call_api_async(
                self.embedding_url, {"text": "health check"}
            )
            if not embedding_response:
                logger.warning(
                    "RAG health check failed: embedding endpoint unavailable"
                )
                return False

            # Test retrieval endpoint with simple embedding
            retrieval_response = await self.call_api_async(
                self.retrieval_url,
                {
                    "embedding": [0.1, 0.2, 0.3],
                    "n_items": 1,
                    "table": self.retrieval_table,
                },
                method="post",
            )
            if not retrieval_response:
                logger.warning(
                    "RAG health check failed: retrieval endpoint unavailable"
                )
                return False

            # Test ranking endpoint
            ranking_response = await self.call_api_async(
                self.ranking_url,
                {
                    "text": "test text",
                    "question": "test question",
                    "table": self.retrieval_table,
                },
            )
            if not ranking_response:
                logger.warning("RAG health check failed: ranking endpoint unavailable")
                return False

            logger.debug("RAG health check passed for all endpoints")
            return True

        except Exception as e:
            logger.warning(f"RAG health check failed with exception: {e}")
            return False

    # Cache-enabled methods

    async def embed_cached_async(self, text: str) -> Optional[Dict[str, List[float]]]:
        """
        Embed text with caching support.

        Args:
            text: Text to embed

        Returns:
            Embedding result or None if failed
        """
        # Check if caching should be used
        if not self._should_use_cache():
            return await self.embed_async(text)

        # Generate cache key
        cache_key = self._generate_embedding_cache_key(text)

        # Try to get from cache
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result is not None:
            logger.debug(f"RAG embedding cache hit for key: {cache_key}")
            return cached_result

        # Cache miss - call API and cache result
        logger.debug(f"RAG embedding cache miss for key: {cache_key}")
        result = await self.embed_async(text)

        if result is not None:
            # Cache the result
            await self._cache_embedding_result(cache_key, result)

        return result

    async def retrieve_cached_async(
        self, embedding: List[float]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve with caching support.

        Args:
            embedding: Embedding vector

        Returns:
            Retrieval result or None if failed
        """
        # Check if caching should be used
        if not self._should_use_cache():
            return await self.retrieve_async(embedding)

        # Generate cache key
        cache_key = self._generate_retrieval_cache_key(embedding)

        # Try to get from cache
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result is not None:
            logger.debug(f"RAG retrieval cache hit for key: {cache_key}")
            return cached_result

        # Cache miss - call API and cache result
        logger.debug(f"RAG retrieval cache miss for key: {cache_key}")
        result = await self.retrieve_async(embedding)

        if result is not None:
            # Cache the result
            await self._cache_retrieval_result(cache_key, result)

        return result

    async def rerank_cached_async(
        self, question: str, text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Rerank with caching support.

        Args:
            question: Question for reranking
            text: Text to rerank

        Returns:
            Rerank result or None if failed
        """
        # Check if caching should be used
        if not self._should_use_cache():
            return await self.rerank_async(question, text)

        # Generate cache key
        cache_key = self._generate_rerank_cache_key(question, text)

        # Try to get from cache
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result is not None:
            logger.debug(f"RAG rerank cache hit for key: {cache_key}")
            return cached_result

        # Cache miss - call API and cache result
        logger.debug(f"RAG rerank cache miss for key: {cache_key}")
        result = await self.rerank_async(question, text)

        if result is not None:
            # Cache the result
            await self._cache_rerank_result(cache_key, result)

        return result

    def _should_use_cache(self) -> bool:
        """
        Determine if caching should be used for this request.

        Returns:
            bool: True if caching should be used
        """
        return (
            self.cache_enabled
            and self.cache_manager is not None
            and self.cache_manager.enabled
        )

    def _generate_embedding_cache_key(self, text: str) -> str:
        """
        Generate cache key for embedding requests.

        Args:
            text: Text to embed

        Returns:
            Cache key string
        """
        key_params = {
            "text": text,
            "embedding_url": self.embedding_url,
            "model": "embedding",
        }

        return self.cache_manager.generate_cache_key("rag_embedding", **key_params)

    def _generate_retrieval_cache_key(self, embedding: List[float]) -> str:
        """
        Generate cache key for retrieval requests.

        Args:
            embedding: Embedding vector

        Returns:
            Cache key string
        """
        # Convert embedding to string for consistent hashing
        embedding_str = ",".join(
            [f"{x:.6f}" for x in embedding[:10]]
        )  # Use first 10 values

        key_params = {
            "embedding_prefix": embedding_str,
            "n_items": self.n_retrieval_candidates,
            "table": self.retrieval_table,
            "retrieval_url": self.retrieval_url,
        }

        return self.cache_manager.generate_cache_key("rag_retrieval", **key_params)

    def _generate_rerank_cache_key(self, question: str, text: str) -> str:
        """
        Generate cache key for rerank requests.

        Args:
            question: Question for reranking
            text: Text to rerank

        Returns:
            Cache key string
        """
        key_params = {
            "question": question,
            "text": text,
            "table": self.retrieval_table,
            "ranking_url": self.ranking_url,
        }

        return self.cache_manager.generate_cache_key("rag_rerank", **key_params)

    async def _cache_embedding_result(
        self, cache_key: str, result: Dict[str, List[float]]
    ) -> None:
        """
        Cache embedding result.

        Args:
            cache_key: Cache key
            result: Embedding result to cache
        """
        try:
            ttl = get_ttl_for_strategy(CacheStrategy.RAG_EMBEDDING)
            await self.cache_manager.set(cache_key, result, ttl)
            logger.debug(f"Cached RAG embedding result with TTL {ttl}s")

        except Exception as e:
            logger.error(f"Failed to cache RAG embedding result: {e}")

    async def _cache_retrieval_result(
        self, cache_key: str, result: Dict[str, Any]
    ) -> None:
        """
        Cache retrieval result.

        Args:
            cache_key: Cache key
            result: Retrieval result to cache
        """
        try:
            ttl = get_ttl_for_strategy(CacheStrategy.RAG_RETRIEVAL)
            await self.cache_manager.set(cache_key, result, ttl)
            logger.debug(f"Cached RAG retrieval result with TTL {ttl}s")

        except Exception as e:
            logger.error(f"Failed to cache RAG retrieval result: {e}")

    async def _cache_rerank_result(
        self, cache_key: str, result: Dict[str, Any]
    ) -> None:
        """
        Cache rerank result.

        Args:
            cache_key: Cache key
            result: Rerank result to cache
        """
        try:
            # Use retrieval TTL for rerank results
            ttl = get_ttl_for_strategy(CacheStrategy.RAG_RETRIEVAL)
            await self.cache_manager.set(cache_key, result, ttl)
            logger.debug(f"Cached RAG rerank result with TTL {ttl}s")

        except Exception as e:
            logger.error(f"Failed to cache RAG rerank result: {e}")

    async def invalidate_cache_pattern(self, pattern: str) -> int:
        """
        Invalidate RAG cache entries matching a pattern.

        Args:
            pattern: Redis pattern to match

        Returns:
            int: Number of keys deleted
        """
        if not self._should_use_cache():
            return 0

        return await self.cache_manager.delete_pattern(pattern)
