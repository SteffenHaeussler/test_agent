import asyncio
from abc import ABC
from typing import Any, Dict, Optional

import instructor
from langfuse import get_client, observe
from litellm import completion, acompletion
from loguru import logger
from pydantic import BaseModel

from src.agent.exceptions import LLMAPIException
from src.agent.observability.context import ctx_query_id
from src.agent.adapters.cache import CacheManager, CacheStrategy, get_ttl_for_strategy


class AbstractLLM(ABC):
    """
    AbstractLLM is an abstract base class for all LLM models.

    Methods:
        - use(self, question: str, response_model: BaseModel) -> BaseModel: Use the LLM model (sync).
        - use_async(self, question: str, response_model: BaseModel) -> BaseModel: Use the LLM model (async).
        - health_check(self) -> bool: Check if the LLM service is healthy (async).
    """

    def __init__(self):
        """
        Initialize the LLM model.

        Args:
            kwargs: Dict: The kwargs.

        Returns:
            None
        """
        pass

    def __enter__(self):
        """Enter the synchronous context manager."""
        # For backward compatibility, provide a synchronous interface
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the synchronous context manager."""
        pass

    async def __aenter__(self):
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager."""
        pass

    def use(self, question: str, response_model: BaseModel) -> BaseModel:
        """
        Calls the LLM model with the given question and response model (synchronous).

        Args:
            question: str: The question to use the LLM model.
            response_model: BaseModel: The response model.

        Returns:
            response: BaseModel: The response from the LLM model.
        """
        pass

    async def use_async(self, question: str, response_model: BaseModel) -> BaseModel:
        """
        Calls the LLM model with the given question and response model (asynchronous).

        Args:
            question: str: The question to use the LLM model.
            response_model: BaseModel: The response model.

        Returns:
            response: BaseModel: The response from the LLM model.
        """
        pass

    async def health_check(self) -> bool:
        """
        Check if the LLM service is healthy.

        Returns:
            bool: True if healthy, False otherwise
        """
        pass


class LLM(AbstractLLM):
    """
    Async-first LLM adapter with retry logic, timeout handling, and backward compatibility.

    Features:
    - Async LLM API calls with acompletion
    - Exponential backoff retry logic
    - Timeout handling for LLM requests
    - Connection pooling support
    - Health check functionality
    - Backward compatibility with sync wrapper
    """

    def __init__(self, kwargs: Dict[str, Any]):
        """
        Initialize the LLM model with async support.

        Args:
            kwargs: Dict: Configuration parameters including:
                - model_id: The LLM model identifier
                - temperature: Temperature for generation
                - timeout: Request timeout in seconds (default: 60)
                - max_retries: Maximum retry attempts (default: 3)
                - base_delay: Base delay for exponential backoff (default: 1.0)
                - max_delay: Maximum delay for exponential backoff (default: 60.0)
                - cache_manager: Optional CacheManager instance for caching
        """
        super().__init__()
        self.model_id = kwargs["model_id"]
        self.temperature = float(kwargs["temperature"])

        # Async configuration
        self.timeout = kwargs.get("timeout", 60.0)
        self.max_retries = kwargs.get("max_retries", 3)
        self.base_delay = kwargs.get("base_delay", 1.0)
        self.max_delay = kwargs.get("max_delay", 60.0)

        # Cache configuration
        self.cache_manager: Optional[CacheManager] = kwargs.get("cache_manager")
        self.cache_enabled = kwargs.get("cache_enabled", True)

        # Initialize both sync and async clients
        self.client = self.init_llm()
        self.async_client = self.init_async_llm()

    def init_llm(self):
        """
        Initialize the synchronous LLM model.

        Returns:
            client: instructor.from_litellm: The synchronous LLM model.
        """
        client = instructor.from_litellm(completion)
        return client

    def init_async_llm(self):
        """
        Initialize the asynchronous LLM model.

        Returns:
            client: instructor.from_litellm: The asynchronous LLM model.
        """
        client = instructor.from_litellm(acompletion)
        return client

    @observe(as_type="generation")
    def use(self, question: str, response_model: BaseModel) -> BaseModel:
        """
        Synchronous wrapper for backward compatibility.

        This method runs the async use_async in a new event loop.
        """
        import asyncio

        async def _run_llm_call():
            return await self.use_async(question, response_model)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If there's already a running loop, we need to use a thread
                import concurrent.futures

                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(_run_llm_call())
                    finally:
                        # Clean up any pending tasks before closing the loop
                        pending = asyncio.all_tasks(new_loop)
                        for task in pending:
                            task.cancel()
                        if pending:
                            new_loop.run_until_complete(
                                asyncio.gather(*pending, return_exceptions=True)
                            )
                        new_loop.close()
                        asyncio.set_event_loop(None)

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(run_in_thread).result()
            else:
                return loop.run_until_complete(_run_llm_call())
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_run_llm_call())
            finally:
                # Clean up any pending tasks before closing the loop
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.close()
                asyncio.set_event_loop(None)

    @observe(as_type="generation")
    async def use_async(self, question: str, response_model: BaseModel) -> BaseModel:
        """
        Calls the LLM model asynchronously with retry logic and timeout handling.

        Args:
            question: str: The question to use the LLM model.
            response_model: BaseModel: The response model.

        Returns:
            response: BaseModel: The response from the LLM model.

        Raises:
            LLMAPIException: If the LLM API call fails after all retries.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]

        langfuse = get_client()

        langfuse.update_current_trace(
            name="llm_call",
            input=messages.copy(),
            metadata={"temperature": self.temperature, "model": self.model_id},
            session_id=ctx_query_id.get(),
        )

        retry_count = 0
        last_exception = None

        while retry_count <= self.max_retries:
            try:
                # Execute LLM call with timeout
                response = await asyncio.wait_for(
                    self._make_llm_call_async(messages, response_model),
                    timeout=self.timeout,
                )

                logger.debug(f"LLM call successful after {retry_count} retries")
                return response

            except asyncio.TimeoutError as e:
                last_exception = e
                retry_count += 1

                if retry_count <= self.max_retries:
                    delay = min(
                        self.base_delay * (2 ** (retry_count - 1)), self.max_delay
                    )
                    logger.warning(
                        f"LLM call timeout (attempt {retry_count}/{self.max_retries + 1}): {e}. "
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
                        f"LLM call failed (attempt {retry_count}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(f"LLM call failed after {retry_count} attempts")
        context = {
            "model_id": self.model_id,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout,
            "retry_count": retry_count,
            "max_retries": self.max_retries,
            "messages": messages,
            "operation": "llm_call",
        }

        raise LLMAPIException(
            f"LLM API call failed after {retry_count} attempts: {last_exception}",
            context=context,
            original_exception=last_exception,
        )

    async def _make_llm_call_async(
        self, messages: list, response_model: BaseModel
    ) -> BaseModel:
        """
        Make the actual async LLM API call.

        Args:
            messages: List of message dictionaries
            response_model: Pydantic model for response

        Returns:
            BaseModel: The response from the LLM
        """
        response = await self.async_client.chat.completions.create(
            messages=messages,
            response_model=response_model,
            model=self.model_id,
            temperature=self.temperature,
        )
        return response

    async def health_check(self) -> bool:
        """
        Check if the LLM service is healthy by making a simple test call.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            from pydantic import BaseModel

            class HealthResponse(BaseModel):
                status: str

            # Make a simple test call with 10-second timeout
            await asyncio.wait_for(
                self._make_llm_call_async(
                    [{"role": "user", "content": "Say 'healthy' if you can respond."}],
                    HealthResponse,
                ),
                timeout=10.0,
            )

            logger.debug("LLM health check passed")
            return True

        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            return False

    # Cache-enabled methods

    async def use_cached_async(
        self, question: str, response_model: BaseModel, **kwargs
    ) -> BaseModel:
        """
        Use LLM with caching support.

        Args:
            question: The question to ask the LLM
            response_model: Pydantic model for response structure
            **kwargs: Additional parameters for cache key generation

        Returns:
            BaseModel: The response from LLM or cache
        """
        # Check if caching should be used
        if not self._should_use_cache():
            return await self.use_async(question, response_model)

        # Generate cache key
        cache_key = self._generate_cache_key(question, response_model, kwargs)

        # Try to get from cache
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result is not None:
            logger.debug(f"LLM cache hit for key: {cache_key}")
            # Reconstruct the response model from cached data
            return response_model(**cached_result)

        # Cache miss - call LLM and cache result
        logger.debug(f"LLM cache miss for key: {cache_key}")
        result = await self.use_async(question, response_model)

        # Cache the result
        await self._cache_result(cache_key, result)

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

    def _generate_cache_key(
        self, question: str, response_model: BaseModel, extra_params: Dict[str, Any]
    ) -> str:
        """
        Generate a cache key for the LLM request.

        Args:
            question: The question being asked
            response_model: The response model type
            extra_params: Additional parameters

        Returns:
            str: The generated cache key
        """
        key_params = {
            "question": question,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "response_model": response_model.__name__,
            **extra_params,
        }

        return self.cache_manager.generate_cache_key("llm_response", **key_params)

    async def _cache_result(self, cache_key: str, result: BaseModel) -> None:
        """
        Cache the LLM result.

        Args:
            cache_key: The cache key
            result: The result to cache
        """
        try:
            # Determine complexity for TTL calculation
            complexity = "complex" if len(str(result)) > 1000 else "simple"
            ttl = get_ttl_for_strategy(CacheStrategy.LLM_RESPONSE, complexity)

            # Convert result to dict for JSON serialization
            result_dict = (
                result.model_dump() if hasattr(result, "model_dump") else result.dict()
            )

            await self.cache_manager.set(cache_key, result_dict, ttl)
            logger.debug(f"Cached LLM result with TTL {ttl}s")

        except Exception as e:
            logger.error(f"Failed to cache LLM result: {e}")

    async def invalidate_cache_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern.

        Args:
            pattern: Redis pattern to match

        Returns:
            int: Number of keys deleted
        """
        if not self._should_use_cache():
            return 0

        return await self.cache_manager.delete_pattern(pattern)
