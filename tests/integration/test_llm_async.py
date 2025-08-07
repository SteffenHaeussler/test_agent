import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.agent.adapters.llm import LLM
from src.agent.domain.commands import LLMResponseModel
from src.agent.exceptions import LLMAPIException


class TestLLMAsync:
    """Test suite for the async LLM adapter."""

    @pytest.fixture
    def llm_config(self):
        """Standard LLM configuration for tests."""
        return {
            "model_id": "gpt-3.5-turbo",
            "temperature": 0.5,
            "timeout": 30.0,
            "max_retries": 2,
            "base_delay": 0.1,
            "max_delay": 1.0,
        }

    @pytest.fixture
    def mock_response(self):
        """Mock LLM response."""
        return LLMResponseModel(
            response="Test async response",
            chain_of_thought="Test async chain of thought",
        )

    def test_llm_init(self, llm_config):
        """Test LLM initialization with async configuration."""
        llm = LLM(llm_config)

        assert llm.model_id == "gpt-3.5-turbo"
        assert llm.temperature == 0.5
        assert llm.timeout == 30.0
        assert llm.max_retries == 2
        assert llm.base_delay == 0.1
        assert llm.max_delay == 1.0
        assert llm.client is not None
        assert llm.async_client is not None

    def test_llm_init_with_defaults(self):
        """Test LLM initialization with default async values."""
        llm = LLM({"model_id": "gpt-4", "temperature": 0.7})

        assert llm.timeout == 60.0
        assert llm.max_retries == 3
        assert llm.base_delay == 1.0
        assert llm.max_delay == 60.0

    @pytest.mark.asyncio
    async def test_use_async_success(self, llm_config, mock_response):
        """Test successful async LLM call."""
        with patch(
            "src.agent.adapters.llm.instructor.from_litellm"
        ) as mock_from_litellm:
            # Mock the async client
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_from_litellm.return_value = mock_client

            llm = LLM(llm_config)
            llm.async_client = mock_client

            question = "What is async programming?"
            response = await llm.use_async(question, LLMResponseModel)

            assert response.response == "Test async response"
            assert response.chain_of_thought == "Test async chain of thought"

            # Verify the async call was made
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-3.5-turbo"
            assert call_args[1]["temperature"] == 0.5
            assert len(call_args[1]["messages"]) == 2

    @pytest.mark.asyncio
    async def test_use_async_with_timeout(self, llm_config):
        """Test async LLM call that times out."""
        with patch(
            "src.agent.adapters.llm.instructor.from_litellm"
        ) as mock_from_litellm:
            # Mock the async client to simulate timeout
            mock_client = AsyncMock()

            # Since our test optimization makes sleeps instant,
            # we need to directly raise TimeoutError to simulate a timeout
            async def timeout_response(*args, **kwargs):
                raise asyncio.TimeoutError("Operation timed out")

            mock_client.chat.completions.create = timeout_response
            mock_from_litellm.return_value = mock_client

            # Short timeout for testing
            llm_config["timeout"] = 0.1
            llm_config["max_retries"] = 1
            llm_config["base_delay"] = 0.01

            llm = LLM(llm_config)
            llm.async_client = mock_client

            question = "What is async programming?"

            with pytest.raises(LLMAPIException) as exc_info:
                await llm.use_async(question, LLMResponseModel)

            assert "LLM API call failed after" in str(exc_info.value)
            assert exc_info.value.context["timeout_seconds"] == 0.1
            assert exc_info.value.context["max_retries"] == 1

    @pytest.mark.asyncio
    async def test_use_async_with_retries(self, llm_config, mock_response):
        """Test async LLM call with retry logic."""
        with patch(
            "src.agent.adapters.llm.instructor.from_litellm"
        ) as mock_from_litellm:
            # Mock the async client to fail once then succeed
            mock_client = AsyncMock()
            call_count = 0

            async def mock_call(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("First call fails")
                return mock_response

            mock_client.chat.completions.create = mock_call
            mock_from_litellm.return_value = mock_client

            # Fast retries for testing
            llm_config["max_retries"] = 2
            llm_config["base_delay"] = 0.01

            llm = LLM(llm_config)
            llm.async_client = mock_client

            question = "What is async programming?"
            response = await llm.use_async(question, LLMResponseModel)

            assert response.response == "Test async response"
            assert call_count == 2  # First call failed, second succeeded

    @pytest.mark.asyncio
    async def test_use_async_retry_exhaustion(self, llm_config):
        """Test async LLM call that exhausts all retries."""
        with patch(
            "src.agent.adapters.llm.instructor.from_litellm"
        ) as mock_from_litellm:
            # Mock the async client to always fail
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Always fails")
            )
            mock_from_litellm.return_value = mock_client

            # Fast retries for testing
            llm_config["max_retries"] = 1
            llm_config["base_delay"] = 0.01

            llm = LLM(llm_config)
            llm.async_client = mock_client

            question = "What is async programming?"

            with pytest.raises(LLMAPIException) as exc_info:
                await llm.use_async(question, LLMResponseModel)

            assert "LLM API call failed after 2 attempts" in str(exc_info.value)
            assert exc_info.value.context["retry_count"] == 2
            assert exc_info.value.context["max_retries"] == 1

    @pytest.mark.asyncio
    async def test_health_check_success(self, llm_config):
        """Test successful health check."""
        with patch(
            "src.agent.adapters.llm.instructor.from_litellm"
        ) as mock_from_litellm:
            # Mock the async client for health check
            mock_client = AsyncMock()
            mock_health_response = Mock()
            mock_health_response.status = "healthy"
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_health_response
            )
            mock_from_litellm.return_value = mock_client

            llm = LLM(llm_config)
            llm.async_client = mock_client

            is_healthy = await llm.health_check()

            assert is_healthy is True
            mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, llm_config):
        """Test health check failure."""
        with patch(
            "src.agent.adapters.llm.instructor.from_litellm"
        ) as mock_from_litellm:
            # Mock the async client to fail
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Health check failed")
            )
            mock_from_litellm.return_value = mock_client

            llm = LLM(llm_config)
            llm.async_client = mock_client

            is_healthy = await llm.health_check()

            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, llm_config):
        """Test health check timeout."""
        with patch(
            "src.agent.adapters.llm.instructor.from_litellm"
        ) as mock_from_litellm:
            # Mock the async client to be slow
            mock_client = AsyncMock()

            async def slow_health_check(*args, **kwargs):
                # Instead of actually waiting, just raise TimeoutError
                import asyncio

                raise asyncio.TimeoutError("Health check timed out")

            mock_client.chat.completions.create = slow_health_check
            mock_from_litellm.return_value = mock_client

            # Override config to have shorter timeout for test
            llm_config["health_check_timeout"] = 0.1  # 100ms timeout
            llm = LLM(llm_config)
            llm.async_client = mock_client

            is_healthy = await llm.health_check()

            assert is_healthy is False

    def test_sync_use_backward_compatibility(self, llm_config, mock_response):
        """Test backward compatibility of sync use method."""
        with patch(
            "src.agent.adapters.llm.instructor.from_litellm"
        ) as mock_from_litellm:
            # Mock both sync and async clients
            mock_sync_client = Mock()
            mock_sync_client.chat.completions.create.return_value = mock_response

            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            def mock_from_litellm_side_effect(completion_func):
                if asyncio.iscoroutinefunction(completion_func):
                    return mock_async_client
                else:
                    return mock_sync_client

            mock_from_litellm.side_effect = mock_from_litellm_side_effect

            llm = LLM(llm_config)

            question = "What is async programming?"
            response = llm.use(question, LLMResponseModel)

            assert response.response == "Test async response"
            assert response.chain_of_thought == "Test async chain of thought"

    @pytest.mark.asyncio
    async def test_context_manager_async(self, llm_config):
        """Test async context manager support."""
        llm = LLM(llm_config)

        async with llm as llm_context:
            assert llm_context is llm

    def test_context_manager_sync(self, llm_config):
        """Test sync context manager support for backward compatibility."""
        llm = LLM(llm_config)

        with llm as llm_context:
            assert llm_context is llm

    @pytest.mark.asyncio
    async def test_exponential_backoff_calculation(self, llm_config):
        """Test that exponential backoff delays are calculated correctly."""
        with patch(
            "src.agent.adapters.llm.instructor.from_litellm"
        ) as mock_from_litellm:
            with patch("asyncio.sleep") as mock_sleep:
                # Mock the async client to always fail
                mock_client = AsyncMock()
                mock_client.chat.completions.create = AsyncMock(
                    side_effect=Exception("Always fails")
                )
                mock_from_litellm.return_value = mock_client

                # Set specific values for backoff testing
                llm_config.update(
                    {
                        "max_retries": 3,
                        "base_delay": 0.5,
                        "max_delay": 5.0,
                    }
                )

                llm = LLM(llm_config)
                llm.async_client = mock_client

                question = "What is async programming?"

                with pytest.raises(LLMAPIException):
                    await llm.use_async(question, LLMResponseModel)

                # Verify exponential backoff delays: 0.5, 1.0, 2.0
                expected_delays = [0.5, 1.0, 2.0]
                actual_calls = [call[0][0] for call in mock_sleep.call_args_list]

                assert len(actual_calls) == 3
                for expected, actual in zip(expected_delays, actual_calls):
                    assert (
                        abs(actual - expected) < 0.01
                    )  # Small tolerance for floating point

    @pytest.mark.asyncio
    async def test_max_delay_cap(self, llm_config):
        """Test that delays are capped at max_delay."""
        with patch(
            "src.agent.adapters.llm.instructor.from_litellm"
        ) as mock_from_litellm:
            with patch("asyncio.sleep") as mock_sleep:
                # Mock the async client to always fail
                mock_client = AsyncMock()
                mock_client.chat.completions.create = AsyncMock(
                    side_effect=Exception("Always fails")
                )
                mock_from_litellm.return_value = mock_client

                # Set values where exponential backoff would exceed max_delay
                llm_config.update(
                    {
                        "max_retries": 5,
                        "base_delay": 2.0,
                        "max_delay": 3.0,  # Cap at 3 seconds
                    }
                )

                llm = LLM(llm_config)
                llm.async_client = mock_client

                question = "What is async programming?"

                with pytest.raises(LLMAPIException):
                    await llm.use_async(question, LLMResponseModel)

                # Verify delays are capped: 2.0, 3.0 (capped), 3.0 (capped), etc.
                actual_calls = [call[0][0] for call in mock_sleep.call_args_list]

                # Should have delays: 2.0, 3.0 (capped), 3.0 (capped), 3.0 (capped), 3.0 (capped)
                expected_delays = [2.0, 3.0, 3.0, 3.0, 3.0]

                assert len(actual_calls) == 5
                for expected, actual in zip(expected_delays, actual_calls):
                    assert (
                        abs(actual - expected) < 0.01
                    )  # Small tolerance for floating point

    @pytest.mark.asyncio
    async def test_langfuse_integration(self, llm_config, mock_response):
        """Test that Langfuse integration works with async calls."""
        with patch(
            "src.agent.adapters.llm.instructor.from_litellm"
        ) as mock_from_litellm:
            with patch("src.agent.adapters.llm.get_client") as mock_get_client:
                # Mock the async client
                mock_client = AsyncMock()
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_response
                )
                mock_from_litellm.return_value = mock_client

                # Mock Langfuse client
                mock_langfuse = Mock()
                mock_get_client.return_value = mock_langfuse

                llm = LLM(llm_config)
                llm.async_client = mock_client

                question = "What is async programming?"
                response = await llm.use_async(question, LLMResponseModel)

                assert response.response == "Test async response"

                # Verify Langfuse was called
                mock_get_client.assert_called_once()
                mock_langfuse.update_current_trace.assert_called_once()

                # Check trace metadata
                trace_call = mock_langfuse.update_current_trace.call_args
                metadata = trace_call[1]["metadata"]
                assert metadata["temperature"] == 0.5
                assert metadata["model"] == "gpt-3.5-turbo"
