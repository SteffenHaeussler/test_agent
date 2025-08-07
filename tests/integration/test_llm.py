from unittest.mock import patch, AsyncMock

from instructor.client import Instructor

from src.agent.adapters.llm import LLM
from src.agent.domain.commands import LLMResponseModel


class TestLLM:
    def test_llm_init(self):
        llm = LLM(
            {
                "model_id": "1",
                "temperature": 0.5,
            },
        )

        assert llm.model_id == "1"
        assert llm.temperature == 0.5
        assert type(llm.client) is Instructor

    @patch("src.agent.adapters.llm.instructor.from_litellm")
    def test_llm_call(self, mock_from_litellm):
        # Mock response
        mock_response = LLMResponseModel(
            response="Test response",
            chain_of_thought="Test chain of thought",
        )

        # Since the sync use() method now calls async use_async(),
        # we need to mock the async client properly
        mock_async_client = AsyncMock()
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        mock_sync_client = mock_from_litellm.return_value
        mock_sync_client.chat.completions.create.return_value = mock_response

        llm = LLM(
            {
                "model_id": "1",
                "temperature": 0.5,
            },
        )

        # Replace the async client with our mock for the test
        llm.async_client = mock_async_client

        question = "What is the capital of France?"
        response = llm.use(question, LLMResponseModel)

        assert response.response == "Test response"
        assert response.chain_of_thought == "Test chain of thought"
