"""
Tests for the Command Handler Registry system.

These tests verify that the command registry can properly handle all command types
and that the handlers produce the same results as the original match/case logic.
"""

import pytest
from unittest.mock import Mock

from src.agent.domain import commands, events
from src.agent.utils.command_registry import CommandHandlerRegistry
from src.agent.utils.command_handlers import (
    QuestionHandler,
    CheckHandler,
    RetrieveHandler,
    RerankHandler,
    EnhanceHandler,
    UseToolsHandler,
    LLMResponseHandler,
    FinalCheckHandler,
)


class TestCommandHandlerRegistry:
    """Test the command registry infrastructure."""

    def test_registry_can_register_and_retrieve_handlers(self):
        """Test that handlers can be registered and retrieved."""
        registry = CommandHandlerRegistry()
        handler = QuestionHandler()

        registry.register(commands.Question, handler)

        question_cmd = commands.Question(question="test", q_id="test")
        retrieved_handler = registry.get_handler(question_cmd)

        assert retrieved_handler is handler

    def test_registry_raises_error_for_unregistered_command(self):
        """Test that unregistered commands raise NotImplementedError."""
        registry = CommandHandlerRegistry()
        unknown_cmd = commands.Question(question="test", q_id="test")

        with pytest.raises(
            NotImplementedError,
            match="No handler registered for command type: Question",
        ):
            registry.process(unknown_cmd, Mock())

    def test_registry_can_clear_all_handlers(self):
        """Test that all handlers can be cleared."""
        registry = CommandHandlerRegistry()
        registry.register(commands.Question, QuestionHandler())

        registry.clear()

        assert len(registry.get_registered_types()) == 0


class TestQuestionHandler:
    """Test the Question command handler."""

    def test_question_handler_can_handle_question_commands(self):
        """Test that QuestionHandler can handle Question commands."""
        handler = QuestionHandler()
        question_cmd = commands.Question(question="test", q_id="test")

        assert handler.can_handle(question_cmd) is True

    def test_question_handler_produces_check_command(self):
        """Test that QuestionHandler produces a Check command like the original logic."""
        handler = QuestionHandler()
        question_cmd = commands.Question(question="test query", q_id="test session id")

        # Create a mock agent with the necessary attributes
        mock_agent = Mock()
        mock_agent.prepare_guardrails_check.return_value = commands.Check(
            question="test guardrails pre check", q_id="test session id"
        )

        result = handler.handle(question_cmd, mock_agent)

        assert isinstance(result, commands.Check)
        assert result.question == "test guardrails pre check"
        assert result.q_id == "test session id"
        mock_agent.prepare_guardrails_check.assert_called_once_with(question_cmd)


class TestCheckHandler:
    """Test the Check command handler."""

    def test_check_handler_can_handle_check_commands(self):
        """Test that CheckHandler can handle Check commands."""
        handler = CheckHandler()
        check_cmd = commands.Check(question="test", q_id="test", approved=True)

        assert handler.can_handle(check_cmd) is True

    def test_check_handler_produces_retrieve_command_when_approved(self):
        """Test that CheckHandler produces a Retrieve command when approved."""
        handler = CheckHandler()
        check_cmd = commands.Check(question="test", q_id="test", approved=True)

        mock_agent = Mock()
        mock_agent.prepare_retrieval.return_value = commands.Retrieve(
            question="test", q_id="test"
        )

        result = handler.handle(check_cmd, mock_agent)

        assert isinstance(result, commands.Retrieve)
        mock_agent.prepare_retrieval.assert_called_once_with(check_cmd)

    def test_check_handler_produces_rejected_request_when_not_approved(self):
        """Test that CheckHandler produces a RejectedRequest when not approved."""
        handler = CheckHandler()
        check_cmd = commands.Check(
            question="test", q_id="test", approved=False, response="rejected"
        )

        mock_agent = Mock()
        mock_agent.prepare_retrieval.return_value = events.RejectedRequest(
            question="test", response="rejected", q_id="test"
        )

        result = handler.handle(check_cmd, mock_agent)

        assert isinstance(result, events.RejectedRequest)
        mock_agent.prepare_retrieval.assert_called_once_with(check_cmd)


class TestRetrieveHandler:
    """Test the Retrieve command handler."""

    def test_retrieve_handler_can_handle_retrieve_commands(self):
        """Test that RetrieveHandler can handle Retrieve commands."""
        handler = RetrieveHandler()
        retrieve_cmd = commands.Retrieve(question="test", q_id="test")

        assert handler.can_handle(retrieve_cmd) is True

    def test_retrieve_handler_produces_rerank_command(self):
        """Test that RetrieveHandler produces a Rerank command."""
        handler = RetrieveHandler()
        retrieve_cmd = commands.Retrieve(question="test", q_id="test", candidates=[])

        mock_agent = Mock()
        mock_agent.prepare_rerank.return_value = commands.Rerank(
            question="test", q_id="test", candidates=[]
        )

        result = handler.handle(retrieve_cmd, mock_agent)

        assert isinstance(result, commands.Rerank)
        mock_agent.prepare_rerank.assert_called_once_with(retrieve_cmd)


class TestRerankHandler:
    """Test the Rerank command handler."""

    def test_rerank_handler_can_handle_rerank_commands(self):
        """Test that RerankHandler can handle Rerank commands."""
        handler = RerankHandler()
        rerank_cmd = commands.Rerank(question="test", q_id="test")

        assert handler.can_handle(rerank_cmd) is True

    def test_rerank_handler_produces_enhance_command(self):
        """Test that RerankHandler produces an Enhance command."""
        handler = RerankHandler()
        rerank_cmd = commands.Rerank(question="test", q_id="test", candidates=[])

        mock_agent = Mock()
        mock_agent.prepare_enhancement.return_value = commands.Enhance(
            question="test prompt", q_id="test"
        )

        result = handler.handle(rerank_cmd, mock_agent)

        assert isinstance(result, commands.Enhance)
        mock_agent.prepare_enhancement.assert_called_once_with(rerank_cmd)


class TestEnhanceHandler:
    """Test the Enhance command handler."""

    def test_enhance_handler_can_handle_enhance_commands(self):
        """Test that EnhanceHandler can handle Enhance commands."""
        handler = EnhanceHandler()
        enhance_cmd = commands.Enhance(question="test", q_id="test")

        assert handler.can_handle(enhance_cmd) is True

    def test_enhance_handler_produces_use_tools_command(self):
        """Test that EnhanceHandler produces a UseTools command."""
        handler = EnhanceHandler()
        enhance_cmd = commands.Enhance(
            question="test", q_id="test", response="enhanced"
        )

        mock_agent = Mock()
        mock_agent.prepare_agent_call.return_value = commands.UseTools(
            question="enhanced", q_id="test"
        )

        result = handler.handle(enhance_cmd, mock_agent)

        assert isinstance(result, commands.UseTools)
        mock_agent.prepare_agent_call.assert_called_once_with(enhance_cmd)


class TestUseToolsHandler:
    """Test the UseTools command handler."""

    def test_use_tools_handler_can_handle_use_tools_commands(self):
        """Test that UseToolsHandler can handle UseTools commands."""
        handler = UseToolsHandler()
        use_tools_cmd = commands.UseTools(question="test", q_id="test")

        assert handler.can_handle(use_tools_cmd) is True

    def test_use_tools_handler_produces_llm_response_command(self):
        """Test that UseToolsHandler produces an LLMResponse command."""
        handler = UseToolsHandler()
        use_tools_cmd = commands.UseTools(
            question="test", q_id="test", response="response"
        )

        mock_agent = Mock()
        mock_agent.prepare_finalization.return_value = commands.LLMResponse(
            question="test prompt", q_id="test"
        )

        result = handler.handle(use_tools_cmd, mock_agent)

        assert isinstance(result, commands.LLMResponse)
        mock_agent.prepare_finalization.assert_called_once_with(use_tools_cmd)


class TestLLMResponseHandler:
    """Test the LLMResponse command handler."""

    def test_llm_response_handler_can_handle_llm_response_commands(self):
        """Test that LLMResponseHandler can handle LLMResponse commands."""
        handler = LLMResponseHandler()
        llm_response_cmd = commands.LLMResponse(question="test", q_id="test")

        assert handler.can_handle(llm_response_cmd) is True

    def test_llm_response_handler_produces_final_check_command(self):
        """Test that LLMResponseHandler produces a FinalCheck command."""
        handler = LLMResponseHandler()
        llm_response_cmd = commands.LLMResponse(
            question="test", q_id="test", response="response"
        )

        mock_agent = Mock()
        mock_agent.prepare_response.return_value = commands.FinalCheck(
            question="test guardrails post check", q_id="test"
        )

        result = handler.handle(llm_response_cmd, mock_agent)

        assert isinstance(result, commands.FinalCheck)
        mock_agent.prepare_response.assert_called_once_with(llm_response_cmd)


class TestFinalCheckHandler:
    """Test the FinalCheck command handler."""

    def test_final_check_handler_can_handle_final_check_commands(self):
        """Test that FinalCheckHandler can handle FinalCheck commands."""
        handler = FinalCheckHandler()
        final_check_cmd = commands.FinalCheck(question="test", q_id="test")

        assert handler.can_handle(final_check_cmd) is True

    def test_final_check_handler_produces_none_command(self):
        """Test that FinalCheckHandler produces None (end of chain)."""
        handler = FinalCheckHandler()
        final_check_cmd = commands.FinalCheck(
            question="test", q_id="test", approved=True
        )

        mock_agent = Mock()
        mock_agent.prepare_evaluation.return_value = None

        result = handler.handle(final_check_cmd, mock_agent)

        assert result is None
        mock_agent.prepare_evaluation.assert_called_once_with(final_check_cmd)


class TestRegistryIntegration:
    """Integration tests for the complete registry system."""

    def test_full_command_flow_through_registry(self):
        """Test that the registry can handle a complete command flow."""
        registry = CommandHandlerRegistry()

        # Register all handlers
        registry.register(commands.Question, QuestionHandler())
        registry.register(commands.Check, CheckHandler())
        registry.register(commands.Retrieve, RetrieveHandler())
        registry.register(commands.Rerank, RerankHandler())
        registry.register(commands.Enhance, EnhanceHandler())
        registry.register(commands.UseTools, UseToolsHandler())
        registry.register(commands.LLMResponse, LLMResponseHandler())
        registry.register(commands.FinalCheck, FinalCheckHandler())

        # Create a mock agent
        mock_agent = Mock()

        # Test Question -> Check
        question_cmd = commands.Question(question="test", q_id="test")
        mock_agent.prepare_guardrails_check.return_value = commands.Check(
            question="check", q_id="test"
        )

        result = registry.process(question_cmd, mock_agent)

        assert isinstance(result, commands.Check)
        mock_agent.prepare_guardrails_check.assert_called_once_with(question_cmd)

        # Verify all command types are registered
        expected_types = {
            commands.Question,
            commands.Check,
            commands.Retrieve,
            commands.Rerank,
            commands.Enhance,
            commands.UseTools,
            commands.LLMResponse,
            commands.FinalCheck,
        }
        assert set(registry.get_registered_types()) == expected_types
