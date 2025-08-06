"""
Tests for BaseAgent prompt methods.

This module tests the refactored prompt creation methods that replace the
large create_prompt() method with smaller, focused methods.
"""

import pytest
from unittest.mock import patch

from src.agent.domain import commands
from src.agent.domain.model import BaseAgent


class TestBaseAgentPromptMethods:
    """Test suite for BaseAgent prompt methods."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("src.agent.domain.model.yaml.safe_load"):
            with patch("builtins.open"):
                question = commands.Question(question="Test question", q_id="test-id")
                self.agent = BaseAgent(question, kwargs={"prompt_path": "test.yaml"})

        # Mock base prompts
        self.agent.base_prompts = {
            "finalize": "Finalize template: {{ question }} - {{ response }}",
            "enhance": "Enhance template: {{ question }} - {{ information }}",
            "guardrails": {
                "pre_check": "Pre-check template: {{ question }}",
                "post_check": "Post-check template: {{ question }} - {{ response }} - {{ memory }}",
            },
        }

    def test_get_prompt_template_returns_correct_template_for_use_tools(self):
        """Test that get_prompt_template returns finalize template for UseTools commands."""
        command = commands.UseTools(question="test", response="test", q_id="test")

        template = self.agent._get_prompt_template(command)

        assert template == "Finalize template: {{ question }} - {{ response }}"

    def test_get_prompt_template_returns_correct_template_for_rerank(self):
        """Test that get_prompt_template returns enhance template for Rerank commands."""
        command = commands.Rerank(question="test", q_id="test", candidates=[])

        template = self.agent._get_prompt_template(command)

        assert template == "Enhance template: {{ question }} - {{ information }}"

    def test_get_prompt_template_returns_correct_template_for_question(self):
        """Test that get_prompt_template returns pre_check template for Question commands."""
        command = commands.Question(question="test", q_id="test")

        template = self.agent._get_prompt_template(command)

        assert template == "Pre-check template: {{ question }}"

    def test_get_prompt_template_returns_correct_template_for_llm_response(self):
        """Test that get_prompt_template returns post_check template for LLMResponse commands."""
        command = commands.LLMResponse(question="test", response="test", q_id="test")

        template = self.agent._get_prompt_template(command)

        assert (
            template
            == "Post-check template: {{ question }} - {{ response }} - {{ memory }}"
        )

    def test_get_prompt_template_raises_error_for_invalid_command(self):
        """Test that get_prompt_template raises ValueError for invalid commands."""
        command = commands.Check(question="test", q_id="test")  # Unsupported command

        with pytest.raises(ValueError, match="Invalid command type"):
            self.agent._get_prompt_template(command)

    def test_get_prompt_template_raises_error_when_template_not_found(self):
        """Test that get_prompt_template raises ValueError when template is None."""
        # Remove the template from base_prompts
        self.agent.base_prompts = {}
        command = commands.UseTools(question="test", response="test", q_id="test")

        with pytest.raises(ValueError, match="Prompt not found"):
            self.agent._get_prompt_template(command)

    def test_get_prompt_variables_for_use_tools_returns_correct_variables(self):
        """Test that _get_prompt_variables returns correct variables for UseTools."""
        command = commands.UseTools(
            question="Test question", response="Test response", q_id="test"
        )

        variables = self.agent._get_prompt_variables(command, memory=None)

        expected = {
            "question": "Test question",
            "response": "Test response",
        }
        assert variables == expected

    def test_get_prompt_variables_for_rerank_returns_correct_variables(self):
        """Test that _get_prompt_variables returns correct variables for Rerank."""
        from src.agent.domain.commands import KBResponse

        candidate = KBResponse(
            description="test candidate",
            score=0.9,
            id="test-1",
            tag="test",
            name="test",
        )
        command = commands.Rerank(
            question="Test question", q_id="test", candidates=[candidate]
        )

        variables = self.agent._get_prompt_variables(command, memory=None)

        expected = {
            "question": "Test question",
            "information": '[{"description": "test candidate", "score": 0.9, "id": "test-1", "tag": "test", "name": "test"}]',
        }
        assert variables == expected

    def test_get_prompt_variables_for_question_returns_correct_variables(self):
        """Test that _get_prompt_variables returns correct variables for Question."""
        command = commands.Question(question="Test question", q_id="test")

        variables = self.agent._get_prompt_variables(command, memory=None)

        expected = {"question": "Test question"}
        assert variables == expected

    def test_get_prompt_variables_for_llm_response_returns_correct_variables(self):
        """Test that _get_prompt_variables returns correct variables for LLMResponse."""
        command = commands.LLMResponse(
            question="Test question", response="Test response", q_id="test"
        )
        memory = ["memory1", "memory2"]

        variables = self.agent._get_prompt_variables(command, memory)

        expected = {
            "question": "Test question",
            "response": "Test response",
            "memory": "memory1\nmemory2",
        }
        assert variables == expected

    def test_get_prompt_variables_for_llm_response_with_none_memory(self):
        """Test that _get_prompt_variables handles None memory for LLMResponse."""
        command = commands.LLMResponse(
            question="Test question", response="Test response", q_id="test"
        )

        variables = self.agent._get_prompt_variables(command, memory=None)

        expected = {
            "question": "Test question",
            "response": "Test response",
            "memory": "",
        }
        assert variables == expected

    def test_get_prompt_variables_raises_error_for_invalid_command(self):
        """Test that _get_prompt_variables raises ValueError for invalid commands."""
        command = commands.Check(question="test", q_id="test")  # Unsupported command

        with pytest.raises(ValueError, match="Invalid command type"):
            self.agent._get_prompt_variables(command, memory=None)

    @patch("src.agent.domain.model.populate_template")
    def test_create_prompt_integrates_all_methods_correctly(self, mock_populate):
        """Test that create_prompt integrates all helper methods correctly."""
        mock_populate.return_value = "Final prompt"
        command = commands.UseTools(
            question="Test question", response="Test response", q_id="test"
        )

        result = self.agent.create_prompt(command)

        # Verify populate_template was called with correct arguments
        mock_populate.assert_called_once_with(
            "Finalize template: {{ question }} - {{ response }}",
            {"question": "Test question", "response": "Test response"},
        )
        assert result == "Final prompt"

    @patch("src.agent.domain.model.populate_template")
    def test_create_prompt_backward_compatibility(self, mock_populate):
        """Test that create_prompt maintains backward compatibility with existing interface."""
        mock_populate.return_value = "Processed prompt"
        command = commands.Question(question="Test question", q_id="test")

        # This should work exactly as before
        result = self.agent.create_prompt(command)

        mock_populate.assert_called_once()
        assert result == "Processed prompt"


class TestPromptMethodsIntegration:
    """Integration tests for prompt methods with real templates."""

    def setup_method(self):
        """Set up test fixtures with realistic prompt templates."""
        with patch("src.agent.domain.model.yaml.safe_load"):
            with patch("builtins.open"):
                question = commands.Question(
                    question="What is the status of order 123?", q_id="order-123"
                )
                self.agent = BaseAgent(question, kwargs={"prompt_path": "test.yaml"})

        # Use realistic prompt templates
        self.agent.base_prompts = {
            "finalize": "Based on the question: {{ question }} and the information: {{ response }}, provide a comprehensive answer.",
            "enhance": "Given the question: {{ question }} and the following information: {{ information }}, enhance the question for better results.",
            "guardrails": {
                "pre_check": "Please check if this question is appropriate: {{ question }}",
                "post_check": "Review this response for accuracy: Question: {{ question }} Response: {{ response }} Context: {{ memory }}",
            },
        }

    def test_end_to_end_prompt_creation_for_use_tools(self):
        """Test complete prompt creation flow for UseTools command."""
        command = commands.UseTools(
            question="What is the status of order 123?",
            response="Order data retrieved from database",
            q_id="order-123",
        )

        result = self.agent.create_prompt(command)

        expected = "Based on the question: What is the status of order 123? and the information: Order data retrieved from database, provide a comprehensive answer."
        assert result == expected

    def test_end_to_end_prompt_creation_for_llm_response_with_memory(self):
        """Test complete prompt creation flow for LLMResponse command with memory."""
        command = commands.LLMResponse(
            question="What is the status of order 123?",
            response="Order 123 is currently being processed",
            q_id="order-123",
        )
        memory = ["Customer called yesterday", "Order placed on Monday"]

        result = self.agent.create_prompt(command, memory)

        expected = "Review this response for accuracy: Question: What is the status of order 123? Response: Order 123 is currently being processed Context: Customer called yesterday\nOrder placed on Monday"
        assert result == expected
